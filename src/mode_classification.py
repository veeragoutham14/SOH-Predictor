from __future__ import annotations

import pandas as pd

from src.config import SignalColumnConfig


MODE_CHARGING = "charging"
MODE_DISCHARGING = "discharging"
MODE_REST = "rest"
MODE_NAMES = (MODE_CHARGING, MODE_DISCHARGING, MODE_REST)


def validate_mode_input_columns(df: pd.DataFrame, columns: SignalColumnConfig) -> None:
    """Ensure the columns needed for row-level mode classification exist."""
    required = (columns.timestamp_col, columns.serial_col, columns.current_col)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required mode-classification columns: {missing}")


def classify_operating_modes(
    df: pd.DataFrame,
    columns: SignalColumnConfig,
    *,
    rest_current_threshold_a: float | None = None,
) -> pd.DataFrame:
    """Classify each telemetry row as charging, discharging, or rest.

    Current is the primary signal:
    - current > threshold means charging
    - current < -threshold means discharging
    - otherwise the row is treated as rest
    """
    validate_mode_input_columns(df, columns)
    threshold = (
        columns.rest_current_threshold_a
        if rest_current_threshold_a is None
        else rest_current_threshold_a
    )
    if threshold < 0:
        raise ValueError("rest_current_threshold_a must be nonnegative.")

    classified = df.copy()
    current = pd.to_numeric(classified[columns.current_col], errors="coerce")

    mode = pd.Series(MODE_REST, index=classified.index, dtype="object")
    mode[current > threshold] = MODE_CHARGING
    mode[current < -threshold] = MODE_DISCHARGING
    classified[columns.mode_col] = mode

    return classified


def mode_counts(df: pd.DataFrame, mode_col: str) -> dict[str, int]:
    """Return a compact mode distribution for logging."""
    if mode_col not in df.columns:
        return {}
    counts = df[mode_col].value_counts(dropna=False)
    return {str(mode): int(count) for mode, count in counts.items()}
