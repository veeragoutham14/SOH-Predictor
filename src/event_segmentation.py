from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.config import SignalColumnConfig


EVENT_OUTPUT_COLUMNS = (
    "event_id",
    "serial",
    "mode",
    "start_timestamp",
    "end_timestamp",
    "duration_seconds",
    "num_rows",
    "start_soc",
    "end_soc",
    "mean_soc",
    "start_soh",
    "end_soh",
    "mean_soh",
    "mean_current_a",
    "mean_voltage_v",
    "throughput_ah",
    "energy_wh",
)

MODE_MISSING = "missing"
EVENT_ID_PREFIX_BY_MODE = {
    "rest": "r",
    "charging": "c",
    "discharging": "d",
    MODE_MISSING: "missing",
}


def validate_event_input_columns(df: pd.DataFrame, columns: SignalColumnConfig) -> None:
    """Ensure all columns needed for event summaries and integration exist."""
    required = (
        columns.timestamp_col,
        columns.serial_col,
        columns.mode_col,
        columns.current_col,
        columns.voltage_col,
        columns.soc_col,
        columns.soh_col,
    )
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required event-segmentation columns: {missing}")


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _effective_abs_current(current: pd.Series, threshold: float) -> pd.Series:
    abs_current = pd.to_numeric(current, errors="coerce").abs()
    return abs_current.where(abs_current > threshold, 0.0).fillna(0.0)


def _event_id_for_mode(mode: str, position: int) -> str:
    prefix = EVENT_ID_PREFIX_BY_MODE.get(str(mode), str(mode).lower().replace(" ", "_"))
    return f"event_{prefix}_{position}"


def _timestamp_gap_seconds(left_end: Any, right_start: Any) -> float:
    return (pd.Timestamp(right_start) - pd.Timestamp(left_end)).total_seconds()


def _build_missing_event(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    expected_sample_interval_seconds: float,
) -> dict[str, Any]:
    """Represent a missing telemetry gap between two observed events."""
    left_end = pd.Timestamp(left["end_timestamp"])
    right_start = pd.Timestamp(right["start_timestamp"])
    missing_start = left_end + pd.Timedelta(seconds=expected_sample_interval_seconds)
    if missing_start > right_start:
        missing_start = left_end
    missing_duration_seconds = max(0.0, (right_start - missing_start).total_seconds())

    return {
        "serial": left["serial"],
        "mode": MODE_MISSING,
        "start_timestamp": missing_start,
        "end_timestamp": right_start,
        "duration_seconds": missing_duration_seconds,
        "num_rows": 0,
        "start_soc": left["end_soc"],
        "end_soc": right["start_soc"],
        "mean_soc": float("nan"),
        "start_soh": left["end_soh"],
        "end_soh": right["start_soh"],
        "mean_soh": float("nan"),
        "mean_current_a": float("nan"),
        "mean_voltage_v": float("nan"),
        "throughput_ah": float("nan"),
        "energy_wh": float("nan"),
    }


def insert_missing_gap_events(
    events: pd.DataFrame,
    *,
    missing_gap_threshold_seconds: float,
    expected_sample_interval_seconds: float,
) -> pd.DataFrame:
    """Insert explicit missing-data events between observed events with large gaps."""
    if events.empty:
        return events

    rows: list[dict[str, Any]] = []
    previous_by_serial: dict[str, dict[str, Any]] = {}

    for event in events.to_dict(orient="records"):
        serial_key = str(event["serial"])
        previous = previous_by_serial.get(serial_key)

        if previous is not None:
            gap_seconds = _timestamp_gap_seconds(previous["end_timestamp"], event["start_timestamp"])
            if gap_seconds > missing_gap_threshold_seconds:
                rows.append(
                    _build_missing_event(
                        previous,
                        event,
                        expected_sample_interval_seconds=expected_sample_interval_seconds,
                    )
                )

        rows.append(event)
        previous_by_serial[serial_key] = event

    return pd.DataFrame(rows)


def prepare_classified_frame(
    df: pd.DataFrame,
    columns: SignalColumnConfig,
) -> pd.DataFrame:
    """Normalize types and sort classified telemetry before segmentation."""
    validate_event_input_columns(df, columns)

    work = df.copy()
    work[columns.timestamp_col] = pd.to_datetime(
        work[columns.timestamp_col],
        errors="coerce",
        utc=True,
    )
    work = work.dropna(subset=[columns.timestamp_col, columns.serial_col, columns.mode_col])
    work = work.sort_values(
        [columns.serial_col, columns.timestamp_col],
        kind="mergesort",
    ).reset_index(drop=True)
    return work


def summarize_classified_events(
    df: pd.DataFrame,
    columns: SignalColumnConfig,
    *,
    rest_current_threshold_a: float | None = None,
) -> pd.DataFrame:
    """Create event summaries from contiguous rows with the same operating mode."""
    if df.empty:
        return pd.DataFrame(columns=EVENT_OUTPUT_COLUMNS[1:])

    threshold = (
        columns.rest_current_threshold_a
        if rest_current_threshold_a is None
        else rest_current_threshold_a
    )
    missing_gap_threshold_seconds = columns.missing_gap_threshold_seconds
    work = prepare_classified_frame(df, columns)
    if work.empty:
        return pd.DataFrame(columns=EVENT_OUTPUT_COLUMNS[1:])

    current = _numeric_series(work, columns.current_col)
    voltage = _numeric_series(work, columns.voltage_col)
    soc = _numeric_series(work, columns.soc_col)
    soh = _numeric_series(work, columns.soh_col)

    work["_current_a"] = current
    work["_voltage_v"] = voltage
    work["_soc_pct"] = soc
    work["_soh_pct"] = soh

    timestamp = work[columns.timestamp_col]
    previous_serial = work[columns.serial_col].shift()
    previous_mode = work[columns.mode_col].shift()
    previous_timestamp = timestamp.shift()

    delta_seconds = (timestamp - previous_timestamp).dt.total_seconds()
    same_serial = work[columns.serial_col].eq(previous_serial)
    valid_delta_seconds = delta_seconds.where(
        same_serial
        & (delta_seconds > 0)
        & (delta_seconds <= missing_gap_threshold_seconds),
        0.0,
    ).fillna(0.0)
    delta_hours = valid_delta_seconds / 3600.0

    effective_current = _effective_abs_current(current, threshold)
    safe_voltage = voltage.fillna(0.0)
    work["_throughput_ah"] = effective_current * delta_hours
    work["_energy_wh"] = safe_voltage * effective_current * delta_hours

    mode_break = (
        work[columns.serial_col].ne(previous_serial)
        | work[columns.mode_col].ne(previous_mode)
        | (same_serial & delta_seconds.notna() & (delta_seconds <= 0))
        | (same_serial & delta_seconds.notna() & (delta_seconds > missing_gap_threshold_seconds))
    )
    work["_event_key"] = mode_break.cumsum()

    grouped = work.groupby("_event_key", sort=False)
    events = grouped.agg(
        serial=(columns.serial_col, "first"),
        mode=(columns.mode_col, "first"),
        start_timestamp=(columns.timestamp_col, "first"),
        end_timestamp=(columns.timestamp_col, "last"),
        num_rows=(columns.timestamp_col, "size"),
        start_soc=("_soc_pct", "first"),
        end_soc=("_soc_pct", "last"),
        mean_soc=("_soc_pct", "mean"),
        start_soh=("_soh_pct", "first"),
        end_soh=("_soh_pct", "last"),
        mean_soh=("_soh_pct", "mean"),
        mean_current_a=("_current_a", "mean"),
        mean_voltage_v=("_voltage_v", "mean"),
        throughput_ah=("_throughput_ah", "sum"),
        energy_wh=("_energy_wh", "sum"),
    ).reset_index(drop=True)

    events["duration_seconds"] = (
        events["end_timestamp"] - events["start_timestamp"]
    ).dt.total_seconds().clip(lower=0)

    events = events[
        [
            "serial",
            "mode",
            "start_timestamp",
            "end_timestamp",
            "duration_seconds",
            "num_rows",
            "start_soc",
            "end_soc",
            "mean_soc",
            "start_soh",
            "end_soh",
            "mean_soh",
            "mean_current_a",
            "mean_voltage_v",
            "throughput_ah",
            "energy_wh",
        ]
    ]
    return insert_missing_gap_events(
        events,
        missing_gap_threshold_seconds=missing_gap_threshold_seconds,
        expected_sample_interval_seconds=columns.expected_sample_interval_seconds,
    )


def _weighted_mean(left_value: Any, left_weight: int, right_value: Any, right_weight: int) -> float:
    left = pd.to_numeric(pd.Series([left_value]), errors="coerce").iloc[0]
    right = pd.to_numeric(pd.Series([right_value]), errors="coerce").iloc[0]

    total = 0.0
    weight = 0
    if pd.notna(left):
        total += float(left) * left_weight
        weight += left_weight
    if pd.notna(right):
        total += float(right) * right_weight
        weight += right_weight
    return float("nan") if weight == 0 else total / weight


def _bridge_throughput(
    left: dict[str, Any],
    right: dict[str, Any],
    threshold: float,
) -> tuple[float, float]:
    start = pd.Timestamp(right["start_timestamp"])
    end = pd.Timestamp(left["end_timestamp"])
    delta_seconds = (start - end).total_seconds()
    if delta_seconds <= 0:
        return 0.0, 0.0

    current = abs(float(right["mean_current_a"])) if pd.notna(right["mean_current_a"]) else 0.0
    if current <= threshold:
        current = 0.0

    voltage = float(right["mean_voltage_v"]) if pd.notna(right["mean_voltage_v"]) else 0.0
    delta_hours = delta_seconds / 3600.0
    return current * delta_hours, voltage * current * delta_hours


def merge_adjacent_events(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    rest_current_threshold_a: float,
) -> dict[str, Any]:
    """Merge two same-serial, same-mode events split by file/chunk boundaries."""
    left_rows = int(left["num_rows"])
    right_rows = int(right["num_rows"])
    total_rows = left_rows + right_rows
    bridge_ah, bridge_wh = _bridge_throughput(left, right, rest_current_threshold_a)

    merged = dict(left)
    merged["end_timestamp"] = right["end_timestamp"]
    merged["duration_seconds"] = max(
        0.0,
        (pd.Timestamp(right["end_timestamp"]) - pd.Timestamp(left["start_timestamp"])).total_seconds(),
    )
    merged["num_rows"] = total_rows
    merged["end_soc"] = right["end_soc"]
    merged["end_soh"] = right["end_soh"]
    merged["mean_soc"] = _weighted_mean(left["mean_soc"], left_rows, right["mean_soc"], right_rows)
    merged["mean_soh"] = _weighted_mean(left["mean_soh"], left_rows, right["mean_soh"], right_rows)
    merged["mean_current_a"] = _weighted_mean(
        left["mean_current_a"],
        left_rows,
        right["mean_current_a"],
        right_rows,
    )
    merged["mean_voltage_v"] = _weighted_mean(
        left["mean_voltage_v"],
        left_rows,
        right["mean_voltage_v"],
        right_rows,
    )
    merged["throughput_ah"] = float(left["throughput_ah"]) + float(right["throughput_ah"]) + bridge_ah
    merged["energy_wh"] = float(left["energy_wh"]) + float(right["energy_wh"]) + bridge_wh
    return merged


@dataclass
class EventMerger:
    """Stateful merger for event summaries produced file by file."""

    rest_current_threshold_a: float
    missing_gap_threshold_seconds: float
    expected_sample_interval_seconds: float
    completed_events: list[dict[str, Any]] = field(default_factory=list)
    open_events_by_serial: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_events(self, events: pd.DataFrame) -> None:
        for event in events.to_dict(orient="records"):
            serial_key = str(event["serial"])
            open_event = self.open_events_by_serial.get(serial_key)

            if open_event is not None:
                gap_seconds = _timestamp_gap_seconds(open_event["end_timestamp"], event["start_timestamp"])
                if gap_seconds > self.missing_gap_threshold_seconds:
                    self.completed_events.append(open_event)
                    self.completed_events.append(
                        _build_missing_event(
                            open_event,
                            event,
                            expected_sample_interval_seconds=self.expected_sample_interval_seconds,
                        )
                    )
                    self.open_events_by_serial[serial_key] = event
                    continue

            if (
                open_event is not None
                and open_event["serial"] == event["serial"]
                and open_event["mode"] == event["mode"]
                and event["mode"] != MODE_MISSING
            ):
                self.open_events_by_serial[serial_key] = merge_adjacent_events(
                    open_event,
                    event,
                    rest_current_threshold_a=self.rest_current_threshold_a,
                )
                continue

            if open_event is not None:
                self.completed_events.append(open_event)
            self.open_events_by_serial[serial_key] = event

    def to_frame(self, event_id_col: str = "event_id") -> pd.DataFrame:
        all_events = self.completed_events + list(self.open_events_by_serial.values())
        if not all_events:
            return pd.DataFrame(columns=EVENT_OUTPUT_COLUMNS)

        events = pd.DataFrame(all_events)
        events = events.sort_values(["serial", "start_timestamp"], kind="mergesort").reset_index(drop=True)
        event_ids = [
            _event_id_for_mode(mode, position)
            for position, mode in enumerate(events["mode"])
        ]
        events.insert(0, event_id_col, event_ids)
        if event_id_col != "event_id":
            ordered = (event_id_col,) + tuple(col for col in EVENT_OUTPUT_COLUMNS[1:])
            return events[list(ordered)]
        return events[list(EVENT_OUTPUT_COLUMNS)]
