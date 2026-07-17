from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import CapacityModelConfig, StorageConfig
from src.io_utils import write_parquet_chunk
from src.logging_utils import configure_logging


logger = logging.getLogger(__name__)

DEFAULT_SCENARIOS = {
    "low_usage": 0.7,
    "normal_usage": 1.0,
    "high_usage": 1.3,
}

DEFAULT_USAGE_RATE_MODES = (
    "historical_mean",
    "historical_median",
    "recent_median",
)


@dataclass(frozen=True)
class FittedTrendModel:
    model_name: str
    target_column: str
    feature_columns: tuple[str, ...]
    intercept: float
    coefficients: tuple[float, ...]
    train_rows: int
    train_rmse: float
    train_mae: float
    holdout_rows: int
    holdout_rmse: float
    holdout_mae: float
    residual_std: float


@dataclass(frozen=True)
class FittedDegradationModel:
    model_name: str
    target_column: str
    feature_column: str
    baseline_capacity_ah: float
    reference_cumulative_ah: float
    intercept_loss_ah: float
    loss_slope_per_1000ah: float
    train_rows: int
    train_rmse: float
    train_mae: float
    holdout_rows: int
    holdout_rmse: float
    holdout_mae: float
    residual_std: float
    calendar_loss_per_day: float = 0.0
    reference_calendar_age_days: float = 0.0
    baseline_capacity_method: str = ""


@dataclass(frozen=True)
class CapacityMLResult:
    training_rows: int
    valid_training_rows: int
    forecast_rows: int
    best_capacity_ah_model: str | None
    output_dir: Path


def parse_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    """Parse a timestamp and return UTC.

    Naive values are treated as UTC because the local Parquet/Grafana workflow
    has been aligned to UTC.
    """
    timestamp = pd.to_datetime(value, errors="raise")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def find_latest_capacity_run(capacity_trend_dir: Path, serial: str | int) -> Path:
    """Return the latest capacity_run directory for a serial."""
    serial_dir = capacity_trend_dir / f"serial={serial}"
    if not serial_dir.exists():
        raise FileNotFoundError(f"No capacity trend directory found: {serial_dir}")

    runs = sorted(path for path in serial_dir.glob("capacity_run=*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No capacity_run folders found under: {serial_dir}")
    return runs[-1]


def load_capacity_run(capacity_run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load capacity measurements and optional discharge usage ledger."""
    if capacity_run_dir.is_file():
        measurements_path = capacity_run_dir
        run_dir = capacity_run_dir.parent
    else:
        run_dir = capacity_run_dir
        measurements_path = run_dir / "capacity_measurements.parquet"

    if not measurements_path.exists():
        raise FileNotFoundError(f"Missing capacity measurements: {measurements_path}")

    usage_path = run_dir / "discharge_usage_ledger.parquet"
    measurements = pd.read_parquet(measurements_path)
    usage_ledger = pd.read_parquet(usage_path) if usage_path.exists() else pd.DataFrame()
    return measurements, usage_ledger


def resolve_nominal_capacity_ah(
    measurements: pd.DataFrame,
    nominal_capacity_ah: float | None,
) -> tuple[float, str]:
    """Resolve SOH reference capacity from override or discharge event 1."""
    if nominal_capacity_ah is not None:
        if nominal_capacity_ah <= 0:
            raise ValueError("nominal_capacity_ah must be positive.")
        return float(nominal_capacity_ah), "explicit"

    required_columns = {"event_discharge_id", "capacity_ah"}
    missing_columns = required_columns.difference(measurements.columns)
    if measurements.empty or missing_columns:
        raise ValueError(
            "Cannot infer nominal capacity from event_discharge_id 1: "
            f"capacity measurements are empty or missing columns {sorted(missing_columns)}."
        )

    work = measurements[["event_discharge_id", "capacity_ah"]].copy()
    work["capacity_ah"] = pd.to_numeric(work["capacity_ah"], errors="coerce")
    work["event_discharge_id"] = pd.to_numeric(
        work["event_discharge_id"],
        errors="coerce",
    )

    event_one = work[work["event_discharge_id"].eq(1)].dropna(subset=["capacity_ah"])
    event_one = event_one[event_one["capacity_ah"].gt(0)]
    if event_one.empty:
        raise ValueError(
            "Cannot infer nominal capacity: event_discharge_id 1 has no positive capacity_ah."
        )

    return float(event_one.iloc[0]["capacity_ah"]), "event_discharge_id_1_capacity"


def _as_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    work = df.copy()
    for column in columns:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    return work


def _median_abs_deviation_outlier(values: pd.Series, threshold: float = 3.5) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    mad = (numeric - median).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(False, index=values.index)
    robust_z = 0.6745 * (numeric - median) / mad
    return robust_z.abs() > threshold


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add transformed feature columns used by the candidate trend models."""
    work = df.copy()
    cumulative_ah = pd.to_numeric(work["cumulative_all_discharge_ah"], errors="coerce")
    cumulative_wh = pd.to_numeric(work["cumulative_all_discharge_wh"], errors="coerce")

    work["_usage_ah_k"] = cumulative_ah / 1000.0
    work["_usage_wh_k"] = cumulative_wh / 1000.0
    work["_usage_ah_sqrt"] = np.sqrt(cumulative_ah.clip(lower=0.0))
    work["_usage_wh_sqrt"] = np.sqrt(cumulative_wh.clip(lower=0.0))
    work["_usage_ah_log1p"] = np.log1p(cumulative_ah.clip(lower=0.0))
    work["_usage_wh_log1p"] = np.log1p(cumulative_wh.clip(lower=0.0))
    work["_usage_ah_k_sq"] = work["_usage_ah_k"] ** 2
    work["_usage_wh_k_sq"] = work["_usage_wh_k"] ** 2
    return work


def prepare_training_table(
    measurements: pd.DataFrame,
    *,
    nominal_capacity_ah: float,
    bms_soh_reliable_after: pd.Timestamp,
    min_reasonable_capacity_ah: float,
    max_reasonable_capacity_ah: float,
    exclude_statistical_outliers: bool = False,
    exclude_high_rest_ratio_capacity_rows: bool = False,
    max_rest_to_discharge_ratio_for_training: float = 1.0,
) -> pd.DataFrame:
    """Build the capacity ML table from anchor-based capacity measurements."""
    if measurements.empty:
        return pd.DataFrame()
    if max_rest_to_discharge_ratio_for_training <= 0:
        raise ValueError("max_rest_to_discharge_ratio_for_training must be greater than 0.")

    work = measurements.copy()
    work["start_anchor_time"] = pd.to_datetime(
        work["start_anchor_time"],
        errors="coerce",
        utc=True,
    )
    work["end_anchor_time"] = pd.to_datetime(
        work["end_anchor_time"],
        errors="coerce",
        utc=True,
    )
    work = _as_numeric(
        work,
        [
            "event_discharge_id",
            "capacity_ah",
            "capacity_wh",
            "cumulative_all_discharge_ah",
            "cumulative_all_discharge_wh",
            "bms_soh_start",
            "bms_soh_end",
            "discharge_duration_seconds",
            "rest_duration_seconds",
            "rest_to_discharge_ratio",
            "rest_fraction",
            "num_discharge_events",
        ],
    )
    work = work.dropna(subset=["serial", "end_anchor_time", "capacity_ah"])
    work = work.sort_values(["serial", "end_anchor_time"], kind="mergesort").reset_index(drop=True)

    if work.empty:
        return work

    first_time = work.groupby("serial")["end_anchor_time"].transform("min")
    work["calendar_age_days"] = (
        work["end_anchor_time"] - first_time
    ).dt.total_seconds() / 86400.0
    work["measured_soh_pct"] = work["capacity_ah"] / nominal_capacity_ah * 100.0
    work["bms_soh_reliable"] = work["end_anchor_time"] >= bms_soh_reliable_after
    work["bms_soh_error_pct"] = np.where(
        work["bms_soh_reliable"],
        work["bms_soh_end"] - work["measured_soh_pct"],
        np.nan,
    )

    discharge_seconds = pd.to_numeric(
        work.get(
            "discharge_duration_seconds",
            pd.Series(np.nan, index=work.index),
        ),
        errors="coerce",
    )
    rest_seconds = pd.to_numeric(
        work.get(
            "rest_duration_seconds",
            pd.Series(np.nan, index=work.index),
        ),
        errors="coerce",
    )
    total_seconds = discharge_seconds + rest_seconds
    work["rest_to_discharge_ratio"] = np.where(
        discharge_seconds.gt(0),
        rest_seconds / discharge_seconds,
        np.nan,
    )
    work["rest_fraction"] = np.where(
        total_seconds.gt(0),
        rest_seconds / total_seconds,
        np.nan,
    )
    work["capacity_measurement_duration_valid"] = (
        discharge_seconds.gt(0) & rest_seconds.ge(0)
    )
    work["long_rest_capacity_measurement"] = (
        pd.Series(work["rest_to_discharge_ratio"], index=work.index)
        .gt(max_rest_to_discharge_ratio_for_training)
        .fillna(False)
    )
    work["capacity_measurement_quality"] = np.select(
        [
            ~work["capacity_measurement_duration_valid"],
            work["long_rest_capacity_measurement"],
        ],
        [
            "invalid_duration",
            "long_rest_interrupted",
        ],
        default="clean_anchor_100_to_0",
    )
    work["rest_ratio_training_excluded"] = (
        exclude_high_rest_ratio_capacity_rows
        & (
            ~work["capacity_measurement_duration_valid"]
            | work["long_rest_capacity_measurement"]
        )
    )

    work["capacity_reasonable"] = work["capacity_ah"].between(
        min_reasonable_capacity_ah,
        max_reasonable_capacity_ah,
        inclusive="both",
    )
    work["capacity_statistical_outlier"] = _median_abs_deviation_outlier(work["capacity_ah"])
    work["has_usage_axis"] = work["cumulative_all_discharge_ah"].notna()
    work["valid_training_row"] = (
        work["capacity_reasonable"]
        & work["has_usage_axis"]
        & work["capacity_ah"].notna()
        & work["capacity_wh"].notna()
    )
    if exclude_statistical_outliers:
        work["valid_training_row"] = (
            work["valid_training_row"] & ~work["capacity_statistical_outlier"]
        )
    if exclude_high_rest_ratio_capacity_rows:
        work["valid_training_row"] = (
            work["valid_training_row"] & ~work["rest_ratio_training_excluded"]
        )

    return add_model_features(work)


def _model_features_for_target(target_column: str) -> dict[str, tuple[str, ...]]:
    if target_column == "capacity_wh":
        usage = "_usage_wh_k"
        usage_sqrt = "_usage_wh_sqrt"
        usage_log = "_usage_wh_log1p"
        usage_sq = "_usage_wh_k_sq"
    else:
        usage = "_usage_ah_k"
        usage_sqrt = "_usage_ah_sqrt"
        usage_log = "_usage_ah_log1p"
        usage_sq = "_usage_ah_k_sq"

    return {
        "event_linear": ("event_discharge_id",),
        "usage_linear": (usage,),
        "usage_sqrt": (usage_sqrt,),
        "usage_log": (usage_log,),
        "usage_quadratic": (usage, usage_sq),
        "usage_calendar_linear": (usage, "calendar_age_days"),
    }


def _prediction_from_coefficients(
    df: pd.DataFrame,
    *,
    intercept: float,
    coefficients: Sequence[float],
    feature_columns: Sequence[str],
) -> pd.Series:
    if not feature_columns:
        return pd.Series(intercept, index=df.index)
    x = df[list(feature_columns)].astype(float).to_numpy()
    return pd.Series(intercept + x @ np.asarray(coefficients, dtype=float), index=df.index)


def _predict_degradation_capacity(
    df: pd.DataFrame,
    model: FittedDegradationModel,
) -> pd.Series:
    usage = (
        pd.to_numeric(df["cumulative_all_discharge_ah"], errors="coerce")
        - model.reference_cumulative_ah
    ).clip(lower=0.0)
    usage_kah = usage / 1000.0
    if model.model_name == "degradation_usage_calendar_constrained":
        calendar_days = (
            pd.to_numeric(df["calendar_age_days"], errors="coerce")
            - model.reference_calendar_age_days
        ).clip(lower=0.0)
        predicted_loss = (
            model.intercept_loss_ah
            + model.loss_slope_per_1000ah * usage_kah
            + model.calendar_loss_per_day * calendar_days
        ).clip(lower=0.0)
        return model.baseline_capacity_ah - predicted_loss

    predicted_loss = (
        model.intercept_loss_ah
        + model.loss_slope_per_1000ah * usage_kah
    ).clip(lower=0.0)
    return model.baseline_capacity_ah - predicted_loss


def _error_metrics(actual: pd.Series, predicted: pd.Series) -> tuple[float, float, float]:
    residual = actual.astype(float) - predicted.astype(float)
    rmse = float(np.sqrt(np.mean(np.square(residual)))) if len(residual) else np.nan
    mae = float(np.mean(np.abs(residual))) if len(residual) else np.nan
    residual_std = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0
    return rmse, mae, residual_std


def _nonnegative_least_squares(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve a small nonnegative least-squares problem without extra packages."""
    n_features = design.shape[1]
    best_coefficients = np.zeros(n_features, dtype=float)
    best_sse = float(np.sum(np.square(target)))

    for mask in range(1, 1 << n_features):
        active = [index for index in range(n_features) if mask & (1 << index)]
        active_design = design[:, active]
        coefficients, *_ = np.linalg.lstsq(active_design, target, rcond=None)
        if np.any(coefficients < -1e-10):
            continue

        candidate = np.zeros(n_features, dtype=float)
        candidate[active] = np.maximum(coefficients, 0.0)
        residual = target - design @ candidate
        sse = float(np.sum(np.square(residual)))
        if sse < best_sse:
            best_sse = sse
            best_coefficients = candidate

    return best_coefficients


def _baseline_capacity_from_rows(
    fit_rows: pd.DataFrame,
    *,
    baseline_capacity_ah: float | None,
    baseline_method: str,
) -> tuple[float, str]:
    """Resolve the capacity baseline used for capacity-loss modeling."""
    if baseline_capacity_ah is not None:
        return float(baseline_capacity_ah), "explicit"

    rows = fit_rows.sort_values(["event_discharge_id", "end_anchor_time"], kind="mergesort")
    if baseline_method == "first":
        return float(rows.iloc[0]["capacity_ah"]), "first_training_capacity"
    if baseline_method == "max":
        return float(rows["capacity_ah"].max()), "max_training_capacity"
    if baseline_method == "p95":
        return float(rows["capacity_ah"].quantile(0.95)), "p95_training_capacity"

    raise ValueError(f"Unknown degradation baseline method: {baseline_method}")


def _degradation_candidate_rows(training_table: pd.DataFrame) -> pd.DataFrame:
    """Return stable calculated-capacity rows for degradation fitting.

    The degradation forecast is fitted only after the configured BMS/SOC
    reliable point. Earlier full-discharge capacity measurements are kept in
    the training table for audit and statistical models, but they are not used
    as the long-term degradation reference.
    """
    if "bms_soh_reliable" not in training_table.columns:
        raise ValueError("Missing bms_soh_reliable column for degradation fitting.")

    rows = training_table[
        training_table["valid_training_row"] & training_table["bms_soh_reliable"]
    ].dropna(
        subset=["capacity_ah", "cumulative_all_discharge_ah"]
    ).copy()
    return rows.sort_values("end_anchor_time", kind="mergesort")


def split_training_and_validation(
    training_table: pd.DataFrame,
    *,
    training_cutoff: pd.Timestamp | None,
    validation_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split capacity rows into model-training rows and future validation rows."""
    if training_cutoff is None:
        return training_table.copy(), pd.DataFrame(columns=training_table.columns)

    work = training_table.copy()
    training_rows = work[work["end_anchor_time"] <= training_cutoff].copy()
    validation_rows = work[work["end_anchor_time"] > training_cutoff].copy()
    if validation_end is not None:
        validation_rows = validation_rows[validation_rows["end_anchor_time"] <= validation_end].copy()

    return (
        training_rows.sort_values("end_anchor_time", kind="mergesort").reset_index(drop=True),
        validation_rows.sort_values("end_anchor_time", kind="mergesort").reset_index(drop=True),
    )


def filter_usage_ledger_for_training(
    usage_ledger: pd.DataFrame,
    *,
    training_cutoff: pd.Timestamp | None,
) -> pd.DataFrame:
    """Remove future discharge usage from usage-rate estimation when cut off."""
    if usage_ledger.empty or training_cutoff is None:
        return usage_ledger

    work = usage_ledger.copy()
    if "end_timestamp" not in work.columns:
        return work

    work["end_timestamp"] = pd.to_datetime(work["end_timestamp"], errors="coerce", utc=True)
    return work[work["end_timestamp"] <= training_cutoff].copy()


def resolve_forecast_validation_anchor(
    train: pd.DataFrame,
    usage_ledger_for_training: pd.DataFrame,
    *,
    training_cutoff: pd.Timestamp | None,
) -> dict[str, Any]:
    """Choose the cumulative-usage anchor for cutoff forecast validation.

    When a cutoff is provided, the post-cutoff forecast should start from known
    cumulative usage at the cutoff rather than the last full capacity measurement.
    """
    train = train.sort_values("end_anchor_time", kind="mergesort")
    last = train.iloc[-1]
    latest_capacity_ah = max(float(last["capacity_ah"]), 1e-9)
    last_time = pd.Timestamp(last["end_anchor_time"])
    last_cumulative_ah = float(last["cumulative_all_discharge_ah"])
    last_cumulative_wh = float(last["cumulative_all_discharge_wh"])

    anchor = {
        "source": "last_capacity_measurement",
        "anchor_time": last_time,
        "usage_timestamp": last_time,
        "cumulative_all_discharge_ah": last_cumulative_ah,
        "cumulative_all_discharge_wh": last_cumulative_wh,
        "event_discharge_id": float(last["event_discharge_id"]),
        "latest_capacity_ah": latest_capacity_ah,
        "last_capacity_time": last_time,
        "last_capacity_cumulative_ah": last_cumulative_ah,
        "last_capacity_cumulative_wh": last_cumulative_wh,
    }
    if training_cutoff is None or usage_ledger_for_training.empty:
        return anchor

    required_columns = {
        "end_timestamp",
        "cumulative_all_discharge_ah",
        "cumulative_all_discharge_wh",
    }
    if not required_columns.issubset(usage_ledger_for_training.columns):
        return anchor

    ledger = usage_ledger_for_training.copy()
    ledger["end_timestamp"] = pd.to_datetime(
        ledger["end_timestamp"],
        errors="coerce",
        utc=True,
    )
    ledger = _as_numeric(
        ledger,
        ["cumulative_all_discharge_ah", "cumulative_all_discharge_wh"],
    )
    ledger = ledger.dropna(
        subset=[
            "end_timestamp",
            "cumulative_all_discharge_ah",
            "cumulative_all_discharge_wh",
        ],
    )
    ledger = ledger[
        (ledger["end_timestamp"] >= last_time)
        & (ledger["end_timestamp"] <= training_cutoff)
    ].sort_values("end_timestamp", kind="mergesort")
    if ledger.empty:
        return anchor

    usage_row = ledger.iloc[-1]
    anchor_cumulative_ah = float(usage_row["cumulative_all_discharge_ah"])
    anchor_cumulative_wh = float(usage_row["cumulative_all_discharge_wh"])
    if anchor_cumulative_ah < last_cumulative_ah or anchor_cumulative_wh < last_cumulative_wh:
        return anchor

    event_increment = (anchor_cumulative_ah - last_cumulative_ah) / latest_capacity_ah
    return {
        **anchor,
        "source": "usage_ledger_training_cutoff",
        "anchor_time": pd.Timestamp(training_cutoff),
        "usage_timestamp": pd.Timestamp(usage_row["end_timestamp"]),
        "cumulative_all_discharge_ah": anchor_cumulative_ah,
        "cumulative_all_discharge_wh": anchor_cumulative_wh,
        "event_discharge_id": float(last["event_discharge_id"]) + event_increment,
    }


def _fit_degradation_model_from_rows(
    fit_rows: pd.DataFrame,
    *,
    metric_rows: pd.DataFrame | None = None,
    holdout_rows: int = 0,
    degradation_model_type: str = "usage_linear",
    baseline_capacity_ah: float | None = None,
    baseline_method: str = "max",
    baseline_rows: pd.DataFrame | None = None,
    reference_cumulative_ah: float | None = None,
    force_zero_intercept: bool = False,
) -> FittedDegradationModel | None:
    """Fit the capacity-loss model from an already-selected training window."""
    if len(fit_rows) < 2:
        return None

    fit_rows = fit_rows.sort_values("end_anchor_time", kind="mergesort")
    metric_rows = fit_rows if metric_rows is None else metric_rows

    if reference_cumulative_ah is not None and reference_cumulative_ah < 0:
        raise ValueError("reference_cumulative_ah must be nonnegative.")
    resolved_reference_cumulative_ah = (
        float(reference_cumulative_ah)
        if reference_cumulative_ah is not None
        else float(fit_rows["cumulative_all_discharge_ah"].min())
    )
    baseline_source = fit_rows if baseline_rows is None else baseline_rows
    baseline_capacity_ah, resolved_baseline_method = _baseline_capacity_from_rows(
        baseline_source,
        baseline_capacity_ah=baseline_capacity_ah,
        baseline_method=baseline_method,
    )
    reference_calendar_age_days = float(
        pd.to_numeric(fit_rows["calendar_age_days"], errors="coerce").min()
    )
    x = (
        pd.to_numeric(fit_rows["cumulative_all_discharge_ah"], errors="coerce")
        - resolved_reference_cumulative_ah
    ).clip(lower=0.0) / 1000.0
    y = (baseline_capacity_ah - fit_rows["capacity_ah"]).clip(lower=0.0)
    y_array = y.to_numpy(dtype=float)

    if degradation_model_type == "usage_linear":
        if force_zero_intercept:
            design = x.to_numpy(dtype=float).reshape(-1, 1)
            coefficients, *_ = np.linalg.lstsq(design, y_array, rcond=None)
            intercept_loss_ah = 0.0
            loss_slope_per_1000ah = max(0.0, float(coefficients[0]))
        else:
            design = np.column_stack([np.ones(len(fit_rows)), x.to_numpy(dtype=float)])
            coefficients, *_ = np.linalg.lstsq(design, y_array, rcond=None)
            intercept_loss_ah = max(0.0, float(coefficients[0]))
            loss_slope_per_1000ah = max(0.0, float(coefficients[1]))
        calendar_loss_per_day = 0.0
        model_name = "degradation_usage_linear"
        feature_column = "cumulative_all_discharge_ah_since_reference"
    elif degradation_model_type == "usage_calendar_constrained":
        calendar_days = (
            pd.to_numeric(fit_rows["calendar_age_days"], errors="coerce")
            - reference_calendar_age_days
        ).clip(lower=0.0)
        if force_zero_intercept:
            design = np.column_stack(
                [
                    x.to_numpy(dtype=float),
                    calendar_days.to_numpy(dtype=float),
                ]
            )
            coefficients = _nonnegative_least_squares(design, y_array)
            intercept_loss_ah = 0.0
            loss_slope_per_1000ah = float(coefficients[0])
            calendar_loss_per_day = float(coefficients[1])
        else:
            design = np.column_stack(
                [
                    np.ones(len(fit_rows)),
                    x.to_numpy(dtype=float),
                    calendar_days.to_numpy(dtype=float),
                ]
            )
            coefficients = _nonnegative_least_squares(design, y_array)
            intercept_loss_ah = float(coefficients[0])
            loss_slope_per_1000ah = float(coefficients[1])
            calendar_loss_per_day = float(coefficients[2])
        model_name = "degradation_usage_calendar_constrained"
        feature_column = (
            "cumulative_all_discharge_ah_since_reference,"
            "calendar_age_days_since_reference"
        )
    else:
        raise ValueError(f"Unknown degradation model type: {degradation_model_type}")

    model = FittedDegradationModel(
        model_name=model_name,
        target_column="capacity_ah",
        feature_column=feature_column,
        baseline_capacity_ah=baseline_capacity_ah,
        reference_cumulative_ah=resolved_reference_cumulative_ah,
        intercept_loss_ah=intercept_loss_ah,
        loss_slope_per_1000ah=loss_slope_per_1000ah,
        train_rows=len(metric_rows),
        train_rmse=np.nan,
        train_mae=np.nan,
        holdout_rows=holdout_rows,
        holdout_rmse=np.nan,
        holdout_mae=np.nan,
        residual_std=np.nan,
        calendar_loss_per_day=calendar_loss_per_day,
        reference_calendar_age_days=reference_calendar_age_days,
        baseline_capacity_method=resolved_baseline_method,
    )

    prediction = _predict_degradation_capacity(metric_rows, model)
    train_rmse, train_mae, residual_std = _error_metrics(metric_rows["capacity_ah"], prediction)
    return FittedDegradationModel(
        model_name=model.model_name,
        target_column=model.target_column,
        feature_column=model.feature_column,
        baseline_capacity_ah=model.baseline_capacity_ah,
        reference_cumulative_ah=model.reference_cumulative_ah,
        intercept_loss_ah=model.intercept_loss_ah,
        loss_slope_per_1000ah=model.loss_slope_per_1000ah,
        train_rows=model.train_rows,
        train_rmse=train_rmse,
        train_mae=train_mae,
        holdout_rows=model.holdout_rows,
        holdout_rmse=model.holdout_rmse,
        holdout_mae=model.holdout_mae,
        residual_std=residual_std,
        calendar_loss_per_day=model.calendar_loss_per_day,
        reference_calendar_age_days=model.reference_calendar_age_days,
        baseline_capacity_method=model.baseline_capacity_method,
    )


def fit_degradation_usage_model(
    training_table: pd.DataFrame,
    *,
    holdout_fraction: float = 0.2,
    degradation_model_type: str = "usage_linear",
    baseline_capacity_ah: float | None = None,
    baseline_method: str = "max",
    reference_cumulative_ah: float | None = None,
    force_zero_intercept: bool = False,
) -> FittedDegradationModel | None:
    """Fit a physically directed capacity-loss model from calculated capacity.

    The ordinary statistical models describe historical shape. This model is for
    future degradation forecasting: it learns capacity loss per cumulative
    discharged Ah after the reliable/stable period. If the stable data does not
    show degradation yet, the learned loss slope becomes zero instead of letting
    future usage improve SOH.
    """
    train = _degradation_candidate_rows(training_table)
    if len(train) < 2:
        return None
    valid_baseline_rows = training_table[
        training_table["valid_training_row"]
    ].dropna(subset=["capacity_ah", "event_discharge_id", "end_anchor_time"]).copy()
    baseline_rows = valid_baseline_rows if baseline_method == "first" else None

    holdout_rows = (
        max(1, int(round(len(train) * holdout_fraction)))
        if holdout_fraction > 0 and len(train) >= 6
        else 0
    )
    if holdout_rows >= len(train):
        holdout_rows = len(train) - 1
    fit_rows = train.iloc[:-holdout_rows].copy() if holdout_rows else train.copy()
    holdout = train.iloc[-holdout_rows:].copy() if holdout_rows else pd.DataFrame()

    model = _fit_degradation_model_from_rows(
        fit_rows,
        metric_rows=train,
        holdout_rows=holdout_rows,
        degradation_model_type=degradation_model_type,
        baseline_capacity_ah=baseline_capacity_ah,
        baseline_method=baseline_method,
        baseline_rows=baseline_rows,
        reference_cumulative_ah=reference_cumulative_ah,
        force_zero_intercept=force_zero_intercept,
    )
    if model is None:
        return None

    if holdout_rows:
        holdout_prediction = _predict_degradation_capacity(holdout, model)
        holdout_rmse, holdout_mae, _ = _error_metrics(holdout["capacity_ah"], holdout_prediction)
    else:
        holdout_rmse = np.nan
        holdout_mae = np.nan

    return FittedDegradationModel(
        model_name=model.model_name,
        target_column=model.target_column,
        feature_column=model.feature_column,
        baseline_capacity_ah=model.baseline_capacity_ah,
        reference_cumulative_ah=model.reference_cumulative_ah,
        intercept_loss_ah=model.intercept_loss_ah,
        loss_slope_per_1000ah=model.loss_slope_per_1000ah,
        train_rows=model.train_rows,
        train_rmse=model.train_rmse,
        train_mae=model.train_mae,
        holdout_rows=model.holdout_rows,
        holdout_rmse=float(holdout_rmse),
        holdout_mae=float(holdout_mae),
        residual_std=model.residual_std,
        calendar_loss_per_day=model.calendar_loss_per_day,
        reference_calendar_age_days=model.reference_calendar_age_days,
        baseline_capacity_method=model.baseline_capacity_method,
    )


def build_degradation_backtest_table(
    training_table: pd.DataFrame,
    *,
    nominal_capacity_ah: float,
    holdout_fraction: float = 0.2,
    degradation_model_type: str = "usage_linear",
    baseline_capacity_ah: float | None = None,
    baseline_method: str = "max",
    reference_cumulative_ah: float | None = None,
    force_zero_intercept: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Hide the newest capacity rows and test the degradation forecast on them."""
    rows = _degradation_candidate_rows(training_table)
    columns = [
        "row_type",
        "serial",
        "event_discharge_id",
        "end_anchor_time",
        "calendar_age_days",
        "cumulative_all_discharge_ah",
        "actual_capacity_ah",
        "predicted_capacity_ah",
        "capacity_error_ah",
        "abs_capacity_error_ah",
        "actual_soh_pct",
        "predicted_soh_pct",
        "soh_error_pct",
        "abs_soh_error_pct",
        "model_name",
        "baseline_capacity_ah",
        "loss_slope_per_1000ah",
        "calendar_loss_per_day",
        "baseline_capacity_method",
    ]
    if holdout_fraction <= 0:
        summary = {
            "backtest_available": False,
            "reason": "Internal holdout disabled.",
            "rows": len(rows),
        }
        return pd.DataFrame(columns=columns), summary

    if len(rows) < 6:
        summary = {
            "backtest_available": False,
            "reason": "Need at least 6 valid stable capacity measurements.",
            "rows": len(rows),
        }
        return pd.DataFrame(columns=columns), summary

    valid_baseline_rows = training_table[
        training_table["valid_training_row"]
    ].dropna(subset=["capacity_ah", "event_discharge_id", "end_anchor_time"]).copy()
    baseline_rows = valid_baseline_rows if baseline_method == "first" else None
    holdout_rows = max(1, int(round(len(rows) * holdout_fraction)))
    if holdout_rows >= len(rows):
        holdout_rows = len(rows) - 1
    fit_rows = rows.iloc[:-holdout_rows].copy()
    evaluation_rows = rows.copy()
    model = _fit_degradation_model_from_rows(
        fit_rows,
        metric_rows=evaluation_rows,
        holdout_rows=holdout_rows,
        degradation_model_type=degradation_model_type,
        baseline_capacity_ah=baseline_capacity_ah,
        baseline_method=baseline_method,
        baseline_rows=baseline_rows,
        reference_cumulative_ah=reference_cumulative_ah,
        force_zero_intercept=force_zero_intercept,
    )
    if model is None:
        summary = {
            "backtest_available": False,
            "reason": "Could not fit degradation model.",
            "rows": len(rows),
        }
        return pd.DataFrame(columns=columns), summary

    backtest = evaluation_rows[
        [
            "serial",
            "event_discharge_id",
            "end_anchor_time",
            "calendar_age_days",
            "cumulative_all_discharge_ah",
            "capacity_ah",
        ]
    ].copy()
    backtest["row_type"] = "train"
    backtest.iloc[-holdout_rows:, backtest.columns.get_loc("row_type")] = "holdout"
    backtest["actual_capacity_ah"] = backtest["capacity_ah"]
    backtest["predicted_capacity_ah"] = _predict_degradation_capacity(backtest, model)
    backtest["capacity_error_ah"] = (
        backtest["actual_capacity_ah"] - backtest["predicted_capacity_ah"]
    )
    backtest["abs_capacity_error_ah"] = backtest["capacity_error_ah"].abs()
    backtest["actual_soh_pct"] = backtest["actual_capacity_ah"] / nominal_capacity_ah * 100.0
    backtest["predicted_soh_pct"] = (
        backtest["predicted_capacity_ah"] / nominal_capacity_ah * 100.0
    )
    backtest["soh_error_pct"] = backtest["actual_soh_pct"] - backtest["predicted_soh_pct"]
    backtest["abs_soh_error_pct"] = backtest["soh_error_pct"].abs()
    backtest["model_name"] = model.model_name
    backtest["baseline_capacity_ah"] = model.baseline_capacity_ah
    backtest["loss_slope_per_1000ah"] = model.loss_slope_per_1000ah
    backtest["calendar_loss_per_day"] = model.calendar_loss_per_day
    backtest["baseline_capacity_method"] = model.baseline_capacity_method

    holdout = backtest[backtest["row_type"].eq("holdout")]
    train_part = backtest[backtest["row_type"].eq("train")]
    summary = {
        "backtest_available": True,
        "model_name": model.model_name,
        "train_rows": len(train_part),
        "holdout_rows": len(holdout),
        "train_mae_ah": float(train_part["abs_capacity_error_ah"].mean()),
        "holdout_mae_ah": float(holdout["abs_capacity_error_ah"].mean()),
        "holdout_rmse_ah": float(np.sqrt(np.mean(np.square(holdout["capacity_error_ah"])))),
        "holdout_mae_soh_pct": float(holdout["abs_soh_error_pct"].mean()),
        "holdout_max_abs_soh_error_pct": float(holdout["abs_soh_error_pct"].max()),
        "baseline_capacity_ah": model.baseline_capacity_ah,
        "loss_slope_per_1000ah": model.loss_slope_per_1000ah,
        "calendar_loss_per_day": model.calendar_loss_per_day,
        "baseline_capacity_method": model.baseline_capacity_method,
        "train_end_time": str(train_part["end_anchor_time"].max()),
        "holdout_start_time": str(holdout["end_anchor_time"].min()),
    }
    return backtest[columns].reset_index(drop=True), summary


def build_cutoff_validation_table(
    validation_table: pd.DataFrame,
    *,
    best_capacity_ah_model: FittedTrendModel | None,
    degradation_model: FittedDegradationModel | None,
    nominal_capacity_ah: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Predict post-cutoff rows and compare against known future capacity."""
    columns = [
        "row_type",
        "serial",
        "event_discharge_id",
        "end_anchor_time",
        "cumulative_all_discharge_ah",
        "actual_capacity_ah",
        "statistical_predicted_capacity_ah",
        "statistical_capacity_error_ah",
        "abs_statistical_capacity_error_ah",
        "statistical_predicted_soh_pct",
        "statistical_soh_error_pct",
        "abs_statistical_soh_error_pct",
        "predicted_capacity_ah",
        "capacity_error_ah",
        "abs_capacity_error_ah",
        "actual_soh_pct",
        "predicted_soh_pct",
        "soh_error_pct",
        "abs_soh_error_pct",
        "capacity_ah_model",
    ]
    if validation_table.empty:
        summary = {
            "validation_available": False,
            "reason": "No rows after training cutoff.",
            "rows": 0,
        }
        return pd.DataFrame(columns=columns), summary

    validation = validation_table[validation_table["valid_training_row"]].copy()
    validation = validation.dropna(
        subset=["capacity_ah", "cumulative_all_discharge_ah", "end_anchor_time"],
    )
    if validation.empty:
        summary = {
            "validation_available": False,
            "reason": "No valid post-cutoff rows to validate.",
            "rows": 0,
        }
        return pd.DataFrame(columns=columns), summary

    validation = validation.sort_values("end_anchor_time", kind="mergesort").reset_index(drop=True)
    validation["row_type"] = "cutoff_validation"
    validation["actual_capacity_ah"] = validation["capacity_ah"]
    validation["actual_soh_pct"] = validation["actual_capacity_ah"] / nominal_capacity_ah * 100.0
    if best_capacity_ah_model is not None:
        validation["statistical_predicted_capacity_ah"] = _prediction_from_coefficients(
            validation,
            intercept=best_capacity_ah_model.intercept,
            coefficients=best_capacity_ah_model.coefficients,
            feature_columns=best_capacity_ah_model.feature_columns,
        )
    else:
        validation["statistical_predicted_capacity_ah"] = np.nan

    validation["statistical_capacity_error_ah"] = (
        validation["actual_capacity_ah"] - validation["statistical_predicted_capacity_ah"]
    )
    validation["abs_statistical_capacity_error_ah"] = (
        validation["statistical_capacity_error_ah"].abs()
    )
    validation["statistical_predicted_soh_pct"] = (
        validation["statistical_predicted_capacity_ah"] / nominal_capacity_ah * 100.0
    )
    validation["statistical_soh_error_pct"] = (
        validation["actual_soh_pct"] - validation["statistical_predicted_soh_pct"]
    )
    validation["abs_statistical_soh_error_pct"] = (
        validation["statistical_soh_error_pct"].abs()
    )

    if degradation_model is not None:
        validation["predicted_capacity_ah"] = _predict_degradation_capacity(
            validation,
            degradation_model,
        )
        validation["capacity_ah_model"] = degradation_model.model_name
    else:
        validation["predicted_capacity_ah"] = validation["statistical_predicted_capacity_ah"]
        validation["capacity_ah_model"] = (
            best_capacity_ah_model.model_name if best_capacity_ah_model else None
        )

    validation["capacity_error_ah"] = (
        validation["actual_capacity_ah"] - validation["predicted_capacity_ah"]
    )
    validation["abs_capacity_error_ah"] = validation["capacity_error_ah"].abs()
    validation["predicted_soh_pct"] = (
        validation["predicted_capacity_ah"] / nominal_capacity_ah * 100.0
    )
    validation["soh_error_pct"] = validation["actual_soh_pct"] - validation["predicted_soh_pct"]
    validation["abs_soh_error_pct"] = validation["soh_error_pct"].abs()

    statistical_valid = validation.dropna(subset=["statistical_predicted_capacity_ah"])
    if best_capacity_ah_model is not None and not statistical_valid.empty:
        statistical_summary: dict[str, Any] = {
            "validation_available": True,
            "model_name": best_capacity_ah_model.model_name,
            "feature_columns": list(best_capacity_ah_model.feature_columns),
            "rows": len(statistical_valid),
            "mae_ah": float(statistical_valid["abs_statistical_capacity_error_ah"].mean()),
            "rmse_ah": float(
                np.sqrt(np.mean(np.square(statistical_valid["statistical_capacity_error_ah"])))
            ),
            "mae_soh_pct": float(statistical_valid["abs_statistical_soh_error_pct"].mean()),
            "max_abs_soh_error_pct": float(
                statistical_valid["abs_statistical_soh_error_pct"].max()
            ),
        }
    else:
        statistical_summary = {
            "validation_available": False,
            "reason": "No best capacity Ah model available.",
            "rows": 0,
        }

    summary = {
        "validation_available": True,
        "rows": len(validation),
        "start_time": str(validation["end_anchor_time"].min()),
        "end_time": str(validation["end_anchor_time"].max()),
        "mae_ah": float(validation["abs_capacity_error_ah"].mean()),
        "rmse_ah": float(np.sqrt(np.mean(np.square(validation["capacity_error_ah"])))),
        "mae_soh_pct": float(validation["abs_soh_error_pct"].mean()),
        "max_abs_soh_error_pct": float(validation["abs_soh_error_pct"].max()),
        "capacity_ah_model": (
            degradation_model.model_name
            if degradation_model is not None
            else best_capacity_ah_model.model_name if best_capacity_ah_model else None
        ),
        "best_capacity_ah_model_cutoff_validation": statistical_summary,
    }
    return validation[columns].reset_index(drop=True), summary


def fit_trend_model(
    df: pd.DataFrame,
    *,
    model_name: str,
    target_column: str,
    feature_columns: Sequence[str],
    holdout_fraction: float = 0.2,
) -> FittedTrendModel | None:
    """Fit one least-squares trend model."""
    required = [target_column, *feature_columns]
    train = df[df["valid_training_row"]].dropna(subset=required).copy()
    if len(train) < 2:
        return None

    train = train.sort_values("end_anchor_time", kind="mergesort")
    holdout_rows = (
        max(1, int(round(len(train) * holdout_fraction)))
        if holdout_fraction > 0 and len(train) >= 6
        else 0
    )
    if holdout_rows >= len(train):
        holdout_rows = len(train) - 1
    fit_rows = train.iloc[:-holdout_rows] if holdout_rows else train
    holdout = train.iloc[-holdout_rows:] if holdout_rows else pd.DataFrame()

    x = fit_rows[list(feature_columns)].astype(float).to_numpy()
    x = np.column_stack([np.ones(len(fit_rows)), x])
    y = fit_rows[target_column].astype(float).to_numpy()
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)

    intercept = float(coefficients[0])
    slopes = tuple(float(value) for value in coefficients[1:])

    train_prediction = _prediction_from_coefficients(
        train,
        intercept=intercept,
        coefficients=slopes,
        feature_columns=feature_columns,
    )
    train_rmse, train_mae, residual_std = _error_metrics(train[target_column], train_prediction)

    if holdout_rows:
        holdout_prediction = _prediction_from_coefficients(
            holdout,
            intercept=intercept,
            coefficients=slopes,
            feature_columns=feature_columns,
        )
        holdout_rmse, holdout_mae, _ = _error_metrics(holdout[target_column], holdout_prediction)
    else:
        holdout_rmse = np.nan
        holdout_mae = np.nan

    return FittedTrendModel(
        model_name=model_name,
        target_column=target_column,
        feature_columns=tuple(feature_columns),
        intercept=intercept,
        coefficients=slopes,
        train_rows=len(train),
        train_rmse=train_rmse,
        train_mae=train_mae,
        holdout_rows=holdout_rows,
        holdout_rmse=float(holdout_rmse),
        holdout_mae=float(holdout_mae),
        residual_std=residual_std,
    )


def fit_candidate_models(
    training_table: pd.DataFrame,
    *,
    holdout_fraction: float = 0.2,
) -> list[FittedTrendModel]:
    """Fit candidate capacity trend models for Ah and Wh targets."""
    models: list[FittedTrendModel] = []
    for target_column in ("capacity_ah", "capacity_wh"):
        for model_name, feature_columns in _model_features_for_target(target_column).items():
            model = fit_trend_model(
                training_table,
                model_name=model_name,
                target_column=target_column,
                feature_columns=feature_columns,
                holdout_fraction=holdout_fraction,
            )
            if model is not None:
                models.append(model)
    return models


def choose_best_model(
    models: Sequence[FittedTrendModel],
    *,
    target_column: str,
) -> FittedTrendModel | None:
    candidates = [model for model in models if model.target_column == target_column]
    if not candidates:
        return None

    def score(model: FittedTrendModel) -> float:
        return model.holdout_rmse if not np.isnan(model.holdout_rmse) else model.train_rmse

    return min(candidates, key=score)


def _models_to_frame(models: Sequence[FittedTrendModel]) -> pd.DataFrame:
    rows = []
    for model in models:
        row = asdict(model)
        row["feature_columns"] = ",".join(model.feature_columns)
        row["coefficients"] = json.dumps(list(model.coefficients))
        rows.append(row)
    return pd.DataFrame(rows)


def _degradation_model_to_frame(model: FittedDegradationModel | None) -> pd.DataFrame:
    """Return a one-row table for the selected degradation forecast model."""
    if model is None:
        return pd.DataFrame()
    return pd.DataFrame([asdict(model)])


def _usage_source_frame(
    usage_ledger: pd.DataFrame,
    training_table: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    """Return cumulative usage rows and timestamp column for rate estimation."""
    source = "capacity_measurements"
    work = training_table.copy()
    time_col = "end_anchor_time"

    if not usage_ledger.empty:
        source = "discharge_usage_ledger"
        work = usage_ledger.copy()
        time_col = "end_timestamp"

    work[time_col] = pd.to_datetime(work[time_col], errors="coerce", utc=True)
    if "start_timestamp" in work.columns:
        work["start_timestamp"] = pd.to_datetime(
            work["start_timestamp"],
            errors="coerce",
            utc=True,
        )
    work = _as_numeric(
        work,
        [
            "cumulative_all_discharge_ah",
            "cumulative_all_discharge_wh",
            "discharge_ah",
            "discharge_wh",
        ],
    )
    work = work.dropna(
        subset=[time_col, "cumulative_all_discharge_ah", "cumulative_all_discharge_wh"],
    )
    return work, source, time_col


def _historical_mean_usage_rates(
    work: pd.DataFrame,
    time_col: str,
) -> tuple[float, float]:
    """Return whole-window average discharged Ah/day and Wh/day."""
    if len(work) < 2:
        return np.nan, np.nan

    work = work.sort_values(time_col, kind="mergesort")
    first = work.iloc[0]
    last = work.iloc[-1]
    elapsed_days = max(
        (last[time_col] - first[time_col]).total_seconds() / 86400.0,
        1e-9,
    )
    ah_per_day = (
        last["cumulative_all_discharge_ah"] - first["cumulative_all_discharge_ah"]
    ) / elapsed_days
    wh_per_day = (
        last["cumulative_all_discharge_wh"] - first["cumulative_all_discharge_wh"]
    ) / elapsed_days
    return float(ah_per_day), float(wh_per_day)


def _daily_usage_from_intervals(work: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Allocate episode Ah/Wh to calendar days by overlap duration."""
    required_columns = {"start_timestamp", time_col, "discharge_ah", "discharge_wh"}
    if not required_columns.issubset(work.columns):
        return _daily_usage_from_cumulative_end_day(work, time_col)

    intervals = work.dropna(
        subset=["start_timestamp", time_col, "discharge_ah", "discharge_wh"],
    )
    if intervals.empty:
        return _daily_usage_from_cumulative_end_day(work, time_col)

    daily_usage: dict[pd.Timestamp, list[float]] = {}
    for row in intervals.itertuples(index=False):
        start = getattr(row, "start_timestamp")
        end = getattr(row, time_col)
        discharge_ah = max(0.0, float(getattr(row, "discharge_ah")))
        discharge_wh = max(0.0, float(getattr(row, "discharge_wh")))

        if pd.isna(start) or pd.isna(end) or end <= start:
            day = pd.Timestamp(end).floor("D")
            values = daily_usage.setdefault(day, [0.0, 0.0])
            values[0] += discharge_ah
            values[1] += discharge_wh
            continue

        total_seconds = (end - start).total_seconds()
        cursor = start
        while cursor < end:
            day = pd.Timestamp(cursor).floor("D")
            next_day = day + pd.Timedelta(days=1)
            segment_end = min(end, next_day)
            fraction = (segment_end - cursor).total_seconds() / total_seconds
            values = daily_usage.setdefault(day, [0.0, 0.0])
            values[0] += discharge_ah * fraction
            values[1] += discharge_wh * fraction
            cursor = segment_end

    if not daily_usage:
        return _daily_usage_from_cumulative_end_day(work, time_col)

    daily = pd.DataFrame.from_dict(
        daily_usage,
        orient="index",
        columns=["delta_ah", "delta_wh"],
    )
    daily.index.name = "usage_day"
    return daily.sort_index()


def _daily_usage_from_cumulative_end_day(work: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Fallback daily usage table assigning cumulative deltas to the end day."""
    work = work.sort_values(time_col, kind="mergesort").copy()
    work["delta_ah"] = work["cumulative_all_discharge_ah"].diff().clip(lower=0.0)
    work["delta_wh"] = work["cumulative_all_discharge_wh"].diff().clip(lower=0.0)
    work[["delta_ah", "delta_wh"]] = work[["delta_ah", "delta_wh"]].fillna(0.0)
    work["usage_day"] = work[time_col].dt.floor("D")
    return work.groupby("usage_day", sort=True)[["delta_ah", "delta_wh"]].sum()


def _recent_median_usage_rates(
    work: pd.DataFrame,
    time_col: str,
    recent_usage_days: int,
) -> tuple[float, float]:
    """Return median daily discharged Ah and Wh over the recent lookback window."""
    if len(work) < 2:
        return np.nan, np.nan
    if recent_usage_days < 1:
        raise ValueError("recent_usage_days must be at least 1.")

    daily = _daily_usage_from_intervals(work, time_col)
    if daily.empty:
        return np.nan, np.nan

    last_day = daily.index.max()
    first_recent_day = last_day - pd.Timedelta(days=recent_usage_days - 1)
    recent_index = pd.date_range(first_recent_day, last_day, freq="D")
    recent_daily = daily.reindex(recent_index, fill_value=0.0)
    return (
        float(recent_daily["delta_ah"].median()),
        float(recent_daily["delta_wh"].median()),
    )


def _historical_median_usage_rates(
    work: pd.DataFrame,
    time_col: str,
) -> tuple[float, float]:
    """Return median daily discharged Ah and Wh over the whole training window."""
    if len(work) < 2:
        return np.nan, np.nan

    daily = _daily_usage_from_intervals(work, time_col)
    if daily.empty:
        return np.nan, np.nan

    start_day = daily.index.min()
    end_day = daily.index.max()
    full_index = pd.date_range(start_day, end_day, freq="D")
    daily = daily.reindex(full_index, fill_value=0.0)
    return (
        float(daily["delta_ah"].median()),
        float(daily["delta_wh"].median()),
    )


def estimate_usage_rates(
    usage_ledger: pd.DataFrame,
    training_table: pd.DataFrame,
) -> dict[str, float | str | int | None]:
    """Estimate historical daily discharge usage for future scenarios."""
    return estimate_usage_rates_with_mode(
        usage_ledger,
        training_table,
        usage_rate_mode="historical_mean",
        recent_usage_days=90,
    )


def estimate_usage_rates_with_mode(
    usage_ledger: pd.DataFrame,
    training_table: pd.DataFrame,
    *,
    usage_rate_mode: str,
    recent_usage_days: int,
) -> dict[str, float | str | int | None]:
    """Estimate forecast Ah/day and Wh/day using the selected usage-rate mode."""
    valid_modes = {"historical_mean", "historical_median", "recent_median"}
    if usage_rate_mode not in valid_modes:
        raise ValueError(
            "usage_rate_mode must be one of: historical_mean, "
            "historical_median, recent_median."
        )

    work, source, time_col = _usage_source_frame(usage_ledger, training_table)
    historical_ah, historical_wh = _historical_mean_usage_rates(work, time_col)
    historical_median_ah, historical_median_wh = _historical_median_usage_rates(
        work,
        time_col,
    )
    recent_median_ah, recent_median_wh = _recent_median_usage_rates(
        work,
        time_col,
        recent_usage_days,
    )

    if len(work) < 2:
        return {
            "source": source,
            "usage_rate_mode": usage_rate_mode,
            "recent_usage_days": recent_usage_days,
            "daily_usage_allocation": "interval_overlap",
            "historical_ah_per_day": historical_ah,
            "historical_wh_per_day": historical_wh,
            "historical_median_ah_per_day": historical_median_ah,
            "historical_median_wh_per_day": historical_median_wh,
            "recent_median_ah_per_day": recent_median_ah,
            "recent_median_wh_per_day": recent_median_wh,
            "forecast_ah_per_day": np.nan,
            "forecast_wh_per_day": np.nan,
        }

    if usage_rate_mode == "recent_median":
        forecast_ah = recent_median_ah
        forecast_wh = recent_median_wh
    elif usage_rate_mode == "historical_median":
        forecast_ah = historical_median_ah
        forecast_wh = historical_median_wh
    else:
        forecast_ah = historical_ah
        forecast_wh = historical_wh

    return {
        "source": source,
        "usage_rate_mode": usage_rate_mode,
        "recent_usage_days": recent_usage_days,
        "daily_usage_allocation": "interval_overlap",
        "historical_ah_per_day": historical_ah,
        "historical_wh_per_day": historical_wh,
        "historical_median_ah_per_day": historical_median_ah,
        "historical_median_wh_per_day": historical_median_wh,
        "recent_median_ah_per_day": recent_median_ah,
        "recent_median_wh_per_day": recent_median_wh,
        "forecast_ah_per_day": forecast_ah,
        "forecast_wh_per_day": forecast_wh,
    }


def parse_scenarios(value: str | None) -> dict[str, float] | None:
    """Parse scenario multipliers from label=value,label=value text."""
    if value is None or value.strip() == "":
        return None

    scenarios: dict[str, float] = {}
    for part in value.split(","):
        if "=" not in part:
            raise ValueError("Scenarios must use label=value pairs, separated by commas.")
        label, multiplier = part.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError("Scenario label cannot be empty.")
        scenarios[label] = float(multiplier)
    return scenarios


def parse_usage_rate_modes(value: str | None) -> tuple[str, ...]:
    """Parse comma-separated usage-rate modes."""
    if value is None or value.strip() == "":
        return DEFAULT_USAGE_RATE_MODES

    modes = tuple(part.strip() for part in value.split(",") if part.strip())
    valid_modes = set(DEFAULT_USAGE_RATE_MODES)
    unknown = sorted(set(modes).difference(valid_modes))
    if unknown:
        raise ValueError(f"Unknown usage-rate mode(s): {unknown}")
    return modes


def build_forecast_table(
    training_table: pd.DataFrame,
    usage_rates: dict[str, float | str],
    *,
    best_capacity_ah_model: FittedTrendModel | None,
    best_capacity_wh_model: FittedTrendModel | None,
    degradation_model: FittedDegradationModel | None,
    nominal_capacity_ah: float,
    scenarios: dict[str, float],
    future_days: int,
    step_days: int,
) -> pd.DataFrame:
    """Create scenario-based future capacity predictions."""
    if training_table.empty or (best_capacity_ah_model is None and degradation_model is None):
        return pd.DataFrame()

    valid = training_table[training_table["valid_training_row"]].copy()
    if valid.empty:
        return pd.DataFrame()

    valid = valid.sort_values("end_anchor_time", kind="mergesort")
    first_time = valid["end_anchor_time"].min()
    last = valid.iloc[-1]
    latest_capacity_ah = max(float(last["capacity_ah"]), 1e-9)
    forecast_ah_per_day = float(
        usage_rates.get("forecast_ah_per_day", usage_rates["historical_ah_per_day"])
    )
    forecast_wh_per_day = float(
        usage_rates.get("forecast_wh_per_day", usage_rates["historical_wh_per_day"])
    )
    usage_rate_model = str(usage_rates.get("usage_rate_mode", "historical_mean"))

    if np.isnan(forecast_ah_per_day) or forecast_ah_per_day <= 0:
        forecast_ah_per_day = latest_capacity_ah / 7.0
    if np.isnan(forecast_wh_per_day) or forecast_wh_per_day <= 0:
        forecast_wh_per_day = float(last["capacity_wh"]) / 7.0

    rows: list[dict[str, Any]] = []
    days = range(step_days, future_days + 1, step_days)
    for scenario, multiplier in scenarios.items():
        for day in days:
            forecast_time = last["end_anchor_time"] + pd.Timedelta(days=day)
            future_cumulative_ah = (
                float(last["cumulative_all_discharge_ah"])
                + forecast_ah_per_day * multiplier * day
            )
            future_cumulative_wh = (
                float(last["cumulative_all_discharge_wh"])
                + forecast_wh_per_day * multiplier * day
            )
            event_increment = (
                future_cumulative_ah - float(last["cumulative_all_discharge_ah"])
            ) / latest_capacity_ah
            rows.append(
                {
                    "row_type": "predicted",
                    "serial": last["serial"],
                    "scenario": scenario,
                    "usage_rate_model": usage_rate_model,
                    "usage_multiplier": multiplier,
                    "forecast_ah_per_day": forecast_ah_per_day,
                    "forecast_wh_per_day": forecast_wh_per_day,
                    "forecast_days_after_last_measurement": day,
                    "forecast_timestamp": forecast_time,
                    "event_discharge_id": float(last["event_discharge_id"]) + event_increment,
                    "calendar_age_days": (forecast_time - first_time).total_seconds() / 86400.0,
                    "cumulative_all_discharge_ah": future_cumulative_ah,
                    "cumulative_all_discharge_wh": future_cumulative_wh,
                }
            )

    forecast = add_model_features(pd.DataFrame(rows))
    if best_capacity_ah_model is not None:
        forecast["statistical_predicted_capacity_ah"] = _prediction_from_coefficients(
            forecast,
            intercept=best_capacity_ah_model.intercept,
            coefficients=best_capacity_ah_model.coefficients,
            feature_columns=best_capacity_ah_model.feature_columns,
        )
    else:
        forecast["statistical_predicted_capacity_ah"] = np.nan

    if degradation_model is not None:
        forecast["predicted_capacity_ah"] = _predict_degradation_capacity(
            forecast,
            degradation_model,
        )
        residual = degradation_model.residual_std
        forecast["capacity_ah_model"] = degradation_model.model_name
    else:
        forecast["predicted_capacity_ah"] = forecast["statistical_predicted_capacity_ah"]
        residual = best_capacity_ah_model.residual_std if best_capacity_ah_model else 0.0
        forecast["capacity_ah_model"] = best_capacity_ah_model.model_name if best_capacity_ah_model else None

    forecast["predicted_measured_soh_pct"] = (
        forecast["predicted_capacity_ah"] / nominal_capacity_ah * 100.0
    )
    forecast["predicted_future_soh_pct"] = forecast["predicted_measured_soh_pct"]
    forecast["equivalent_full_cycles"] = (
        forecast["cumulative_all_discharge_ah"] / nominal_capacity_ah
    )
    forecast["predicted_capacity_ah_lower_95"] = forecast["predicted_capacity_ah"] - 1.96 * residual
    forecast["predicted_capacity_ah_upper_95"] = forecast["predicted_capacity_ah"] + 1.96 * residual

    if best_capacity_wh_model is not None:
        forecast["predicted_capacity_wh"] = _prediction_from_coefficients(
            forecast,
            intercept=best_capacity_wh_model.intercept,
            coefficients=best_capacity_wh_model.coefficients,
            feature_columns=best_capacity_wh_model.feature_columns,
        )
        forecast["capacity_wh_model"] = best_capacity_wh_model.model_name
    else:
        forecast["predicted_capacity_wh"] = np.nan
        forecast["capacity_wh_model"] = None

    return forecast[
        [
            "row_type",
            "serial",
            "scenario",
            "usage_rate_model",
            "usage_multiplier",
            "forecast_ah_per_day",
            "forecast_wh_per_day",
            "forecast_days_after_last_measurement",
            "forecast_timestamp",
            "event_discharge_id",
            "calendar_age_days",
            "cumulative_all_discharge_ah",
            "cumulative_all_discharge_wh",
            "equivalent_full_cycles",
            "statistical_predicted_capacity_ah",
            "predicted_capacity_ah",
            "predicted_capacity_ah_lower_95",
            "predicted_capacity_ah_upper_95",
            "predicted_capacity_wh",
            "predicted_measured_soh_pct",
            "predicted_future_soh_pct",
            "capacity_ah_model",
            "capacity_wh_model",
        ]
    ]


def build_forecasts_for_usage_rate_modes(
    training_table: pd.DataFrame,
    usage_rates_by_model: dict[str, dict[str, float | str | int | None]],
    *,
    best_capacity_ah_model: FittedTrendModel | None,
    best_capacity_wh_model: FittedTrendModel | None,
    degradation_model: FittedDegradationModel | None,
    nominal_capacity_ah: float,
    scenarios: dict[str, float] | None,
    future_days: int,
    step_days: int,
) -> pd.DataFrame:
    """Create one future forecast table for all selected usage-rate modes."""
    frames: list[pd.DataFrame] = []
    for usage_rate_model, usage_rates in usage_rates_by_model.items():
        forecast_scenarios = scenarios or {usage_rate_model: 1.0}
        frame = build_forecast_table(
            training_table,
            usage_rates,
            best_capacity_ah_model=best_capacity_ah_model,
            best_capacity_wh_model=best_capacity_wh_model,
            degradation_model=degradation_model,
            nominal_capacity_ah=nominal_capacity_ah,
            scenarios=forecast_scenarios,
            future_days=future_days,
            step_days=step_days,
        )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_cutoff_forecast_validation_table(
    validation_table: pd.DataFrame,
    training_table: pd.DataFrame,
    usage_ledger_for_training: pd.DataFrame,
    usage_rates_by_model: dict[str, dict[str, float | str | int | None]],
    *,
    best_capacity_ah_model: FittedTrendModel | None,
    degradation_model: FittedDegradationModel | None,
    nominal_capacity_ah: float,
    training_cutoff: pd.Timestamp | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate full forecast logic by estimating post-cutoff usage from training data."""
    columns = [
        "row_type",
        "serial",
        "usage_rate_model",
        "forecast_ah_per_day",
        "forecast_wh_per_day",
        "forecast_anchor_source",
        "forecast_anchor_time",
        "forecast_anchor_usage_timestamp",
        "forecast_anchor_cumulative_all_discharge_ah",
        "event_discharge_id",
        "end_anchor_time",
        "forecast_days_after_last_measurement",
        "actual_cumulative_all_discharge_ah",
        "predicted_cumulative_all_discharge_ah",
        "cumulative_ah_error",
        "abs_cumulative_ah_error",
        "daily_ah_error_from_cutoff",
        "actual_capacity_ah",
        "predicted_capacity_ah",
        "capacity_error_ah",
        "abs_capacity_error_ah",
        "actual_soh_pct",
        "predicted_soh_pct",
        "soh_error_pct",
        "abs_soh_error_pct",
        "capacity_ah_model",
    ]
    if validation_table.empty or training_table.empty:
        return pd.DataFrame(columns=columns), {
            "validation_available": False,
            "reason": "Missing training or post-cutoff validation rows.",
            "rows": 0,
        }

    train = training_table[training_table["valid_training_row"]].copy()
    validation = validation_table[validation_table["valid_training_row"]].copy()
    validation = validation.dropna(
        subset=["capacity_ah", "cumulative_all_discharge_ah", "end_anchor_time"],
    )
    if train.empty or validation.empty:
        return pd.DataFrame(columns=columns), {
            "validation_available": False,
            "reason": "No valid training or post-cutoff validation rows.",
            "rows": 0,
        }

    train = train.sort_values("end_anchor_time", kind="mergesort")
    validation = validation.sort_values("end_anchor_time", kind="mergesort")
    first_train_time = train["end_anchor_time"].min()
    last = train.iloc[-1]
    anchor = resolve_forecast_validation_anchor(
        train,
        usage_ledger_for_training,
        training_cutoff=training_cutoff,
    )
    latest_capacity_ah = float(anchor["latest_capacity_ah"])
    anchor_time = pd.Timestamp(anchor["anchor_time"])
    anchor_cumulative_ah = float(anchor["cumulative_all_discharge_ah"])
    anchor_cumulative_wh = float(anchor["cumulative_all_discharge_wh"])
    anchor_event_discharge_id = float(anchor["event_discharge_id"])

    frames: list[pd.DataFrame] = []
    summaries: dict[str, dict[str, Any]] = {}
    for usage_rate_model, usage_rates in usage_rates_by_model.items():
        forecast_ah_per_day = float(
            usage_rates.get("forecast_ah_per_day", usage_rates["historical_ah_per_day"])
        )
        forecast_wh_per_day = float(
            usage_rates.get("forecast_wh_per_day", usage_rates["historical_wh_per_day"])
        )
        if np.isnan(forecast_ah_per_day) or forecast_ah_per_day <= 0:
            forecast_ah_per_day = latest_capacity_ah / 7.0
        if np.isnan(forecast_wh_per_day) or forecast_wh_per_day <= 0:
            forecast_wh_per_day = float(last["capacity_wh"]) / 7.0

        frame = validation.copy().reset_index(drop=True)
        elapsed_days = (
            frame["end_anchor_time"] - anchor_time
        ).dt.total_seconds().clip(lower=0.0) / 86400.0
        predicted_cumulative_ah = anchor_cumulative_ah + forecast_ah_per_day * elapsed_days
        predicted_cumulative_wh = anchor_cumulative_wh + forecast_wh_per_day * elapsed_days
        event_increment = (predicted_cumulative_ah - anchor_cumulative_ah) / latest_capacity_ah

        frame["row_type"] = "cutoff_forecast_validation"
        frame["usage_rate_model"] = usage_rate_model
        frame["forecast_ah_per_day"] = forecast_ah_per_day
        frame["forecast_wh_per_day"] = forecast_wh_per_day
        frame["forecast_anchor_source"] = anchor["source"]
        frame["forecast_anchor_time"] = anchor_time
        frame["forecast_anchor_usage_timestamp"] = anchor["usage_timestamp"]
        frame["forecast_anchor_cumulative_all_discharge_ah"] = anchor_cumulative_ah
        frame["forecast_days_after_last_measurement"] = elapsed_days
        frame["actual_cumulative_all_discharge_ah"] = frame["cumulative_all_discharge_ah"]
        frame["predicted_cumulative_all_discharge_ah"] = predicted_cumulative_ah
        frame["cumulative_ah_error"] = (
            frame["actual_cumulative_all_discharge_ah"]
            - frame["predicted_cumulative_all_discharge_ah"]
        )
        frame["abs_cumulative_ah_error"] = frame["cumulative_ah_error"].abs()
        frame["daily_ah_error_from_cutoff"] = np.where(
            frame["forecast_days_after_last_measurement"].gt(0),
            frame["cumulative_ah_error"]
            / frame["forecast_days_after_last_measurement"],
            np.nan,
        )
        frame["cumulative_all_discharge_ah"] = predicted_cumulative_ah
        frame["cumulative_all_discharge_wh"] = predicted_cumulative_wh
        frame["event_discharge_id"] = anchor_event_discharge_id + event_increment
        frame["calendar_age_days"] = (
            frame["end_anchor_time"] - first_train_time
        ).dt.total_seconds() / 86400.0
        frame["actual_capacity_ah"] = frame["capacity_ah"]
        frame["actual_soh_pct"] = frame["actual_capacity_ah"] / nominal_capacity_ah * 100.0

        if degradation_model is not None:
            frame["predicted_capacity_ah"] = _predict_degradation_capacity(
                frame,
                degradation_model,
            )
            frame["capacity_ah_model"] = degradation_model.model_name
        elif best_capacity_ah_model is not None:
            frame = add_model_features(frame)
            frame["predicted_capacity_ah"] = _prediction_from_coefficients(
                frame,
                intercept=best_capacity_ah_model.intercept,
                coefficients=best_capacity_ah_model.coefficients,
                feature_columns=best_capacity_ah_model.feature_columns,
            )
            frame["capacity_ah_model"] = best_capacity_ah_model.model_name
        else:
            frame["predicted_capacity_ah"] = np.nan
            frame["capacity_ah_model"] = None

        frame["capacity_error_ah"] = (
            frame["actual_capacity_ah"] - frame["predicted_capacity_ah"]
        )
        frame["abs_capacity_error_ah"] = frame["capacity_error_ah"].abs()
        frame["predicted_soh_pct"] = (
            frame["predicted_capacity_ah"] / nominal_capacity_ah * 100.0
        )
        frame["soh_error_pct"] = frame["actual_soh_pct"] - frame["predicted_soh_pct"]
        frame["abs_soh_error_pct"] = frame["soh_error_pct"].abs()
        frames.append(frame[columns])

        valid_error = frame.dropna(
            subset=["forecast_days_after_last_measurement", "cumulative_ah_error"],
        )
        valid_error = valid_error[valid_error["forecast_days_after_last_measurement"].gt(0)]
        if valid_error.empty:
            daily_ah_error_final = float("nan")
            daily_ah_error_slope = float("nan")
            daily_ah_error_through_origin_slope = float("nan")
            cumulative_ah_error_intercept = float("nan")
            validation_days = float("nan")
        else:
            latest_error = valid_error.loc[
                valid_error["forecast_days_after_last_measurement"].idxmax()
            ]
            validation_days = float(latest_error["forecast_days_after_last_measurement"])
            daily_ah_error_final = (
                float(latest_error["cumulative_ah_error"]) / validation_days
                if validation_days > 0
                else float("nan")
            )
            days_array = valid_error["forecast_days_after_last_measurement"].to_numpy(
                dtype=float,
            )
            error_array = valid_error["cumulative_ah_error"].to_numpy(dtype=float)
            denominator = float(np.dot(days_array, days_array))
            daily_ah_error_through_origin_slope = (
                float(np.dot(days_array, error_array) / denominator)
                if denominator > 0
                else daily_ah_error_final
            )
            if len(valid_error) >= 2:
                design = np.column_stack([np.ones(len(valid_error)), days_array])
                coefficients, *_ = np.linalg.lstsq(design, error_array, rcond=None)
                cumulative_ah_error_intercept = float(coefficients[0])
                daily_ah_error_slope = float(coefficients[1])
            else:
                cumulative_ah_error_intercept = float(error_array[0])
                daily_ah_error_slope = daily_ah_error_final

        summaries[usage_rate_model] = {
            "validation_available": True,
            "rows": len(frame),
            "forecast_ah_per_day": forecast_ah_per_day,
            "forecast_wh_per_day": forecast_wh_per_day,
            "mae_ah": float(frame["abs_capacity_error_ah"].mean()),
            "rmse_ah": float(np.sqrt(np.mean(np.square(frame["capacity_error_ah"])))),
            "mae_soh_pct": float(frame["abs_soh_error_pct"].mean()),
            "max_abs_soh_error_pct": float(frame["abs_soh_error_pct"].max()),
            "mae_cumulative_ah": float(frame["abs_cumulative_ah_error"].mean()),
            "max_abs_cumulative_ah": float(frame["abs_cumulative_ah_error"].max()),
            "validation_days": validation_days,
            "forecast_anchor_source": anchor["source"],
            "forecast_anchor_time": str(anchor_time),
            "forecast_anchor_usage_timestamp": str(anchor["usage_timestamp"]),
            "forecast_anchor_cumulative_all_discharge_ah": anchor_cumulative_ah,
            "forecast_anchor_cumulative_all_discharge_wh": anchor_cumulative_wh,
            "forecast_anchor_event_discharge_id": anchor_event_discharge_id,
            "daily_ah_error_final": daily_ah_error_final,
            "abs_daily_ah_error_final": abs(daily_ah_error_final),
            "daily_ah_error_slope": daily_ah_error_slope,
            "abs_daily_ah_error_slope": abs(daily_ah_error_slope),
            "daily_ah_error_through_origin_slope": daily_ah_error_through_origin_slope,
            "abs_daily_ah_error_through_origin_slope": abs(daily_ah_error_through_origin_slope),
            "cumulative_ah_error_intercept": cumulative_ah_error_intercept,
            "abs_cumulative_ah_error_intercept": abs(cumulative_ah_error_intercept),
            "daily_ah_error_method": "linear_with_intercept",
        }

    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    summary = {
        "validation_available": not table.empty,
        "rows": len(table),
        "start_time": str(validation["end_anchor_time"].min()),
        "end_time": str(validation["end_anchor_time"].max()),
        "forecast_anchor_source": anchor["source"],
        "forecast_anchor_time": str(anchor_time),
        "forecast_anchor_usage_timestamp": str(anchor["usage_timestamp"]),
        "forecast_anchor_cumulative_all_discharge_ah": anchor_cumulative_ah,
        "forecast_anchor_cumulative_all_discharge_wh": anchor_cumulative_wh,
        "forecast_anchor_event_discharge_id": anchor_event_discharge_id,
        "by_usage_rate_model": summaries,
    }
    return table, summary


def add_forecast_uncertainty_columns(
    forecast: pd.DataFrame,
    cutoff_forecast_validation_summary: dict[str, Any],
    *,
    degradation_model: FittedDegradationModel | None,
    nominal_capacity_ah: float,
) -> pd.DataFrame:
    """Add validation-derived usage uncertainty estimates to future forecasts."""
    work = forecast.copy()
    uncertainty_columns = {
        "usage_error_daily_ah_slope": np.nan,
        "usage_error_daily_ah_final": np.nan,
        "projected_cumulative_ah_error_slope": np.nan,
        "projected_cumulative_ah_error_final": np.nan,
        "projected_capacity_error_ah_slope": np.nan,
        "projected_capacity_error_ah_final": np.nan,
        "projected_soh_error_pct_slope": np.nan,
        "projected_soh_error_pct_final": np.nan,
    }
    for column, value in uncertainty_columns.items():
        work[column] = value

    validation_by_mode = cutoff_forecast_validation_summary.get("by_usage_rate_model", {})
    if (
        work.empty
        or degradation_model is None
        or nominal_capacity_ah <= 0
        or not validation_by_mode
    ):
        return work

    loss_slope = abs(float(degradation_model.loss_slope_per_1000ah))
    if not np.isfinite(loss_slope):
        return work

    for usage_rate_model, validation in validation_by_mode.items():
        if not validation.get("validation_available"):
            continue

        slope_daily_error = validation.get("abs_daily_ah_error_slope")
        final_daily_error = validation.get("abs_daily_ah_error_final")
        if slope_daily_error is None or final_daily_error is None:
            continue

        mask = work["usage_rate_model"].astype(str).eq(str(usage_rate_model))
        if not mask.any():
            continue

        elapsed_days = pd.to_numeric(
            work.loc[mask, "forecast_days_after_last_measurement"],
            errors="coerce",
        ).clip(lower=0.0)
        slope_cumulative_error = float(slope_daily_error) * elapsed_days
        final_cumulative_error = float(final_daily_error) * elapsed_days
        slope_capacity_error = loss_slope * slope_cumulative_error / 1000.0
        final_capacity_error = loss_slope * final_cumulative_error / 1000.0

        work.loc[mask, "usage_error_daily_ah_slope"] = float(slope_daily_error)
        work.loc[mask, "usage_error_daily_ah_final"] = float(final_daily_error)
        work.loc[mask, "projected_cumulative_ah_error_slope"] = slope_cumulative_error
        work.loc[mask, "projected_cumulative_ah_error_final"] = final_cumulative_error
        work.loc[mask, "projected_capacity_error_ah_slope"] = slope_capacity_error
        work.loc[mask, "projected_capacity_error_ah_final"] = final_capacity_error
        work.loc[mask, "projected_soh_error_pct_slope"] = (
            slope_capacity_error / nominal_capacity_ah * 100.0
        )
        work.loc[mask, "projected_soh_error_pct_final"] = (
            final_capacity_error / nominal_capacity_ah * 100.0
        )

    return work


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def build_capacity_ml_forecast(
    *,
    capacity_run_dir: Path,
    output_base_dir: Path,
    nominal_capacity_ah: float | None,
    bms_soh_reliable_after: pd.Timestamp,
    min_reasonable_capacity_ah: float,
    max_reasonable_capacity_ah: float,
    exclude_statistical_outliers: bool = False,
    exclude_high_rest_ratio_capacity_rows: bool = False,
    max_rest_to_discharge_ratio_for_training: float = 1.0,
    training_cutoff: pd.Timestamp | None = None,
    validation_end: pd.Timestamp | None = None,
    internal_holdout_fraction: float | None = None,
    degradation_model_type: str = "usage_linear",
    degradation_baseline_capacity_ah: float | None = None,
    degradation_baseline_method: str | None = None,
    degradation_reference_cumulative_ah: float | None = None,
    degradation_force_zero_intercept: bool = False,
    usage_rate_modes: Sequence[str] = DEFAULT_USAGE_RATE_MODES,
    usage_rate_mode: str = "historical_mean",
    recent_usage_days: int = 90,
    scenarios: dict[str, float] | None = None,
    future_days: int = 365,
    step_days: int = 30,
    compression: str | None = "snappy",
    run_id: str | None = None,
) -> CapacityMLResult:
    """Build ML training and future forecast tables from capacity measurements."""
    measurements, usage_ledger = load_capacity_run(capacity_run_dir)
    nominal_capacity_ah, nominal_capacity_source = resolve_nominal_capacity_ah(
        measurements,
        nominal_capacity_ah,
    )
    training_table = prepare_training_table(
        measurements,
        nominal_capacity_ah=nominal_capacity_ah,
        bms_soh_reliable_after=bms_soh_reliable_after,
        min_reasonable_capacity_ah=min_reasonable_capacity_ah,
        max_reasonable_capacity_ah=max_reasonable_capacity_ah,
        exclude_statistical_outliers=exclude_statistical_outliers,
        exclude_high_rest_ratio_capacity_rows=exclude_high_rest_ratio_capacity_rows,
        max_rest_to_discharge_ratio_for_training=max_rest_to_discharge_ratio_for_training,
    )
    if training_table.empty:
        raise ValueError("No capacity measurements available for ML.")

    model_training_table, cutoff_validation_source = split_training_and_validation(
        training_table,
        training_cutoff=training_cutoff,
        validation_end=validation_end,
    )
    if model_training_table.empty:
        raise ValueError("No capacity measurements available at or before the training cutoff.")

    effective_holdout_fraction = (
        internal_holdout_fraction
        if internal_holdout_fraction is not None
        else 0.0 if training_cutoff is not None else 0.2
    )
    if effective_holdout_fraction < 0 or effective_holdout_fraction >= 1:
        raise ValueError("internal_holdout_fraction must be in the range [0, 1).")

    if degradation_model_type not in {"usage_linear", "usage_calendar_constrained"}:
        raise ValueError(
            "degradation_model_type must be one of: usage_linear, usage_calendar_constrained"
        )
    resolved_baseline_method = (
        degradation_baseline_method
        or ("first" if degradation_model_type == "usage_calendar_constrained" else "max")
    )

    usage_ledger_for_training = filter_usage_ledger_for_training(
        usage_ledger,
        training_cutoff=training_cutoff,
    )

    models = fit_candidate_models(
        model_training_table,
        holdout_fraction=effective_holdout_fraction,
    )
    best_capacity_ah_model = choose_best_model(models, target_column="capacity_ah")
    best_capacity_wh_model = choose_best_model(models, target_column="capacity_wh")
    degradation_model = fit_degradation_usage_model(
        model_training_table,
        holdout_fraction=effective_holdout_fraction,
        degradation_model_type=degradation_model_type,
        baseline_capacity_ah=degradation_baseline_capacity_ah,
        baseline_method=resolved_baseline_method,
        reference_cumulative_ah=degradation_reference_cumulative_ah,
        force_zero_intercept=degradation_force_zero_intercept,
    )
    backtest_table, backtest_summary = build_degradation_backtest_table(
        model_training_table,
        nominal_capacity_ah=nominal_capacity_ah,
        holdout_fraction=effective_holdout_fraction,
        degradation_model_type=degradation_model_type,
        baseline_capacity_ah=degradation_baseline_capacity_ah,
        baseline_method=resolved_baseline_method,
        reference_cumulative_ah=degradation_reference_cumulative_ah,
        force_zero_intercept=degradation_force_zero_intercept,
    )
    cutoff_validation_table, cutoff_validation_summary = build_cutoff_validation_table(
        cutoff_validation_source,
        best_capacity_ah_model=best_capacity_ah_model,
        degradation_model=degradation_model,
        nominal_capacity_ah=nominal_capacity_ah,
    )
    selected_usage_rate_modes = tuple(usage_rate_modes or (usage_rate_mode,))
    usage_rates_by_model = {
        mode: estimate_usage_rates_with_mode(
            usage_ledger_for_training,
            model_training_table,
            usage_rate_mode=mode,
            recent_usage_days=recent_usage_days,
        )
        for mode in selected_usage_rate_modes
    }
    primary_usage_rates = usage_rates_by_model[selected_usage_rate_modes[0]]
    cutoff_forecast_validation_table, cutoff_forecast_validation_summary = (
        build_cutoff_forecast_validation_table(
            cutoff_validation_source,
            model_training_table,
            usage_ledger_for_training,
            usage_rates_by_model,
            best_capacity_ah_model=best_capacity_ah_model,
            degradation_model=degradation_model,
            nominal_capacity_ah=nominal_capacity_ah,
            training_cutoff=training_cutoff,
        )
    )
    forecast = build_forecasts_for_usage_rate_modes(
        model_training_table,
        usage_rates_by_model,
        best_capacity_ah_model=best_capacity_ah_model,
        best_capacity_wh_model=best_capacity_wh_model,
        degradation_model=degradation_model,
        nominal_capacity_ah=nominal_capacity_ah,
        scenarios=scenarios,
        future_days=future_days,
        step_days=step_days,
    )
    forecast = add_forecast_uncertainty_columns(
        forecast,
        cutoff_forecast_validation_summary,
        degradation_model=degradation_model,
        nominal_capacity_ah=nominal_capacity_ah,
    )

    serial_label = str(model_training_table["serial"].iloc[0])
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_base_dir / f"serial={serial_label}" / f"ml_run={run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    write_parquet_chunk(
        model_training_table,
        output_dir / "ml_training_table.parquet",
        compression=compression,
    )
    write_parquet_chunk(
        training_table,
        output_dir / "ml_all_capacity_rows.parquet",
        compression=compression,
    )
    write_parquet_chunk(
        _models_to_frame(models),
        output_dir / "model_summary.parquet",
        compression=compression,
    )
    write_parquet_chunk(
        _degradation_model_to_frame(degradation_model),
        output_dir / "degradation_model_summary.parquet",
        compression=compression,
    )
    write_parquet_chunk(
        forecast,
        output_dir / "capacity_forecast.parquet",
        compression=compression,
    )
    write_parquet_chunk(
        backtest_table,
        output_dir / "model_backtest.parquet",
        compression=compression,
    )
    write_parquet_chunk(
        cutoff_validation_table,
        output_dir / "cutoff_validation.parquet",
        compression=compression,
    )
    write_parquet_chunk(
        cutoff_forecast_validation_table,
        output_dir / "cutoff_forecast_validation.parquet",
        compression=compression,
    )
    write_json(
        output_dir / "model_summary.json",
        {
            "capacity_run_dir": str(capacity_run_dir),
            "nominal_capacity_ah": nominal_capacity_ah,
            "nominal_capacity_source": nominal_capacity_source,
            "bms_soh_reliable_after": str(bms_soh_reliable_after),
            "min_reasonable_capacity_ah": min_reasonable_capacity_ah,
            "max_reasonable_capacity_ah": max_reasonable_capacity_ah,
            "exclude_statistical_outliers": exclude_statistical_outliers,
            "exclude_high_rest_ratio_capacity_rows": exclude_high_rest_ratio_capacity_rows,
            "max_rest_to_discharge_ratio_for_training": max_rest_to_discharge_ratio_for_training,
            "training_cutoff": str(training_cutoff) if training_cutoff is not None else None,
            "validation_end": str(validation_end) if validation_end is not None else None,
            "internal_holdout_fraction": effective_holdout_fraction,
            "degradation_model_type": degradation_model_type,
            "degradation_baseline_capacity_ah": degradation_baseline_capacity_ah,
            "degradation_baseline_method": resolved_baseline_method,
            "degradation_reference_cumulative_ah": degradation_reference_cumulative_ah,
            "degradation_force_zero_intercept": degradation_force_zero_intercept,
            "usage_rate_modes": list(selected_usage_rate_modes),
            "usage_rates": primary_usage_rates,
            "usage_rates_by_model": usage_rates_by_model,
            "scenarios": scenarios or {
                mode: 1.0 for mode in selected_usage_rate_modes
            },
            "best_capacity_ah_model": asdict(best_capacity_ah_model)
            if best_capacity_ah_model is not None
            else None,
            "best_capacity_wh_model": asdict(best_capacity_wh_model)
            if best_capacity_wh_model is not None
            else None,
            "degradation_forecast_model": asdict(degradation_model)
            if degradation_model is not None
            else None,
            "degradation_backtest": backtest_summary,
            "cutoff_validation": cutoff_validation_summary,
            "cutoff_forecast_validation": cutoff_forecast_validation_summary,
            "all_capacity_rows": len(training_table),
            "training_rows": len(model_training_table),
            "valid_training_rows": int(model_training_table["valid_training_row"].sum()),
            "rest_ratio_training_excluded_rows": int(
                model_training_table["rest_ratio_training_excluded"].sum()
            )
            if "rest_ratio_training_excluded" in model_training_table.columns
            else 0,
            "all_rest_ratio_training_excluded_rows": int(
                training_table["rest_ratio_training_excluded"].sum()
            )
            if "rest_ratio_training_excluded" in training_table.columns
            else 0,
            "post_cutoff_rows": len(cutoff_validation_source),
            "post_cutoff_valid_rows": int(
                cutoff_validation_source["valid_training_row"].sum()
            )
            if not cutoff_validation_source.empty
            else 0,
            "forecast_rows": len(forecast),
        },
    )

    logger.info("Capacity ML all capacity rows: %s", len(training_table))
    logger.info(
        "Nominal capacity: %.6f Ah (%s)",
        nominal_capacity_ah,
        nominal_capacity_source,
    )
    logger.info(
        "Forecast usage modes: %s",
        ", ".join(selected_usage_rate_modes),
    )
    for mode, usage_rates in usage_rates_by_model.items():
        logger.info(
            "Forecast usage rate for %s: %.3f Ah/day",
            mode,
            float(usage_rates.get("forecast_ah_per_day", np.nan)),
        )
    logger.info("Capacity ML training rows: %s", len(model_training_table))
    logger.info("Valid capacity ML training rows: %s", int(model_training_table["valid_training_row"].sum()))
    if exclude_high_rest_ratio_capacity_rows:
        excluded_rows = (
            int(model_training_table["rest_ratio_training_excluded"].sum())
            if "rest_ratio_training_excluded" in model_training_table.columns
            else 0
        )
        logger.info(
            "Long-rest capacity filter excluded %s pre-cutoff rows "
            "(rest/discharge ratio > %.3f).",
            excluded_rows,
            max_rest_to_discharge_ratio_for_training,
        )
    if training_cutoff is not None:
        logger.info("Training cutoff: %s", training_cutoff)
        logger.info("Post-cutoff validation rows: %s", len(cutoff_validation_source))
    logger.info(
        "Best capacity Ah model: %s",
        best_capacity_ah_model.model_name if best_capacity_ah_model else None,
    )
    logger.info(
        "Degradation forecast model: %s",
        degradation_model.model_name if degradation_model else None,
    )
    if backtest_summary.get("backtest_available"):
        logger.info(
            "Backtest holdout MAE: %.3f Ah (%.3f SOH pct)",
            backtest_summary["holdout_mae_ah"],
            backtest_summary["holdout_mae_soh_pct"],
        )
    if cutoff_validation_summary.get("validation_available"):
        logger.info(
            "Cutoff validation MAE: %.3f Ah (%.3f SOH pct)",
            cutoff_validation_summary["mae_ah"],
            cutoff_validation_summary["mae_soh_pct"],
        )
    logger.info("Wrote capacity ML outputs to: %s", output_dir)

    return CapacityMLResult(
        training_rows=len(model_training_table),
        valid_training_rows=int(model_training_table["valid_training_row"].sum()),
        forecast_rows=len(forecast),
        best_capacity_ah_model=best_capacity_ah_model.model_name
        if best_capacity_ah_model
        else None,
        output_dir=output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train local capacity trend models and build future usage forecasts.",
    )
    parser.add_argument("--serial", default=None, help="Serial to process when --capacity-run-dir is omitted.")
    parser.add_argument("--capacity-run-dir", type=Path, default=None, help="Specific capacity_run folder or capacity_measurements parquet.")
    parser.add_argument("--capacity-trend-dir", type=Path, default=None, help="Override capacity trend base directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override capacity ML output base directory.")
    parser.add_argument("--env-file", type=Path, default=None, help="Path to a .env file.")
    parser.add_argument(
        "--nominal-capacity-ah",
        type=float,
        default=None,
        help=(
            "SOH reference capacity in Ah. When omitted, the first measured "
            "full-cycle capacity_ah from capacity_measurements.parquet is used."
        ),
    )
    parser.add_argument(
        "--bms-soh-reliable-after",
        default=None,
        help="UTC timestamp after which BMS SOH is considered reliable.",
    )
    parser.add_argument("--min-capacity-ah", type=float, default=None, help="Minimum reasonable measured capacity.")
    parser.add_argument("--max-capacity-ah", type=float, default=None, help="Maximum reasonable measured capacity.")
    parser.add_argument(
        "--exclude-statistical-outliers",
        action="store_true",
        help="Exclude MAD-based capacity outliers from model fitting.",
    )
    rest_filter_group = parser.add_mutually_exclusive_group()
    rest_filter_group.add_argument(
        "--exclude-high-rest-ratio-capacity-rows",
        action="store_true",
        default=None,
        help=(
            "Exclude full-cycle capacity rows from fitting when "
            "rest_duration_seconds / discharge_duration_seconds exceeds the "
            "configured threshold."
        ),
    )
    rest_filter_group.add_argument(
        "--include-high-rest-ratio-capacity-rows",
        action="store_true",
        default=None,
        help=(
            "Disable the high rest/discharge ratio training filter, even if it "
            "is enabled in the environment."
        ),
    )
    parser.add_argument(
        "--max-rest-to-discharge-ratio",
        type=float,
        default=None,
        help=(
            "Rest/discharge duration ratio above which capacity rows are "
            "excluded when --exclude-high-rest-ratio-capacity-rows is active. "
            "Default is 1.0."
        ),
    )
    parser.add_argument("--future-days", type=int, default=365, help="Forecast horizon in days.")
    parser.add_argument("--step-days", type=int, default=30, help="Forecast interval in days.")
    parser.add_argument(
        "--training-cutoff",
        default=None,
        help=(
            "UTC timestamp. Rows after this timestamp are excluded from all fitting "
            "and written as prediction-only validation rows."
        ),
    )
    parser.add_argument(
        "--validation-end",
        default=None,
        help="Optional UTC timestamp limiting the post-cutoff validation window.",
    )
    parser.add_argument(
        "--internal-holdout-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of pre-cutoff rows to hide for internal holdout. "
            "Defaults to 0.2 normally, and 0.0 when --training-cutoff is used."
        ),
    )
    parser.add_argument(
        "--degradation-model",
        default="usage_linear",
        choices=("usage_linear", "usage_calendar_constrained"),
        help=(
            "Degradation forecast model. usage_calendar_constrained fits "
            "nonnegative capacity-loss terms for usage and calendar age."
        ),
    )
    parser.add_argument(
        "--degradation-baseline-capacity-ah",
        type=float,
        default=None,
        help=(
            "Optional explicit baseline capacity Ah for degradation loss modeling, "
            "for example 48.951."
        ),
    )
    parser.add_argument(
        "--degradation-baseline-method",
        default=None,
        choices=("first", "max", "p95"),
        help=(
            "Baseline method when no explicit baseline is provided. Defaults to "
            "first for usage_calendar_constrained and max for usage_linear."
        ),
    )
    parser.add_argument(
        "--degradation-reference-cumulative-ah",
        type=float,
        default=None,
        help=(
            "Optional cumulative_all_discharge_ah reference for degradation loss "
            "features. When omitted, the minimum cumulative Ah in the fitted "
            "degradation rows is used."
        ),
    )
    parser.add_argument(
        "--degradation-force-zero-intercept",
        action="store_true",
        help=(
            "Force the degradation loss model through zero loss at the configured "
            "reference cumulative Ah. Use with an explicit baseline to model a "
            "fresh/reference capacity point."
        ),
    )
    parser.add_argument(
        "--usage-rate-mode",
        default=None,
        choices=("historical_mean", "historical_median", "recent_median"),
        help=(
            "Single future usage-rate assumption. If omitted, all usage-rate modes "
            "are forecast together."
        ),
    )
    parser.add_argument(
        "--usage-rate-modes",
        default=None,
        help=(
            "Comma-separated usage-rate modes to forecast together. Default: "
            "historical_mean,historical_median,recent_median."
        ),
    )
    parser.add_argument(
        "--recent-usage-days",
        type=int,
        default=None,
        help="Lookback window in days for --usage-rate-mode recent_median.",
    )
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Usage scenarios as label=multiplier,label=multiplier.",
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=("snappy", "zstd", "gzip", "none"),
        help="Parquet compression codec.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console log level.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    try:
        storage_config = StorageConfig.from_env(
            args.env_file,
            capacity_trend_dir=args.capacity_trend_dir,
            capacity_ml_dir=args.output_dir,
        )
        model_config = CapacityModelConfig.from_env(args.env_file)

        if args.capacity_run_dir is not None:
            capacity_run_dir = args.capacity_run_dir
        elif args.serial is not None:
            capacity_run_dir = find_latest_capacity_run(
                storage_config.capacity_trend_dir,
                args.serial,
            )
        else:
            raise ValueError("Provide either --serial or --capacity-run-dir.")

        compression = None if args.compression == "none" else args.compression
        reliable_after = parse_utc_timestamp(
            args.bms_soh_reliable_after or model_config.bms_soh_reliable_after
        )
        training_cutoff = (
            parse_utc_timestamp(args.training_cutoff)
            if args.training_cutoff
            else None
        )
        validation_end = (
            parse_utc_timestamp(args.validation_end)
            if args.validation_end
            else None
        )
        if args.usage_rate_mode is not None and args.usage_rate_modes is not None:
            raise ValueError("Use either --usage-rate-mode or --usage-rate-modes, not both.")
        usage_rate_modes = (
            (args.usage_rate_mode,)
            if args.usage_rate_mode is not None
            else parse_usage_rate_modes(args.usage_rate_modes or model_config.usage_rate_modes)
        )
        if args.exclude_high_rest_ratio_capacity_rows:
            exclude_high_rest_ratio_capacity_rows = True
        elif args.include_high_rest_ratio_capacity_rows:
            exclude_high_rest_ratio_capacity_rows = False
        else:
            exclude_high_rest_ratio_capacity_rows = (
                model_config.exclude_high_rest_ratio_capacity_rows
            )
        max_rest_to_discharge_ratio = (
            args.max_rest_to_discharge_ratio
            if args.max_rest_to_discharge_ratio is not None
            else model_config.max_rest_to_discharge_ratio_for_training
        )
        build_capacity_ml_forecast(
            capacity_run_dir=capacity_run_dir,
            output_base_dir=storage_config.capacity_ml_dir,
            nominal_capacity_ah=(
                args.nominal_capacity_ah
                if args.nominal_capacity_ah is not None
                else model_config.nominal_capacity_ah
            ),
            bms_soh_reliable_after=reliable_after,
            min_reasonable_capacity_ah=args.min_capacity_ah
            if args.min_capacity_ah is not None
            else model_config.min_reasonable_capacity_ah,
            max_reasonable_capacity_ah=args.max_capacity_ah
            if args.max_capacity_ah is not None
            else model_config.max_reasonable_capacity_ah,
            exclude_statistical_outliers=args.exclude_statistical_outliers,
            exclude_high_rest_ratio_capacity_rows=exclude_high_rest_ratio_capacity_rows,
            max_rest_to_discharge_ratio_for_training=max_rest_to_discharge_ratio,
            training_cutoff=training_cutoff,
            validation_end=validation_end,
            internal_holdout_fraction=args.internal_holdout_fraction,
            degradation_model_type=args.degradation_model,
            degradation_baseline_capacity_ah=args.degradation_baseline_capacity_ah,
            degradation_baseline_method=args.degradation_baseline_method,
            degradation_reference_cumulative_ah=args.degradation_reference_cumulative_ah,
            degradation_force_zero_intercept=args.degradation_force_zero_intercept,
            usage_rate_modes=usage_rate_modes,
            usage_rate_mode=args.usage_rate_mode or model_config.usage_rate_mode,
            recent_usage_days=args.recent_usage_days
            if args.recent_usage_days is not None
            else model_config.recent_usage_days,
            scenarios=parse_scenarios(args.scenarios),
            future_days=args.future_days,
            step_days=args.step_days,
            compression=compression,
        )
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Capacity ML build failed.")
        else:
            logger.error("Capacity ML build failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
