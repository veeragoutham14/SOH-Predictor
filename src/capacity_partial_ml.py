from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.capacity_ml import (
    DEFAULT_USAGE_RATE_MODES,
    add_model_features,
    add_forecast_uncertainty_columns,
    build_cutoff_forecast_validation_table,
    build_cutoff_validation_table,
    build_degradation_backtest_table,
    build_forecasts_for_usage_rate_modes,
    choose_best_model,
    estimate_usage_rates_with_mode,
    filter_usage_ledger_for_training,
    find_latest_capacity_run,
    fit_candidate_models,
    fit_degradation_usage_model,
    parse_usage_rate_modes,
    parse_utc_timestamp,
    resolve_nominal_capacity_ah,
    write_json,
    _degradation_model_to_frame,
    _models_to_frame,
)
from src.config import CapacityModelConfig, StorageConfig
from src.io_utils import write_parquet_chunk
from src.logging_utils import configure_logging


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PartialCapacityMLResult:
    partial_measurement_rows: int
    valid_partial_measurement_rows: int
    aggregate_rows: int
    training_rows: int
    valid_training_rows: int
    forecast_rows: int
    output_dir: Path


def _as_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    work = df.copy()
    for column in columns:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    return work


def _load_capacity_run_inputs(
    capacity_run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    """Load full-cycle measurements, discharge episodes, and usage ledger."""
    run_dir = capacity_run_dir.parent if capacity_run_dir.is_file() else capacity_run_dir
    measurements_path = run_dir / "capacity_measurements.parquet"
    episodes_path = run_dir / "discharge_episodes.parquet"
    usage_path = run_dir / "discharge_usage_ledger.parquet"

    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing discharge episodes: {episodes_path}")

    measurements = (
        pd.read_parquet(measurements_path) if measurements_path.exists() else pd.DataFrame()
    )
    episodes = pd.read_parquet(episodes_path)
    usage_ledger = pd.read_parquet(usage_path) if usage_path.exists() else pd.DataFrame()
    return measurements, episodes, usage_ledger, run_dir


def _infer_nominal_from_partial(partial: pd.DataFrame) -> tuple[float, str]:
    valid = partial[partial["valid_partial_capacity_row"]].dropna(
        subset=["estimated_capacity_ah", "end_timestamp"]
    )
    if valid.empty:
        raise ValueError(
            "Cannot infer nominal capacity: no valid full-cycle or partial-cycle "
            "capacity estimates are available."
        )
    valid = valid.sort_values("end_timestamp", kind="mergesort")
    return float(valid.iloc[0]["estimated_capacity_ah"]), "first_valid_partial_capacity"


def build_partial_capacity_measurements(
    episodes: pd.DataFrame,
    *,
    nominal_capacity_ah: float,
    min_soc_drop_pct: float,
    min_discharge_ah: float,
    min_discharge_duration_seconds: float,
    max_rest_to_discharge_ratio: float,
    min_reasonable_capacity_ah: float,
    max_reasonable_capacity_ah: float,
) -> pd.DataFrame:
    """Convert discharge episodes into partial-cycle full-capacity estimates."""
    required_columns = {
        "discharge_episode_id",
        "serial",
        "start_timestamp",
        "end_timestamp",
        "start_soc",
        "end_soc",
        "discharge_ah",
        "discharge_wh",
        "discharge_duration_seconds",
        "cumulative_all_discharge_ah",
        "cumulative_all_discharge_wh",
    }
    missing = required_columns.difference(episodes.columns)
    if missing:
        raise ValueError(f"discharge_episodes.parquet is missing columns: {sorted(missing)}")

    work = episodes.copy()
    work["start_timestamp"] = pd.to_datetime(
        work["start_timestamp"], errors="coerce", utc=True
    )
    work["end_timestamp"] = pd.to_datetime(work["end_timestamp"], errors="coerce", utc=True)
    numeric_columns = [
        "discharge_episode_id",
        "serial",
        "start_soc",
        "end_soc",
        "soc_drop",
        "discharge_ah",
        "discharge_wh",
        "discharge_duration_seconds",
        "elapsed_duration_seconds",
        "rest_duration_seconds_inside",
        "cumulative_all_discharge_ah",
        "cumulative_all_discharge_wh",
    ]
    work = _as_numeric(work, numeric_columns)

    if "soc_drop" not in work.columns:
        work["soc_drop"] = work["start_soc"] - work["end_soc"]
    else:
        work["soc_drop"] = work["soc_drop"].fillna(work["start_soc"] - work["end_soc"])

    rest_seconds = (
        work["rest_duration_seconds_inside"]
        if "rest_duration_seconds_inside" in work.columns
        else 0.0
    )
    work["rest_to_discharge_ratio"] = rest_seconds / work[
        "discharge_duration_seconds"
    ].replace(0, np.nan)
    work["rest_to_discharge_ratio"] = work["rest_to_discharge_ratio"].replace(
        [np.inf, -np.inf], np.nan
    )

    soc_fraction = work["soc_drop"] / 100.0
    work["estimated_capacity_ah"] = work["discharge_ah"] / soc_fraction
    work["estimated_capacity_wh"] = work["discharge_wh"] / soc_fraction
    work[["estimated_capacity_ah", "estimated_capacity_wh"]] = work[
        ["estimated_capacity_ah", "estimated_capacity_wh"]
    ].replace([np.inf, -np.inf], np.nan)

    valid = (
        work["start_timestamp"].notna()
        & work["end_timestamp"].notna()
        & work["end_timestamp"].gt(work["start_timestamp"])
        & work["soc_drop"].ge(min_soc_drop_pct)
        & work["discharge_ah"].ge(min_discharge_ah)
        & work["discharge_duration_seconds"].ge(min_discharge_duration_seconds)
        & work["estimated_capacity_ah"].between(
            min_reasonable_capacity_ah,
            max_reasonable_capacity_ah,
            inclusive="both",
        )
    )
    if max_rest_to_discharge_ratio >= 0:
        valid &= work["rest_to_discharge_ratio"].fillna(np.inf).le(
            max_rest_to_discharge_ratio
        )

    work["valid_partial_capacity_row"] = valid
    work["partial_cycle_weight"] = (work["soc_drop"] / 100.0).clip(lower=0.0, upper=1.0)
    work["estimated_soh_pct"] = work["estimated_capacity_ah"] / nominal_capacity_ah * 100.0
    work["partial_capacity_quality"] = np.where(
        work["valid_partial_capacity_row"],
        "valid_partial_discharge",
        "filtered_partial_discharge",
    )

    return work.sort_values("end_timestamp", kind="mergesort").reset_index(drop=True)


def build_partial_capacity_aggregates(
    partial_measurements: pd.DataFrame,
    *,
    nominal_capacity_ah: float,
    bms_soh_reliable_after: pd.Timestamp,
    aggregation_days: int,
    min_episodes_per_bin: int,
) -> pd.DataFrame:
    """Aggregate noisy partial estimates into robust time-binned capacity points."""
    if aggregation_days < 1:
        raise ValueError("aggregation_days must be at least 1.")
    if min_episodes_per_bin < 1:
        raise ValueError("min_episodes_per_bin must be at least 1.")

    valid = partial_measurements[
        partial_measurements["valid_partial_capacity_row"]
    ].dropna(
        subset=[
            "serial",
            "start_timestamp",
            "end_timestamp",
            "estimated_capacity_ah",
            "cumulative_all_discharge_ah",
            "cumulative_all_discharge_wh",
        ]
    ).copy()
    if valid.empty:
        return pd.DataFrame()

    valid = valid.sort_values("end_timestamp", kind="mergesort")
    first_day = valid["end_timestamp"].min().floor("D")
    end_day = valid["end_timestamp"].dt.floor("D")
    bin_index = ((end_day - first_day).dt.days // aggregation_days).astype(int)
    valid["capacity_window_start"] = first_day + pd.to_timedelta(
        bin_index * aggregation_days,
        unit="D",
    )

    grouped = valid.groupby("capacity_window_start", sort=True)
    aggregate = grouped.agg(
        serial=("serial", "first"),
        start_anchor_time=("start_timestamp", "min"),
        end_anchor_time=("end_timestamp", "max"),
        capacity_ah=("estimated_capacity_ah", "median"),
        capacity_wh=("estimated_capacity_wh", "median"),
        estimated_capacity_ah_mean=("estimated_capacity_ah", "mean"),
        estimated_capacity_ah_q25=("estimated_capacity_ah", lambda s: s.quantile(0.25)),
        estimated_capacity_ah_q75=("estimated_capacity_ah", lambda s: s.quantile(0.75)),
        partial_episode_count=("estimated_capacity_ah", "size"),
        median_soc_drop_pct=("soc_drop", "median"),
        max_soc_drop_pct=("soc_drop", "max"),
        mean_rest_to_discharge_ratio=("rest_to_discharge_ratio", "mean"),
        mean_partial_cycle_weight=("partial_cycle_weight", "mean"),
    ).reset_index()

    last_rows = grouped.tail(1)[
        [
            "capacity_window_start",
            "discharge_episode_id",
            "cumulative_all_discharge_ah",
            "cumulative_all_discharge_wh",
        ]
    ].rename(
        columns={
            "discharge_episode_id": "event_discharge_id",
        }
    )
    aggregate = aggregate.merge(last_rows, on="capacity_window_start", how="left")

    first_time = aggregate["end_anchor_time"].min()
    aggregate["calendar_age_days"] = (
        aggregate["end_anchor_time"] - first_time
    ).dt.total_seconds() / 86400.0
    aggregate["row_type"] = "partial_capacity_aggregate"
    aggregate["measured_soh_pct"] = aggregate["capacity_ah"] / nominal_capacity_ah * 100.0
    aggregate["bms_soh_reliable"] = aggregate["end_anchor_time"].ge(
        bms_soh_reliable_after
    )
    aggregate["valid_training_row"] = aggregate["partial_episode_count"].ge(
        min_episodes_per_bin
    )
    aggregate["capacity_ah_iqr"] = (
        aggregate["estimated_capacity_ah_q75"] - aggregate["estimated_capacity_ah_q25"]
    )
    return aggregate.sort_values("end_anchor_time", kind="mergesort").reset_index(
        drop=True
    )


def split_training_and_validation(
    table: pd.DataFrame,
    *,
    training_cutoff: pd.Timestamp | None,
    validation_end: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if training_cutoff is None:
        return table.copy(), pd.DataFrame(columns=table.columns)

    train = table[table["end_anchor_time"] <= training_cutoff].copy()
    validation = table[table["end_anchor_time"] > training_cutoff].copy()
    if validation_end is not None:
        validation = validation[validation["end_anchor_time"] <= validation_end].copy()
    return train, validation


def build_partial_capacity_ml_forecast(
    *,
    capacity_run_dir: Path,
    output_base_dir: Path,
    nominal_capacity_ah: float | None,
    bms_soh_reliable_after: pd.Timestamp,
    min_reasonable_capacity_ah: float,
    max_reasonable_capacity_ah: float,
    min_soc_drop_pct: float = 20.0,
    min_discharge_ah: float = 2.0,
    min_discharge_duration_seconds: float = 300.0,
    max_rest_to_discharge_ratio: float = 0.5,
    aggregation_days: int = 7,
    min_episodes_per_bin: int = 1,
    training_cutoff: pd.Timestamp | None = None,
    validation_end: pd.Timestamp | None = None,
    internal_holdout_fraction: float | None = None,
    degradation_model_type: str = "usage_linear",
    degradation_baseline_capacity_ah: float | None = None,
    degradation_baseline_method: str | None = None,
    usage_rate_modes: Sequence[str] = DEFAULT_USAGE_RATE_MODES,
    usage_rate_mode: str = "historical_mean",
    recent_usage_days: int = 90,
    future_days: int = 365,
    step_days: int = 30,
    compression: str | None = "snappy",
    run_id: str | None = None,
) -> PartialCapacityMLResult:
    """Build a partial-cycle capacity forecast from discharge episodes."""
    measurements, episodes, usage_ledger, run_dir = _load_capacity_run_inputs(
        capacity_run_dir
    )
    if nominal_capacity_ah is not None:
        resolved_nominal_capacity_ah, nominal_capacity_source = (
            float(nominal_capacity_ah),
            "explicit",
        )
    elif not measurements.empty:
        resolved_nominal_capacity_ah, nominal_capacity_source = resolve_nominal_capacity_ah(
            measurements,
            None,
        )
    else:
        resolved_nominal_capacity_ah = 1.0
        nominal_capacity_source = "temporary_partial_reference"

    partial_measurements = build_partial_capacity_measurements(
        episodes,
        nominal_capacity_ah=resolved_nominal_capacity_ah,
        min_soc_drop_pct=min_soc_drop_pct,
        min_discharge_ah=min_discharge_ah,
        min_discharge_duration_seconds=min_discharge_duration_seconds,
        max_rest_to_discharge_ratio=max_rest_to_discharge_ratio,
        min_reasonable_capacity_ah=min_reasonable_capacity_ah,
        max_reasonable_capacity_ah=max_reasonable_capacity_ah,
    )

    if nominal_capacity_source == "temporary_partial_reference":
        resolved_nominal_capacity_ah, nominal_capacity_source = _infer_nominal_from_partial(
            partial_measurements
        )
        partial_measurements["estimated_soh_pct"] = (
            partial_measurements["estimated_capacity_ah"]
            / resolved_nominal_capacity_ah
            * 100.0
        )

    aggregate_table = build_partial_capacity_aggregates(
        partial_measurements,
        nominal_capacity_ah=resolved_nominal_capacity_ah,
        bms_soh_reliable_after=bms_soh_reliable_after,
        aggregation_days=aggregation_days,
        min_episodes_per_bin=min_episodes_per_bin,
    )
    if aggregate_table.empty:
        raise ValueError("No valid partial-cycle capacity estimates are available.")
    aggregate_table = add_model_features(aggregate_table)

    model_training_table, cutoff_validation_source = split_training_and_validation(
        aggregate_table,
        training_cutoff=training_cutoff,
        validation_end=validation_end,
    )
    if model_training_table.empty:
        raise ValueError("No partial capacity rows are available before the cutoff.")

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
    )

    backtest_table, backtest_summary = build_degradation_backtest_table(
        model_training_table,
        nominal_capacity_ah=resolved_nominal_capacity_ah,
        holdout_fraction=effective_holdout_fraction,
        degradation_model_type=degradation_model_type,
        baseline_capacity_ah=degradation_baseline_capacity_ah,
        baseline_method=resolved_baseline_method,
    )
    cutoff_validation_table, cutoff_validation_summary = build_cutoff_validation_table(
        cutoff_validation_source,
        best_capacity_ah_model=best_capacity_ah_model,
        degradation_model=degradation_model,
        nominal_capacity_ah=resolved_nominal_capacity_ah,
    )
    cutoff_forecast_validation_table, cutoff_forecast_validation_summary = (
        build_cutoff_forecast_validation_table(
            cutoff_validation_source,
            model_training_table,
            usage_ledger_for_training,
            usage_rates_by_model,
            best_capacity_ah_model=best_capacity_ah_model,
            degradation_model=degradation_model,
            nominal_capacity_ah=resolved_nominal_capacity_ah,
            training_cutoff=training_cutoff,
        )
    )
    forecast = build_forecasts_for_usage_rate_modes(
        model_training_table,
        usage_rates_by_model,
        best_capacity_ah_model=best_capacity_ah_model,
        best_capacity_wh_model=best_capacity_wh_model,
        degradation_model=degradation_model,
        nominal_capacity_ah=resolved_nominal_capacity_ah,
        scenarios=None,
        future_days=future_days,
        step_days=step_days,
    )
    forecast = add_forecast_uncertainty_columns(
        forecast,
        cutoff_forecast_validation_summary,
        degradation_model=degradation_model,
        nominal_capacity_ah=resolved_nominal_capacity_ah,
    )

    serial_label = str(aggregate_table["serial"].iloc[0])
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_base_dir / f"serial={serial_label}" / f"partial_ml_run={run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    write_parquet_chunk(
        partial_measurements,
        output_dir / "partial_capacity_measurements.parquet",
        compression=compression,
    )
    write_parquet_chunk(
        aggregate_table,
        output_dir / "partial_capacity_all_rows.parquet",
        compression=compression,
    )
    write_parquet_chunk(
        model_training_table,
        output_dir / "partial_capacity_training_table.parquet",
        compression=compression,
    )
    write_parquet_chunk(
        cutoff_validation_source,
        output_dir / "partial_capacity_validation_source.parquet",
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
        output_dir / "partial_capacity_forecast.parquet",
        compression=compression,
    )
    write_parquet_chunk(
        backtest_table,
        output_dir / "model_backtest.parquet",
        compression=compression,
    )
    write_parquet_chunk(
        cutoff_validation_table,
        output_dir / "partial_cutoff_validation.parquet",
        compression=compression,
    )
    write_parquet_chunk(
        cutoff_forecast_validation_table,
        output_dir / "partial_cutoff_forecast_validation.parquet",
        compression=compression,
    )

    write_json(
        output_dir / "model_summary.json",
        {
            "capacity_run_dir": str(run_dir),
            "model_family": "partial_discharge_capacity",
            "nominal_capacity_ah": resolved_nominal_capacity_ah,
            "nominal_capacity_source": nominal_capacity_source,
            "bms_soh_reliable_after": str(bms_soh_reliable_after),
            "min_reasonable_capacity_ah": min_reasonable_capacity_ah,
            "max_reasonable_capacity_ah": max_reasonable_capacity_ah,
            "partial_filters": {
                "min_soc_drop_pct": min_soc_drop_pct,
                "min_discharge_ah": min_discharge_ah,
                "min_discharge_duration_seconds": min_discharge_duration_seconds,
                "max_rest_to_discharge_ratio": max_rest_to_discharge_ratio,
                "aggregation_days": aggregation_days,
                "min_episodes_per_bin": min_episodes_per_bin,
            },
            "training_cutoff": str(training_cutoff) if training_cutoff is not None else None,
            "validation_end": str(validation_end) if validation_end is not None else None,
            "internal_holdout_fraction": effective_holdout_fraction,
            "degradation_model_type": degradation_model_type,
            "degradation_baseline_capacity_ah": degradation_baseline_capacity_ah,
            "degradation_baseline_method": resolved_baseline_method,
            "usage_rate_modes": list(selected_usage_rate_modes),
            "usage_rates": primary_usage_rates,
            "usage_rates_by_model": usage_rates_by_model,
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
            "partial_measurement_rows": len(partial_measurements),
            "valid_partial_measurement_rows": int(
                partial_measurements["valid_partial_capacity_row"].sum()
            ),
            "partial_aggregate_rows": len(aggregate_table),
            "training_rows": len(model_training_table),
            "valid_training_rows": int(model_training_table["valid_training_row"].sum()),
            "post_cutoff_rows": len(cutoff_validation_source),
            "post_cutoff_valid_rows": int(
                cutoff_validation_source["valid_training_row"].sum()
            )
            if not cutoff_validation_source.empty
            else 0,
            "forecast_rows": len(forecast),
        },
    )

    logger.info("Partial capacity measurement rows: %s", len(partial_measurements))
    logger.info(
        "Valid partial capacity measurement rows: %s",
        int(partial_measurements["valid_partial_capacity_row"].sum()),
    )
    logger.info("Partial aggregate rows: %s", len(aggregate_table))
    logger.info("Partial training rows: %s", len(model_training_table))
    logger.info(
        "Partial degradation model: %s",
        degradation_model.model_name if degradation_model else None,
    )
    if degradation_model is not None:
        logger.info(
            "Partial degradation slope: %.6f Ah / 1000 Ah",
            degradation_model.loss_slope_per_1000ah,
        )
    logger.info("Wrote partial capacity ML outputs to: %s", output_dir)

    return PartialCapacityMLResult(
        partial_measurement_rows=len(partial_measurements),
        valid_partial_measurement_rows=int(
            partial_measurements["valid_partial_capacity_row"].sum()
        ),
        aggregate_rows=len(aggregate_table),
        training_rows=len(model_training_table),
        valid_training_rows=int(model_training_table["valid_training_row"].sum()),
        forecast_rows=len(forecast),
        output_dir=output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a partial-discharge SOH forecast model from discharge episodes.",
    )
    parser.add_argument("--serial", default=None, help="Serial to process.")
    parser.add_argument("--capacity-run-dir", type=Path, default=None)
    parser.add_argument("--capacity-trend-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--nominal-capacity-ah", type=float, default=None)
    parser.add_argument("--bms-soh-reliable-after", default=None)
    parser.add_argument("--min-capacity-ah", type=float, default=None)
    parser.add_argument("--max-capacity-ah", type=float, default=None)
    parser.add_argument("--min-soc-drop-pct", type=float, default=20.0)
    parser.add_argument("--min-discharge-ah", type=float, default=2.0)
    parser.add_argument("--min-discharge-duration-seconds", type=float, default=300.0)
    parser.add_argument("--max-rest-to-discharge-ratio", type=float, default=0.5)
    parser.add_argument("--aggregation-days", type=int, default=7)
    parser.add_argument("--min-episodes-per-bin", type=int, default=1)
    parser.add_argument("--training-cutoff", default=None)
    parser.add_argument("--validation-end", default=None)
    parser.add_argument("--internal-holdout-fraction", type=float, default=None)
    parser.add_argument(
        "--degradation-model",
        choices=["usage_linear", "usage_calendar_constrained"],
        default="usage_linear",
    )
    parser.add_argument("--degradation-baseline-capacity-ah", type=float, default=None)
    parser.add_argument(
        "--degradation-baseline-method",
        choices=["max", "first", "p95"],
        default=None,
    )
    usage_group = parser.add_mutually_exclusive_group()
    usage_group.add_argument(
        "--usage-rate-mode",
        choices=list(DEFAULT_USAGE_RATE_MODES),
        default=None,
    )
    usage_group.add_argument("--usage-rate-modes", default=None)
    parser.add_argument("--recent-usage-days", type=int, default=None)
    parser.add_argument("--future-days", type=int, default=365)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    try:
        storage_config = StorageConfig.from_env(
            args.env_file,
            capacity_trend_dir=args.capacity_trend_dir,
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
            parser.error("Provide --serial or --capacity-run-dir.")

        reliable_after = parse_utc_timestamp(
            args.bms_soh_reliable_after or model_config.bms_soh_reliable_after
        )
        training_cutoff = (
            parse_utc_timestamp(args.training_cutoff)
            if args.training_cutoff
            else None
        )
        validation_end = (
            parse_utc_timestamp(args.validation_end) if args.validation_end else None
        )
        usage_rate_modes = (
            (args.usage_rate_mode,)
            if args.usage_rate_mode is not None
            else parse_usage_rate_modes(args.usage_rate_modes or model_config.usage_rate_modes)
        )
        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else storage_config.data_dir / "processed" / "partial_capacity_ml"
        )

        result = build_partial_capacity_ml_forecast(
            capacity_run_dir=capacity_run_dir,
            output_base_dir=output_dir,
            nominal_capacity_ah=(
                args.nominal_capacity_ah
                if args.nominal_capacity_ah is not None
                else model_config.nominal_capacity_ah
            ),
            bms_soh_reliable_after=reliable_after,
            min_reasonable_capacity_ah=(
                args.min_capacity_ah
                if args.min_capacity_ah is not None
                else model_config.min_reasonable_capacity_ah
            ),
            max_reasonable_capacity_ah=(
                args.max_capacity_ah
                if args.max_capacity_ah is not None
                else model_config.max_reasonable_capacity_ah
            ),
            min_soc_drop_pct=args.min_soc_drop_pct,
            min_discharge_ah=args.min_discharge_ah,
            min_discharge_duration_seconds=args.min_discharge_duration_seconds,
            max_rest_to_discharge_ratio=args.max_rest_to_discharge_ratio,
            aggregation_days=args.aggregation_days,
            min_episodes_per_bin=args.min_episodes_per_bin,
            training_cutoff=training_cutoff,
            validation_end=validation_end,
            internal_holdout_fraction=args.internal_holdout_fraction,
            degradation_model_type=args.degradation_model,
            degradation_baseline_capacity_ah=args.degradation_baseline_capacity_ah,
            degradation_baseline_method=args.degradation_baseline_method,
            usage_rate_modes=usage_rate_modes,
            usage_rate_mode=args.usage_rate_mode or model_config.usage_rate_mode,
            recent_usage_days=args.recent_usage_days or model_config.recent_usage_days,
            future_days=args.future_days,
            step_days=args.step_days,
            compression=args.compression,
        )
        logger.info("Partial capacity ML output: %s", result.output_dir)
        return 0
    except Exception as exc:
        logger.exception("Partial capacity ML failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
