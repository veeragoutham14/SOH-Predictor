from __future__ import annotations

import argparse
import html
import json
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from src.config import StorageConfig
from src.discharge_cycle_builder import MODE_DISCHARGING, MODE_MISSING, load_event_table
from src.io_utils import parse_timestamp
from src.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def parse_loss_slopes(value: str) -> tuple[float, ...]:
    try:
        slopes = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("loss slopes must be comma-separated numbers") from exc
    if not slopes or any(slope <= 0 for slope in slopes):
        raise ValueError("provide at least one positive loss slope")
    if len(set(slopes)) != len(slopes):
        raise ValueError("loss slopes must be unique")
    return slopes


def _parse_cli_timestamp(value: str) -> datetime:
    try:
        return parse_timestamp(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _scenario_name(slope: float) -> str:
    return f"slope_{slope:.12g}"


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return value


def build_usage_observations(
    events: pd.DataFrame,
    *,
    aggregation: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    required = {
        "mode",
        "end_timestamp",
        "duration_seconds",
        "throughput_ah",
        "end_soh",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"event table is missing required columns: {missing}")

    work = events.copy().sort_values("end_timestamp", kind="mergesort")
    work["end_timestamp"] = pd.to_datetime(work["end_timestamp"], errors="coerce", utc=True)
    work["duration_seconds"] = pd.to_numeric(work["duration_seconds"], errors="coerce")
    throughput = pd.to_numeric(work["throughput_ah"], errors="coerce").fillna(0.0)
    work["discharge_ah"] = np.where(
        work["mode"].eq(MODE_DISCHARGING),
        throughput.clip(lower=0.0),
        0.0,
    )
    work["cumulative_all_discharge_ah"] = work["discharge_ah"].cumsum()
    work["bms_soh_pct"] = pd.to_numeric(work["end_soh"], errors="coerce").where(
        lambda values: values.between(0.0, 110.0)
    )
    work = work.dropna(subset=["end_timestamp"])
    if work.empty:
        raise ValueError("event table has no valid timestamps")

    day = work["end_timestamp"].dt.floor("D")
    if aggregation == "daily":
        work["bucket"] = day
    elif aggregation == "weekly":
        work["bucket"] = day - pd.to_timedelta(day.dt.weekday, unit="D")
    else:
        raise ValueError(f"unknown aggregation: {aggregation}")

    observations = (
        work.groupby("bucket", sort=True)
        .agg(
            end_anchor_time=("end_timestamp", "max"),
            cumulative_all_discharge_ah=("cumulative_all_discharge_ah", "last"),
            interval_discharge_ah=("discharge_ah", "sum"),
            bms_soh_end=("bms_soh_pct", "median"),
            bms_sample_count=("bms_soh_pct", "count"),
            event_count=("mode", "size"),
        )
        .reset_index(drop=True)
    )
    observations = observations.dropna(subset=["bms_soh_end"]).reset_index(drop=True)
    if len(observations) < 2:
        raise ValueError("at least two daily or weekly BMS SOH observations are required")

    span_seconds = max(
        0.0,
        (work["end_timestamp"].max() - work["end_timestamp"].min()).total_seconds(),
    )
    missing_seconds = float(
        work.loc[work["mode"].eq(MODE_MISSING), "duration_seconds"].fillna(0.0).sum()
    )
    quality = {
        "observed_span_seconds": span_seconds,
        "missing_duration_seconds": missing_seconds,
        "telemetry_coverage_pct": (
            100.0 * max(0.0, 1.0 - missing_seconds / span_seconds)
            if span_seconds > 0
            else 0.0
        ),
    }
    return observations, quality


def _scenario_metrics(
    observations: pd.DataFrame,
    predicted_soh_pct: pd.Series,
    *,
    slope: float,
    tolerance_pct: float,
) -> dict[str, float | int | str]:
    actual = observations["bms_soh_end"].to_numpy(dtype=float)
    predicted = predicted_soh_pct.to_numpy(dtype=float)
    prediction_error = predicted - actual
    absolute_error = np.abs(prediction_error)
    return {
        "scenario": _scenario_name(slope),
        "loss_slope_per_1000ah": slope,
        "rows": len(observations),
        "mae_soh_pct": float(np.mean(absolute_error)),
        "rmse_soh_pct": float(np.sqrt(np.mean(prediction_error**2))),
        "prediction_bias_soh_pct": float(np.mean(prediction_error)),
        "max_abs_soh_error_pct": float(np.max(absolute_error)),
        "within_bms_tolerance_pct": float(100.0 * np.mean(absolute_error <= tolerance_pct)),
        "predicted_soh_change_pct": float(predicted[-1] - predicted[0]),
        "bms_soh_change_pct": float(actual[-1] - actual[0]),
        "cumulative_ah_span": float(
            observations["cumulative_all_discharge_ah"].iloc[-1]
            - observations["cumulative_all_discharge_ah"].iloc[0]
        ),
    }


def _write_report_html(
    output_path: Path,
    observations: pd.DataFrame,
    scenarios: dict[str, pd.Series],
    metrics: list[dict[str, Any]],
    *,
    serial: str,
    nominal_capacity_ah: float,
    anchor_mode: str,
) -> None:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=observations["end_anchor_time"],
            y=observations["bms_soh_end"],
            mode="markers+lines",
            name="BMS SOH",
            marker={"size": 5, "color": "#26343d"},
            line={"color": "#7d8a91", "width": 1},
        )
    )
    colors = ("#176b87", "#d99018", "#c45147", "#4776b8", "#667a3e")
    for index, (name, values) in enumerate(scenarios.items()):
        figure.add_trace(
            go.Scatter(
                x=observations["end_anchor_time"],
                y=values,
                mode="lines",
                name=name.replace("slope_", "Slope "),
                line={"color": colors[index % len(colors)], "width": 2},
            )
        )
    figure.update_layout(
        title=f"Battery {serial} transferred-slope verification",
        xaxis_title="Date",
        yaxis_title="SOH %",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.08},
        margin={"l": 60, "r": 30, "t": 100, "b": 60},
    )
    rows = "".join(
        "<tr>"
        f"<td>{float(item['loss_slope_per_1000ah']):.6f}</td>"
        f"<td>{float(item['mae_soh_pct']):.3f}%</td>"
        f"<td>{float(item['rmse_soh_pct']):.3f}%</td>"
        f"<td>{float(item['prediction_bias_soh_pct']):+.3f}%</td>"
        f"<td>{float(item['within_bms_tolerance_pct']):.1f}%</td>"
        "</tr>"
        for item in metrics
    )
    chart = pio.to_html(figure, include_plotlyjs=True, full_html=False)
    output_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Fixed-slope SOH verification</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;color:#1f2a30;margin:28px;}"
        "h1{font-size:24px;margin-bottom:4px}p{color:#66747b}table{border-collapse:collapse;"
        "width:100%;margin-top:24px}th,td{padding:9px 12px;border:1px solid #dce2e5;"
        "text-align:right}th{background:#f4f6f7}th:first-child,td:first-child{text-align:left}"
        "</style></head><body>"
        f"<h1>Battery {html.escape(serial)}</h1>"
        f"<p>Nominal capacity {nominal_capacity_ah:.3f} Ah; anchor {html.escape(anchor_mode)}; "
        "cumulative Ah is calculated from observed discharge events.</p>"
        f"{chart}<table><thead><tr><th>Loss slope Ah/1000 Ah</th><th>MAE</th>"
        "<th>RMSE</th><th>Bias</th><th>Within tolerance</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></body></html>",
        encoding="utf-8",
    )


def _write_report_png(
    output_path: Path,
    observations: pd.DataFrame,
    scenarios: dict[str, pd.Series],
    *,
    serial: str,
) -> None:
    figure, axis = plt.subplots(figsize=(14, 7.5))
    axis.scatter(
        observations["end_anchor_time"],
        observations["bms_soh_end"],
        s=18,
        color="#26343d",
        alpha=0.7,
        label="BMS SOH",
    )
    for name, values in scenarios.items():
        axis.plot(
            observations["end_anchor_time"],
            values,
            linewidth=2,
            label=name.replace("slope_", "Slope "),
        )
    axis.set_title(f"Battery {serial} transferred-slope verification")
    axis.set_xlabel("Date")
    axis.set_ylabel("SOH %")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_fixed_slope_verification(
    events: pd.DataFrame,
    *,
    serial: str,
    nominal_capacity_ah: float,
    loss_slopes_per_1000ah: tuple[float, ...],
    anchor_mode: str,
    reference_cumulative_ah: float,
    bms_tolerance_pct: float,
    aggregation: str,
    output_base_dir: Path,
    module_count: int | None = None,
    slope_reference_module_count: int | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    compression: str | None = "snappy",
    run_id: str | None = None,
) -> Path:
    if nominal_capacity_ah <= 0:
        raise ValueError("nominal_capacity_ah must be positive")
    if module_count is not None and module_count <= 0:
        raise ValueError("module_count must be positive")
    if slope_reference_module_count is not None and slope_reference_module_count <= 0:
        raise ValueError("slope_reference_module_count must be positive")
    observations, quality = build_usage_observations(events, aggregation=aggregation)
    if start is not None:
        observations = observations[
            observations["end_anchor_time"] >= pd.Timestamp(start)
        ]
    if end is not None:
        observations = observations[
            observations["end_anchor_time"] < pd.Timestamp(end)
        ]
    observations = observations.reset_index(drop=True)
    if len(observations) < 2:
        raise ValueError("the selected verification range has fewer than two observations")

    if anchor_mode == "absolute":
        resolved_reference_ah = float(reference_cumulative_ah)
        anchor_soh_pct = 100.0
    elif anchor_mode == "first_bms":
        resolved_reference_ah = float(observations.iloc[0]["cumulative_all_discharge_ah"])
        anchor_soh_pct = float(observations.iloc[0]["bms_soh_end"])
    else:
        raise ValueError(f"unknown anchor mode: {anchor_mode}")

    usage_since_reference_kah = (
        observations["cumulative_all_discharge_ah"] - resolved_reference_ah
    ).clip(lower=0.0) / 1000.0
    scenario_values: dict[str, pd.Series] = {}
    metric_rows: list[dict[str, Any]] = []
    forecast_parts: list[pd.DataFrame] = []
    for slope in loss_slopes_per_1000ah:
        scenario = _scenario_name(slope)
        predicted_soh = anchor_soh_pct - (
            100.0 * slope * usage_since_reference_kah / nominal_capacity_ah
        )
        scenario_values[scenario] = predicted_soh
        metric_rows.append(
            _scenario_metrics(
                observations,
                predicted_soh,
                slope=slope,
                tolerance_pct=bms_tolerance_pct,
            )
        )
        forecast_parts.append(
            pd.DataFrame(
                {
                    "forecast_timestamp": observations["end_anchor_time"],
                    "scenario": scenario,
                    "usage_rate_model": "actual_cumulative_discharge",
                    "forecast_days_after_last_measurement": 0.0,
                    "cumulative_all_discharge_ah": observations[
                        "cumulative_all_discharge_ah"
                    ],
                    "predicted_capacity_ah": nominal_capacity_ah * predicted_soh / 100.0,
                    "predicted_future_soh_pct": predicted_soh,
                    "forecast_ah_per_day": observations["interval_discharge_ah"],
                    "loss_slope_per_1000ah": slope,
                }
            )
        )

    best_metrics = min(metric_rows, key=lambda item: float(item["mae_soh_pct"]))
    best_scenario = str(best_metrics["scenario"])
    best_slope = float(best_metrics["loss_slope_per_1000ah"])
    best_prediction = scenario_values[best_scenario]
    actual_soh = observations["bms_soh_end"]
    actual_capacity = nominal_capacity_ah * actual_soh / 100.0
    predicted_capacity = nominal_capacity_ah * best_prediction / 100.0

    history = pd.DataFrame(
        {
            "serial": serial,
            "end_anchor_time": observations["end_anchor_time"],
            "capacity_ah": actual_capacity,
            "measured_soh_pct": actual_soh,
            "bms_soh_end": actual_soh,
            "cumulative_all_discharge_ah": observations["cumulative_all_discharge_ah"],
            "valid_training_row": True,
            "sample_count": observations["bms_sample_count"],
        }
    )
    validation = pd.DataFrame(
        {
            "end_anchor_time": observations["end_anchor_time"],
            "event_discharge_id": np.arange(1, len(observations) + 1),
            "cumulative_all_discharge_ah": observations["cumulative_all_discharge_ah"],
            "actual_capacity_ah": actual_capacity,
            "predicted_capacity_ah": predicted_capacity,
            "capacity_error_ah": actual_capacity - predicted_capacity,
            "actual_soh_pct": actual_soh,
            "predicted_soh_pct": best_prediction,
            "soh_error_pct": actual_soh - best_prediction,
            "abs_soh_error_pct": (actual_soh - best_prediction).abs(),
        }
    )
    forecast = pd.concat(forecast_parts, ignore_index=True)

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        output_base_dir / f"serial={serial}" / f"verification_run={run_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    history.to_parquet(
        output_dir / "fixed_slope_all_capacity_rows.parquet",
        index=False,
        compression=compression,
    )
    validation.to_parquet(
        output_dir / "cutoff_validation.parquet",
        index=False,
        compression=compression,
    )
    forecast.to_parquet(
        output_dir / "capacity_forecast.parquet",
        index=False,
        compression=compression,
    )

    capacity_error = validation["capacity_error_ah"].to_numpy(dtype=float)
    summary = {
        "model_family": "fixed_slope_transfer_verification",
        "serial": serial,
        "nominal_capacity_ah": nominal_capacity_ah,
        "nominal_capacity_source": "explicit",
        "module_count": module_count,
        "slope_reference_module_count": slope_reference_module_count,
        "anchor_mode": anchor_mode,
        "reference_cumulative_ah": resolved_reference_ah,
        "anchor_soh_pct": anchor_soh_pct,
        "bms_tolerance_pct": bms_tolerance_pct,
        "aggregation": aggregation,
        "degradation_forecast_model": {
            "model_name": "fixed_transferred_usage_slope",
            "baseline_capacity_ah": nominal_capacity_ah * anchor_soh_pct / 100.0,
            "reference_cumulative_ah": resolved_reference_ah,
            "intercept_loss_ah": 0.0,
            "loss_slope_per_1000ah": best_slope,
            "train_rows": 0,
            "residual_std": float(np.std(capacity_error, ddof=1))
            if len(capacity_error) > 1
            else 0.0,
            "baseline_capacity_method": anchor_mode,
            "slope_source": "user_supplied_transfer",
            "slope_reference_module_count": slope_reference_module_count,
        },
        "cutoff_validation": {
            "validation_available": True,
            "rows": len(validation),
            "start_time": observations["end_anchor_time"].min(),
            "end_time": observations["end_anchor_time"].max(),
            "mae_ah": float(np.mean(np.abs(capacity_error))),
            "rmse_ah": float(np.sqrt(np.mean(capacity_error**2))),
            "mae_soh_pct": best_metrics["mae_soh_pct"],
            "max_abs_soh_error_pct": best_metrics["max_abs_soh_error_pct"],
            "capacity_ah_model": "fixed_transferred_usage_slope",
        },
        "slope_comparison": {
            "selection_note": "Best means lowest verification MAE; no slope was fitted.",
            "best_scenario": best_scenario,
            "best_loss_slope_per_1000ah": best_slope,
            "by_slope": {
                str(item["loss_slope_per_1000ah"]): item for item in metric_rows
            },
        },
        "data_quality": quality,
        "actual_cumulative_discharge_ah": float(
            observations["cumulative_all_discharge_ah"].iloc[-1]
        ),
        "validation_rows": len(validation),
        "forecast_rows": len(forecast),
    }
    (output_dir / "model_summary.json").write_text(
        json.dumps(_clean_json(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_report_html(
        output_dir / "fixed_slope_verification_report.html",
        observations,
        scenario_values,
        metric_rows,
        serial=serial,
        nominal_capacity_ah=nominal_capacity_ah,
        anchor_mode=anchor_mode,
    )
    _write_report_png(
        output_dir / "fixed_slope_verification_report.png",
        observations,
        scenario_values,
        serial=serial,
    )
    logger.info("Wrote fixed-slope verification to %s", output_dir)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify transferred capacity-loss slopes against BMS SOH."
    )
    parser.add_argument("--serial", required=True)
    parser.add_argument("--event-dir", type=Path, default=None)
    parser.add_argument("--event-run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--nominal-capacity-ah", type=float, required=True)
    parser.add_argument("--module-count", type=int, default=None)
    parser.add_argument("--slope-reference-module-count", type=int, default=None)
    parser.add_argument("--loss-slopes-per-1000ah", required=True)
    parser.add_argument(
        "--anchor-mode",
        choices=("absolute", "first_bms"),
        default="absolute",
    )
    parser.add_argument("--reference-cumulative-ah", type=float, default=0.0)
    parser.add_argument("--bms-tolerance-pct", type=float, default=1.0)
    parser.add_argument("--aggregation", choices=("daily", "weekly"), default="daily")
    parser.add_argument("--start", type=_parse_cli_timestamp, default=None)
    parser.add_argument("--end", type=_parse_cli_timestamp, default=None)
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=("snappy", "zstd", "gzip", "none"),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    try:
        if args.reference_cumulative_ah < 0:
            raise ValueError("reference_cumulative_ah must be nonnegative")
        if args.bms_tolerance_pct < 0:
            raise ValueError("bms_tolerance_pct must be nonnegative")
        storage = StorageConfig.from_env(args.env_file, event_parquet_dir=args.event_dir)
        event_input = storage.event_parquet_dir / f"serial={args.serial}"
        events = load_event_table(
            event_input,
            serial=args.serial,
            event_run_id=args.event_run_id,
        )
        output_base = (
            args.output_dir
            or storage.data_dir / "processed" / "fixed_slope_verification"
        )
        compression = None if args.compression == "none" else args.compression
        run_fixed_slope_verification(
            events,
            serial=str(args.serial),
            nominal_capacity_ah=args.nominal_capacity_ah,
            loss_slopes_per_1000ah=parse_loss_slopes(args.loss_slopes_per_1000ah),
            anchor_mode=args.anchor_mode,
            reference_cumulative_ah=args.reference_cumulative_ah,
            bms_tolerance_pct=args.bms_tolerance_pct,
            aggregation=args.aggregation,
            output_base_dir=output_base,
            module_count=args.module_count,
            slope_reference_module_count=args.slope_reference_module_count,
            start=args.start,
            end=args.end,
            compression=compression,
        )
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Fixed-slope verification failed")
        else:
            logger.error("Fixed-slope verification failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
