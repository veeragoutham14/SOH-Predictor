from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import StorageConfig
from src.logging_utils import configure_logging


logger = logging.getLogger(__name__)
SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_label(value: object) -> str:
    label = SAFE_LABEL_RE.sub("_", str(value).strip())
    return label.strip("_") or "unknown"


def _non_overwriting_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find an unused output filename near: {path}")


def default_output_path(data_dir: Path, serial: str | int, partial_run_dir: Path) -> Path:
    run_label = partial_run_dir.name.removeprefix("partial_ml_run=")
    filename = (
        f"partial_capacity_forecast_serial_{_safe_label(serial)}_"
        f"partial_ml_run_{_safe_label(run_label)}.html"
    )
    return _non_overwriting_path(
        data_dir / "validation_plots" / f"serial={_safe_label(serial)}" / filename
    )


def find_latest_partial_run(partial_ml_dir: Path, serial: str | int) -> Path:
    serial_dir = partial_ml_dir / f"serial={serial}"
    if not serial_dir.exists():
        raise FileNotFoundError(f"No partial capacity ML directory found: {serial_dir}")
    runs = sorted(path for path in serial_dir.glob("partial_ml_run=*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No partial_ml_run folders found under: {serial_dir}")
    return runs[-1]


def load_outputs(partial_run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    history_path = partial_run_dir / "partial_capacity_all_rows.parquet"
    forecast_path = partial_run_dir / "partial_capacity_forecast.parquet"
    summary_path = partial_run_dir / "model_summary.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing partial capacity rows: {history_path}")
    if not forecast_path.exists():
        raise FileNotFoundError(f"Missing partial capacity forecast: {forecast_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing model summary: {summary_path}")
    return (
        pd.read_parquet(history_path),
        pd.read_parquet(forecast_path),
        json.loads(summary_path.read_text(encoding="utf-8")),
    )


def _fmt(value: object, digits: int = 2, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(number):
        return "n/a"
    return f"{number:.{digits}f}{suffix}"


def _fmt_month(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return "not reached"
    return pd.Timestamp(value).strftime("%Y-%m")


def _forecast_summary_rows(forecast: pd.DataFrame, summary: dict) -> list[list[str]]:
    rows: list[list[str]] = []
    validation_by_mode = (
        summary.get("cutoff_forecast_validation", {})
        .get("by_usage_rate_model", {})
    )
    for usage_model, group in forecast.groupby("usage_rate_model", sort=True):
        group = group.sort_values("forecast_timestamp", kind="mergesort")
        crossing = group[group["predicted_future_soh_pct"].le(80.0)]
        first_crossing = crossing.iloc[0] if not crossing.empty else None
        last = group.iloc[-1]
        validation = validation_by_mode.get(str(usage_model), {})
        rows.append(
            [
                str(usage_model),
                _fmt(last.get("forecast_ah_per_day"), 2),
                _fmt(validation.get("mae_soh_pct"), 3, "%"),
                _fmt(validation.get("max_abs_soh_error_pct"), 3, "%"),
                _fmt_month(
                    first_crossing["forecast_timestamp"]
                    if first_crossing is not None
                    else None
                ),
                _fmt(
                    first_crossing.get("equivalent_full_cycles")
                    if first_crossing is not None
                    else None,
                    0,
                ),
                _fmt(last.get("predicted_future_soh_pct"), 2, "%"),
            ]
        )
    return rows


def build_figure(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    summary: dict,
    *,
    title: str,
) -> go.Figure:
    history = history.copy()
    forecast = forecast.copy()
    history["end_anchor_time"] = pd.to_datetime(
        history["end_anchor_time"], errors="coerce", utc=True
    )
    forecast["forecast_timestamp"] = pd.to_datetime(
        forecast["forecast_timestamp"], errors="coerce", utc=True
    )
    for column in [
        "capacity_ah",
        "measured_soh_pct",
        "partial_episode_count",
        "median_soc_drop_pct",
    ]:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")
    for column in [
        "predicted_future_soh_pct",
        "predicted_capacity_ah",
        "forecast_ah_per_day",
        "equivalent_full_cycles",
    ]:
        if column in forecast.columns:
            forecast[column] = pd.to_numeric(forecast[column], errors="coerce")

    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        row_heights=[0.72, 0.28],
        vertical_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(
            x=history["end_anchor_time"],
            y=history["measured_soh_pct"],
            mode="markers",
            name="partial-cycle SOH estimate",
            marker=dict(
                size=8,
                color=history.get("median_soc_drop_pct", pd.Series([20] * len(history))),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="median SOC drop %"),
            ),
            customdata=history[
                [
                    "capacity_ah",
                    "partial_episode_count",
                    "median_soc_drop_pct",
                    "capacity_ah_iqr",
                ]
            ]
            if {"capacity_ah", "partial_episode_count", "median_soc_drop_pct", "capacity_ah_iqr"}.issubset(history.columns)
            else None,
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "SOH %{y:.2f}%<br>"
                "capacity %{customdata[0]:.3f} Ah<br>"
                "episodes %{customdata[1]}<br>"
                "median SOC drop %{customdata[2]:.1f}%<br>"
                "capacity IQR %{customdata[3]:.3f} Ah"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    colors = {
        "historical_mean": "#2563eb",
        "historical_median": "#f59e0b",
        "recent_median": "#dc2626",
    }
    for usage_model, group in forecast.groupby("usage_rate_model", sort=True):
        group = group.sort_values("forecast_timestamp", kind="mergesort")
        fig.add_trace(
            go.Scatter(
                x=group["forecast_timestamp"],
                y=group["predicted_future_soh_pct"],
                mode="lines",
                line=dict(width=3, color=colors.get(str(usage_model))),
                name=str(usage_model),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    "SOH %{y:.2f}%<br>"
                    "capacity %{customdata[0]:.3f} Ah<br>"
                    "EFC %{customdata[1]:.0f}<extra></extra>"
                ),
                customdata=group[["predicted_capacity_ah", "equivalent_full_cycles"]],
            ),
            row=1,
            col=1,
        )

    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="#64748b",
        annotation_text="80% SOH threshold",
        row=1,
        col=1,
    )

    table_rows = _forecast_summary_rows(forecast, summary)
    headers = ["Usage model", "Ah/day", "Val MAE", "Max err", "80% month", "80% EFC", "10y SOH"]
    fig.add_trace(
        go.Table(
            header=dict(values=headers, fill_color="#e5e7eb", align="center"),
            cells=dict(
                values=list(map(list, zip(*table_rows))) if table_rows else [[] for _ in headers],
                fill_color="#f8fafc",
                align="center",
            ),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=850,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="SOH %", row=1, col=1)
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot partial-cycle capacity forecast as interactive HTML.",
    )
    parser.add_argument("--serial", default=None)
    parser.add_argument("--partial-run-dir", type=Path, default=None)
    parser.add_argument("--partial-ml-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    try:
        storage_config = StorageConfig.from_env(args.env_file)
        partial_ml_dir = (
            args.partial_ml_dir
            if args.partial_ml_dir is not None
            else storage_config.data_dir / "processed" / "partial_capacity_ml"
        )
        if args.partial_run_dir is not None:
            partial_run_dir = args.partial_run_dir
            serial = args.serial or partial_run_dir.parent.name.removeprefix("serial=")
        elif args.serial is not None:
            serial = args.serial
            partial_run_dir = find_latest_partial_run(partial_ml_dir, serial)
        else:
            parser.error("Provide --serial or --partial-run-dir.")

        history, forecast, summary = load_outputs(partial_run_dir)
        run_label = partial_run_dir.name.removeprefix("partial_ml_run=")
        output = args.output or default_output_path(
            storage_config.data_dir,
            serial,
            partial_run_dir,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig = build_figure(
            history,
            forecast,
            summary,
            title=args.title
            or f"Serial {serial} Partial-Cycle SOH Forecast - partial_ml_run {run_label}",
        )
        fig.write_html(output, include_plotlyjs="cdn")
        logger.info("Wrote partial capacity forecast HTML: %s", output)
        return 0
    except Exception as exc:
        logger.exception("Partial capacity forecast plot failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
