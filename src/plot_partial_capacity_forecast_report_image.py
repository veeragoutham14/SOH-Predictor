from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

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


def find_latest_partial_run(partial_ml_dir: Path, serial: str | int) -> Path:
    serial_dir = partial_ml_dir / f"serial={serial}"
    if not serial_dir.exists():
        raise FileNotFoundError(f"No partial capacity ML directory found: {serial_dir}")
    runs = sorted(path for path in serial_dir.glob("partial_ml_run=*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No partial_ml_run folders found under: {serial_dir}")
    return runs[-1]


def default_output_path(data_dir: Path, serial: str | int, partial_run_dir: Path) -> Path:
    run_label = partial_run_dir.name.removeprefix("partial_ml_run=")
    filename = (
        f"partial_capacity_forecast_report_serial_{_safe_label(serial)}_"
        f"partial_ml_run_{_safe_label(run_label)}.png"
    )
    return _non_overwriting_path(
        data_dir / "validation_plots" / f"serial={_safe_label(serial)}" / filename
    )


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


def _fmt_signed(value: object, digits: int = 2, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(number):
        return "n/a"
    return f"{number:+.{digits}f}{suffix}"


def _fmt_month(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return "not reached"
    return pd.Timestamp(value).strftime("%Y-%m")


def _fmt_uncertainty_days(value: object) -> str:
    try:
        days = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(days):
        return "n/a"
    if abs(days) >= 60:
        return f"+/-{days / 30.4375:.1f}mo"
    return f"+/-{days:.0f}d"


def prepare_tables(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        "capacity_ah_iqr",
    ]:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")
    for column in [
        "predicted_future_soh_pct",
        "predicted_capacity_ah",
        "forecast_ah_per_day",
        "equivalent_full_cycles",
        "projected_soh_error_pct_slope",
        "estimated_80_date_uncertainty_days",
        "usage_error_daily_ah_slope",
    ]:
        if column in forecast.columns:
            forecast[column] = pd.to_numeric(forecast[column], errors="coerce")
    return (
        history.dropna(subset=["end_anchor_time", "measured_soh_pct"]).sort_values(
            "end_anchor_time", kind="mergesort"
        ),
        forecast.dropna(subset=["forecast_timestamp", "predicted_future_soh_pct"]).sort_values(
            ["usage_rate_model", "forecast_timestamp"], kind="mergesort"
        ),
    )


def forecast_summary_rows(forecast: pd.DataFrame, summary: dict) -> list[list[str]]:
    validation_by_mode = (
        summary.get("cutoff_forecast_validation", {})
        .get("by_usage_rate_model", {})
    )
    rows: list[list[str]] = []
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
                _fmt_signed(last.get("usage_error_daily_ah_slope"), 2),
                _fmt_month(
                    first_crossing["forecast_timestamp"]
                    if first_crossing is not None
                    else None
                ),
                _fmt(
                    first_crossing.get("projected_soh_error_pct_slope")
                    if first_crossing is not None
                    else None,
                    2,
                    "%",
                ),
                _fmt_uncertainty_days(
                    first_crossing.get("estimated_80_date_uncertainty_days")
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


def draw_report(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    summary: dict,
    *,
    output: Path,
    title: str,
) -> None:
    history, forecast = prepare_tables(history, forecast)
    fig = plt.figure(figsize=(18, 10), dpi=160)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.45], width_ratios=[1, 1])
    ax = fig.add_subplot(gs[0, :])
    ax_info = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1])

    colors = {
        "historical_mean": "#2563eb",
        "historical_median": "#f59e0b",
        "recent_median": "#dc2626",
    }
    ax.scatter(
        history["end_anchor_time"],
        history["measured_soh_pct"],
        s=(history.get("partial_episode_count", pd.Series([1] * len(history))) * 10).clip(
            lower=18,
            upper=90,
        ),
        color="#334155",
        alpha=0.62,
        label="partial-cycle SOH estimate",
    )
    for usage_model, group in forecast.groupby("usage_rate_model", sort=True):
        group = group.sort_values("forecast_timestamp", kind="mergesort")
        ax.plot(
            group["forecast_timestamp"],
            group["predicted_future_soh_pct"],
            linewidth=2.4,
            color=colors.get(str(usage_model)),
            label=str(usage_model),
        )
        crossing = group[group["predicted_future_soh_pct"].le(80.0)]
        if not crossing.empty:
            point = crossing.iloc[0]
            ax.scatter(
                [point["forecast_timestamp"]],
                [point["predicted_future_soh_pct"]],
                marker="D",
                s=64,
                color=colors.get(str(usage_model)),
                zorder=5,
            )
            ax.annotate(
                pd.Timestamp(point["forecast_timestamp"]).strftime("%Y-%m"),
                xy=(point["forecast_timestamp"], point["predicted_future_soh_pct"]),
                xytext=(0, 18),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=colors.get(str(usage_model)),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
            )

    ax.axhline(80, color="#64748b", linestyle="--", linewidth=1.5, label="80% SOH threshold")
    ax.set_title("Partial-Cycle Capacity-Based SOH Forecast", loc="left", fontsize=16, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("SOH %")
    ax.grid(True, color="#e2e8f0")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(title, x=0.02, ha="left", fontsize=22, weight="bold")

    ax_info.axis("off")
    degradation = summary.get("degradation_forecast_model") or {}
    filters = summary.get("partial_filters") or {}
    cutoff = summary.get("cutoff_validation") or {}
    info_lines = [
        ("training cutoff", str(summary.get("training_cutoff", "n/a"))[:10]),
        (
            "validation",
            f"{str(cutoff.get('start_time', 'n/a'))[:10]} to {str(cutoff.get('end_time', 'n/a'))[:10]}",
        ),
        ("partial episodes", str(summary.get("valid_partial_measurement_rows", "n/a"))),
        ("aggregate rows", str(summary.get("partial_aggregate_rows", "n/a"))),
        ("aggregation", f"{filters.get('aggregation_days', 'n/a')} days"),
        ("min SOC drop", f"{filters.get('min_soc_drop_pct', 'n/a')}%"),
        (
            "MAE",
            f"{_fmt(cutoff.get('mae_ah'), 3)} Ah / {_fmt(cutoff.get('mae_soh_pct'), 3)}% SOH",
        ),
        ("model", degradation.get("model_name", "n/a")),
        ("baseline", f"{_fmt(degradation.get('baseline_capacity_ah'), 3)} Ah"),
        (
            "loss slope",
            f"{_fmt(degradation.get('loss_slope_per_1000ah'), 4)} Ah / 1000 Ah",
        ),
    ]
    info_text = "\n".join(f"{label:<17}: {value}" for label, value in info_lines)
    ax_info.text(
        0,
        1,
        info_text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )
    ax_info.set_title("Validation and Model Setup", loc="left", fontsize=14, weight="bold")

    ax_table.axis("off")
    headers = [
        "Usage\nmodel",
        "Ah/d",
        "Val\nMAE",
        "Max\nerr",
        "Ah\nerr/d",
        "80%\nmonth",
        "80%\n+/-",
        "Date\n+/-",
        "80%\nEFC",
        "10y\nSOH",
    ]
    rows = forecast_summary_rows(forecast, summary)
    table = ax_table.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="upper left",
        bbox=[0, 0, 1, 0.88],
        colWidths=[0.21, 0.075, 0.085, 0.085, 0.085, 0.105, 0.09, 0.09, 0.08, 0.09],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#94a3b8")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold", color="#0f172a")
        else:
            cell.set_facecolor("#f8fafc" if row % 2 else "#eef2f7")
    ax_table.set_title("Usage-Rate Model Summary", loc="left", fontsize=14, weight="bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a PNG report for a partial-cycle capacity forecast.",
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
        draw_report(
            history,
            forecast,
            summary,
            output=output,
            title=args.title
            or f"Serial {serial} Partial-Cycle SOH Forecast Report - partial_ml_run {run_label}",
        )
        logger.info("Wrote partial capacity forecast report image: %s", output)
        return 0
    except Exception as exc:
        logger.exception("Partial capacity forecast report failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
