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

SCENARIO_COLORS = {
    "low_usage": "#16a34a",
    "normal_usage": "#f59e0b",
    "high_usage": "#dc2626",
    "historical_mean": "#2563eb",
    "historical_median": "#f59e0b",
    "recent_median": "#dc2626",
}


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


def find_latest_ml_run(capacity_ml_dir: Path, serial: str | int) -> Path:
    serial_dir = capacity_ml_dir / f"serial={serial}"
    if not serial_dir.exists():
        raise FileNotFoundError(f"No capacity ML directory found: {serial_dir}")

    runs = sorted(path for path in serial_dir.glob("ml_run=*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No ml_run folders found under: {serial_dir}")
    return runs[-1]


def default_output_path(data_dir: Path, serial: str | int, ml_run_dir: Path) -> Path:
    run_label = ml_run_dir.name.removeprefix("ml_run=")
    filename = (
        f"capacity_forecast_report_serial_{_safe_label(serial)}_"
        f"ml_run_{_safe_label(run_label)}.png"
    )
    return _non_overwriting_path(
        data_dir / "validation_plots" / f"serial={_safe_label(serial)}" / filename
    )


def load_outputs(ml_run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    training_path = ml_run_dir / "ml_training_table.parquet"
    forecast_path = ml_run_dir / "capacity_forecast.parquet"
    summary_path = ml_run_dir / "model_summary.json"

    if not training_path.exists():
        raise FileNotFoundError(f"Missing ML training table: {training_path}")
    if not forecast_path.exists():
        raise FileNotFoundError(f"Missing capacity forecast table: {forecast_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing model summary: {summary_path}")

    return (
        pd.read_parquet(training_path),
        pd.read_parquet(forecast_path),
        json.loads(summary_path.read_text(encoding="utf-8")),
    )


def _fmt(value: object, digits: int = 3, suffix: str = "") -> str:
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_signed(value: object, digits: int = 2, suffix: str = "") -> str:
    try:
        return f"{float(value):+.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "n/a"


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


def prepare_data(
    training: pd.DataFrame,
    forecast: pd.DataFrame,
    nominal_capacity_ah: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = training.copy()
    future = forecast.copy()

    history["end_anchor_time"] = pd.to_datetime(history["end_anchor_time"], errors="coerce", utc=True)
    future["forecast_timestamp"] = pd.to_datetime(
        future["forecast_timestamp"],
        errors="coerce",
        utc=True,
    )

    for column in ("measured_soh_pct", "capacity_ah"):
        history[column] = pd.to_numeric(history[column], errors="coerce")
    for column in (
        "predicted_future_soh_pct",
        "predicted_capacity_ah",
        "usage_multiplier",
        "forecast_ah_per_day",
        "cumulative_all_discharge_ah",
        "equivalent_full_cycles",
        "usage_error_daily_ah_slope",
        "usage_error_daily_ah_final",
        "projected_soh_error_pct_slope",
        "projected_soh_error_pct_final",
    ):
        if column not in future.columns:
            continue
        future[column] = pd.to_numeric(future[column], errors="coerce")
    if "equivalent_full_cycles" not in future.columns:
        future["equivalent_full_cycles"] = (
            future["cumulative_all_discharge_ah"] / nominal_capacity_ah
        )
    if "usage_rate_model" in future.columns:
        future["plot_label"] = future["usage_rate_model"].fillna(future["scenario"])
    else:
        future["plot_label"] = future["scenario"]
    if "usage_rate_model" in future.columns and "usage_multiplier" in future.columns:
        unique_scenarios_per_mode = future.groupby("plot_label")["scenario"].nunique()
        if unique_scenarios_per_mode.max() > 1:
            future = future[pd.to_numeric(future["usage_multiplier"], errors="coerce").eq(1.0)]

    history = history.dropna(subset=["end_anchor_time", "measured_soh_pct"])
    future = future.dropna(subset=["forecast_timestamp", "plot_label", "predicted_future_soh_pct"])
    return history.sort_values("end_anchor_time"), future.sort_values("forecast_timestamp")


def crossing_details(group: pd.DataFrame, threshold_soh_pct: float) -> dict[str, object]:
    group = group.sort_values("forecast_timestamp").reset_index(drop=True)
    below = group[group["predicted_future_soh_pct"] <= threshold_soh_pct]
    if below.empty:
        return {
            "crossing_timestamp": None,
            "crossing_capacity_ah": None,
            "crossing_equivalent_full_cycles": None,
            "crossing_soh_error_slope": None,
            "crossing_soh_error_final": None,
            "crossing_date_uncertainty_days_slope": None,
        }

    index = int(below.index[0])
    if index == 0:
        row = group.loc[index]
        soh_loss_per_day = None
        if len(group) > 1:
            next_row = group.loc[1]
            delta_days = (
                pd.Timestamp(next_row["forecast_timestamp"])
                - pd.Timestamp(row["forecast_timestamp"])
            ).total_seconds() / 86400.0
            if delta_days > 0:
                soh_loss_per_day = abs(
                    float(next_row["predicted_future_soh_pct"])
                    - float(row["predicted_future_soh_pct"])
                ) / delta_days
        crossing_soh_error_slope = (
            float(row["projected_soh_error_pct_slope"])
            if "projected_soh_error_pct_slope" in row
            and pd.notna(row["projected_soh_error_pct_slope"])
            else None
        )
        return {
            "crossing_timestamp": pd.Timestamp(row["forecast_timestamp"]),
            "crossing_capacity_ah": float(row["predicted_capacity_ah"]),
            "crossing_equivalent_full_cycles": float(row["equivalent_full_cycles"]),
            "crossing_soh_error_slope": crossing_soh_error_slope,
            "crossing_soh_error_final": (
                float(row["projected_soh_error_pct_final"])
                if "projected_soh_error_pct_final" in row
                and pd.notna(row["projected_soh_error_pct_final"])
                else None
            ),
            "crossing_date_uncertainty_days_slope": (
                crossing_soh_error_slope / soh_loss_per_day
                if crossing_soh_error_slope is not None
                and soh_loss_per_day is not None
                and soh_loss_per_day > 0
                else None
            ),
        }

    previous = group.loc[index - 1]
    current = group.loc[index]
    y0 = float(previous["predicted_future_soh_pct"])
    y1 = float(current["predicted_future_soh_pct"])
    fraction = 0.0 if y1 == y0 else (threshold_soh_pct - y0) / (y1 - y0)
    fraction = max(0.0, min(1.0, fraction))
    crossing_time = pd.Timestamp(previous["forecast_timestamp"]) + (
        pd.Timestamp(current["forecast_timestamp"]) - pd.Timestamp(previous["forecast_timestamp"])
    ) * fraction
    crossing_capacity = float(previous["predicted_capacity_ah"]) + (
        float(current["predicted_capacity_ah"]) - float(previous["predicted_capacity_ah"])
    ) * fraction
    crossing_efc = float(previous["equivalent_full_cycles"]) + (
        float(current["equivalent_full_cycles"]) - float(previous["equivalent_full_cycles"])
    ) * fraction
    crossing_soh_error_slope = None
    crossing_soh_error_final = None
    if "projected_soh_error_pct_slope" in group.columns:
        previous_value = previous["projected_soh_error_pct_slope"]
        current_value = current["projected_soh_error_pct_slope"]
        if pd.notna(previous_value) and pd.notna(current_value):
            crossing_soh_error_slope = float(previous_value) + (
                float(current_value) - float(previous_value)
            ) * fraction
    if "projected_soh_error_pct_final" in group.columns:
        previous_value = previous["projected_soh_error_pct_final"]
        current_value = current["projected_soh_error_pct_final"]
        if pd.notna(previous_value) and pd.notna(current_value):
            crossing_soh_error_final = float(previous_value) + (
                float(current_value) - float(previous_value)
            ) * fraction
    delta_days = (
        pd.Timestamp(current["forecast_timestamp"]) - pd.Timestamp(previous["forecast_timestamp"])
    ).total_seconds() / 86400.0
    soh_loss_per_day = abs(y1 - y0) / delta_days if delta_days > 0 else None
    return {
        "crossing_timestamp": crossing_time,
        "crossing_capacity_ah": crossing_capacity,
        "crossing_equivalent_full_cycles": crossing_efc,
        "crossing_soh_error_slope": crossing_soh_error_slope,
        "crossing_soh_error_final": crossing_soh_error_final,
        "crossing_date_uncertainty_days_slope": (
            crossing_soh_error_slope / soh_loss_per_day
            if crossing_soh_error_slope is not None
            and soh_loss_per_day is not None
            and soh_loss_per_day > 0
            else None
        ),
    }


def scenario_summary(
    forecast: pd.DataFrame,
    threshold_soh_pct: float,
    forecast_validation: dict | None = None,
) -> pd.DataFrame:
    validation_by_mode = (
        forecast_validation.get("by_usage_rate_model", {})
        if forecast_validation
        else {}
    )
    rows: list[dict[str, object]] = []
    for scenario, group in forecast.groupby("plot_label", sort=False):
        group = group.sort_values("forecast_timestamp")
        crossing = crossing_details(group, threshold_soh_pct)
        crossing_time = crossing["crossing_timestamp"]
        crossing_capacity = crossing["crossing_capacity_ah"]
        crossing_efc = crossing["crossing_equivalent_full_cycles"]
        crossing_soh_error = crossing["crossing_soh_error_slope"]
        crossing_date_uncertainty = crossing["crossing_date_uncertainty_days_slope"]
        validation = validation_by_mode.get(str(scenario), {})
        ah_per_day = (
            float(group["forecast_ah_per_day"].iloc[0])
            if "forecast_ah_per_day" in group.columns
            else float("nan")
        )
        rows.append(
            {
                "scenario": scenario,
                "usage": f"{float(group['usage_multiplier'].iloc[0]):.1f}x",
                "ah_per_day": f"{ah_per_day:.2f}" if not pd.isna(ah_per_day) else "n/a",
                "crossing_month": (
                    crossing_time.strftime("%Y-%m") if crossing_time is not None else "not reached"
                ),
                "crossing_capacity": (
                    f"{float(crossing_capacity):.2f} Ah"
                    if crossing_capacity is not None
                    else "n/a"
                ),
                "crossing_efc": (
                    f"{float(crossing_efc):.0f}"
                    if crossing_efc is not None
                    else "n/a"
                ),
                "crossing_soh_error": (
                    f"+/-{float(crossing_soh_error):.2f}%"
                    if crossing_soh_error is not None
                    else "n/a"
                ),
                "crossing_date_uncertainty": _fmt_uncertainty_days(
                    crossing_date_uncertainty,
                ),
                "final_soh": f"{float(group['predicted_future_soh_pct'].iloc[-1]):.2f}%",
                "final_capacity": f"{float(group['predicted_capacity_ah'].iloc[-1]):.2f} Ah",
                "final_efc": f"{float(group['equivalent_full_cycles'].iloc[-1]):.0f}",
                "daily_ah_error_slope": (
                    _fmt_signed(validation.get("daily_ah_error_slope"), 2)
                    if validation and validation.get("validation_available")
                    else "n/a"
                ),
                "forecast_mae_soh": (
                    f"{float(validation['mae_soh_pct']):.3f}%"
                    if validation
                    and validation.get("validation_available")
                    and validation.get("mae_soh_pct") is not None
                    else "n/a"
                ),
                "forecast_max_soh_error": (
                    f"{float(validation['max_abs_soh_error_pct']):.3f}%"
                    if validation
                    and validation.get("validation_available")
                    and validation.get("max_abs_soh_error_pct") is not None
                    else "n/a"
                ),
                "crossing_timestamp": crossing_time,
            }
        )
    return pd.DataFrame(rows)


def add_text_panel(ax: plt.Axes, title: str, lines: list[str]) -> None:
    add_text_panel_with_font(ax, title, lines, fontsize=10)


def add_text_panel_with_font(
    ax: plt.Axes,
    title: str,
    lines: list[str],
    *,
    fontsize: int,
) -> None:
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=8)
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=fontsize,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )


def add_usage_summary_table(ax: plt.Axes, scenarios: pd.DataFrame) -> None:
    """Render the usage-rate summary as a bordered table."""
    ax.axis("off")
    ax.set_title("Usage-Rate Model Summary", loc="left", fontsize=12, fontweight="bold", pad=8)

    columns = [
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
    rows = [
        [
            str(row["scenario"]),
            str(row["ah_per_day"]),
            str(row["forecast_mae_soh"]),
            str(row["forecast_max_soh_error"]),
            str(row["daily_ah_error_slope"]),
            str(row["crossing_month"]),
            str(row["crossing_soh_error"]),
            str(row["crossing_date_uncertainty"]),
            str(row["crossing_efc"]),
            str(row["final_soh"]),
        ]
        for _, row in scenarios.iterrows()
    ]

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.20, 0.07, 0.08, 0.08, 0.08, 0.10, 0.085, 0.085, 0.075, 0.085],
        bbox=[0.0, 0.08, 1.0, 0.86],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1.0, 1.35)

    for (row_index, _column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#94a3b8")
        cell.set_linewidth(0.65)
        if row_index == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold", color="#0f172a")
        else:
            cell.set_facecolor("#f8fafc" if row_index % 2 else "#ffffff")
            cell.set_text_props(color="#111827")


def plot_report_image(
    training: pd.DataFrame,
    forecast: pd.DataFrame,
    summary: dict,
    *,
    output_path: Path,
    title: str,
    threshold_soh_pct: float,
) -> None:
    nominal_capacity_ah = float(summary["nominal_capacity_ah"])
    history, future = prepare_data(training, forecast, nominal_capacity_ah)
    scenarios = scenario_summary(
        future,
        threshold_soh_pct,
        summary.get("cutoff_forecast_validation"),
    )

    fig = plt.figure(figsize=(16, 9), dpi=150)
    grid = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0], hspace=0.35, wspace=0.18)
    ax = fig.add_subplot(grid[0, :])
    ax_model = fig.add_subplot(grid[1, 0])
    ax_scenarios = fig.add_subplot(grid[1, 1])

    valid_history = history
    if "valid_training_row" in valid_history.columns:
        valid_history = valid_history[valid_history["valid_training_row"].fillna(False)]

    ax.scatter(
        valid_history["end_anchor_time"],
        valid_history["measured_soh_pct"],
        s=16,
        color="#334155",
        alpha=0.45,
        label="measured capacity-based SOH",
    )

    for scenario, group in future.groupby("plot_label", sort=False):
        color = SCENARIO_COLORS.get(str(scenario), "#2563eb")
        ax.plot(
            group["forecast_timestamp"],
            group["predicted_future_soh_pct"],
            color=color,
            linewidth=2.5,
            label=str(scenario),
        )

    ax.axhline(
        threshold_soh_pct,
        color="#64748b",
        linestyle="--",
        linewidth=1.5,
        label=f"{threshold_soh_pct:.0f}% SOH threshold",
    )

    for _, row in scenarios.iterrows():
        crossing = row["crossing_timestamp"]
        if crossing is None or pd.isna(crossing):
            continue
        label_offsets = {
            "high_usage": (-6, 24),
            "normal_usage": (8, 34),
            "low_usage": (0, 24),
            "recent_median": (-18, 30),
            "historical_median": (0, 46),
            "historical_mean": (18, 62),
        }
        label_offset = label_offsets.get(str(row["scenario"]), (0, 26))
        color = SCENARIO_COLORS.get(str(row["scenario"]), "#2563eb")
        ax.scatter(crossing, threshold_soh_pct, color=color, s=70, marker="D", zorder=5)
        ax.annotate(
            str(row["crossing_month"]),
            xy=(crossing, threshold_soh_pct),
            xytext=label_offset,
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=color,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )

    y_min = min(float(future["predicted_future_soh_pct"].min()), threshold_soh_pct) - 3.0
    y_max = max(float(history["measured_soh_pct"].max()), float(future["predicted_future_soh_pct"].max())) + 2.0
    ax.set_ylim(max(0.0, y_min), min(105.0, y_max))
    ax.set_title("Capacity-Based SOH Forecast", loc="left", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("SOH %")
    ax.grid(True, color="#e2e8f0")
    ax.legend(loc="upper right", ncols=2, fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    validation = summary.get("cutoff_validation", {})
    degradation = summary.get("degradation_forecast_model", {})
    usage_rates = summary.get("usage_rates", {})
    usage_rate_modes = summary.get("usage_rate_modes")
    model_lines = [
        f"training cutoff : {str(summary.get('training_cutoff', 'n/a'))[:10]}",
        f"validation      : {str(validation.get('start_time', 'n/a'))[:10]} to {str(validation.get('end_time', 'n/a'))[:10]}",
        f"validation rows : {validation.get('rows', 'n/a')}",
        f"MAE             : {_fmt(validation.get('mae_ah'), 3, ' Ah')} / {_fmt(validation.get('mae_soh_pct'), 3, '% SOH')}",
        f"max SOH error   : {_fmt(validation.get('max_abs_soh_error_pct'), 3, '%')}",
        f"model           : {degradation.get('model_name', 'n/a')}",
        f"usage modes     : {','.join(usage_rate_modes) if usage_rate_modes else usage_rates.get('usage_rate_mode', 'historical_mean')}",
        f"baseline        : {_fmt(degradation.get('baseline_capacity_ah'), 3, ' Ah')}",
        f"reference Ah    : {_fmt(degradation.get('reference_cumulative_ah'), 1, ' Ah')}",
        f"loss slope      : {_fmt(degradation.get('loss_slope_per_1000ah'), 4, ' Ah / 1000 Ah')}",
    ]
    add_text_panel(ax_model, "Validation and Model Setup", model_lines)

    add_usage_summary_table(ax_scenarios, scenarios)

    fig.suptitle(title, fontsize=18, fontweight="bold", x=0.02, y=0.98, ha="left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Wrote Confluence report image to %s", output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a static Confluence-ready capacity forecast PNG with matplotlib.",
    )
    parser.add_argument("--serial", default=None, help="Serial to plot when --ml-run-dir is omitted.")
    parser.add_argument("--ml-run-dir", type=Path, default=None, help="Specific ml_run folder to plot.")
    parser.add_argument("--capacity-ml-dir", type=Path, default=None, help="Override capacity ML base directory.")
    parser.add_argument("--output", type=Path, default=None, help="PNG output path.")
    parser.add_argument("--env-file", type=Path, default=None, help="Path to a .env file.")
    parser.add_argument("--title", default=None, help="Figure title.")
    parser.add_argument("--threshold-soh-pct", type=float, default=80.0, help="SOH threshold to mark.")
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
            capacity_ml_dir=args.capacity_ml_dir,
        )

        if args.ml_run_dir is not None:
            ml_run_dir = args.ml_run_dir
        elif args.serial is not None:
            ml_run_dir = find_latest_ml_run(storage_config.capacity_ml_dir, args.serial)
        else:
            raise ValueError("Provide either --serial or --ml-run-dir.")

        training, forecast, summary = load_outputs(ml_run_dir)
        serial = args.serial or str(training["serial"].iloc[0])
        output_path = (
            _non_overwriting_path(args.output)
            if args.output
            else default_output_path(storage_config.data_dir, serial, ml_run_dir)
        )
        run_label = ml_run_dir.name.removeprefix("ml_run=")
        plot_report_image(
            training,
            forecast,
            summary,
            output_path=output_path,
            title=args.title or f"Serial {serial} SOH Forecast Report - ml_run {run_label}",
            threshold_soh_pct=args.threshold_soh_pct,
        )
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Confluence report image export failed.")
        else:
            logger.error("Confluence report image export failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
