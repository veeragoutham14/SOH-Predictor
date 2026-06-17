from __future__ import annotations

import argparse
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
    """Return a filesystem-friendly label for generated plot filenames."""
    label = SAFE_LABEL_RE.sub("_", str(value).strip())
    return label.strip("_") or "unknown"


def _non_overwriting_path(path: Path) -> Path:
    """Return path, or a numbered variant, without replacing an existing file."""
    if not path.exists():
        return path

    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise FileExistsError(f"Could not find an unused output filename near: {path}")


def default_output_path(
    *,
    data_dir: Path,
    serial: str | int,
    ml_run_dir: Path,
) -> Path:
    """Build the default non-overwriting HTML path for one ML run."""
    run_label = ml_run_dir.name.removeprefix("ml_run=")
    filename = (
        f"capacity_usage_forecast_serial_{_safe_label(serial)}_"
        f"ml_run_{_safe_label(run_label)}.html"
    )
    return _non_overwriting_path(data_dir / "validation_plots" / filename)


def find_latest_ml_run(capacity_ml_dir: Path, serial: str | int) -> Path:
    """Return the latest ml_run directory for a serial."""
    serial_dir = capacity_ml_dir / f"serial={serial}"
    if not serial_dir.exists():
        raise FileNotFoundError(f"No capacity ML directory found: {serial_dir}")

    runs = sorted(path for path in serial_dir.glob("ml_run=*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No ml_run folders found under: {serial_dir}")
    return runs[-1]


def load_ml_outputs(ml_run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the ML training, future forecast, and optional backtest tables."""
    training_path = ml_run_dir / "ml_training_table.parquet"
    forecast_path = ml_run_dir / "capacity_forecast.parquet"
    backtest_path = ml_run_dir / "model_backtest.parquet"

    if not training_path.exists():
        raise FileNotFoundError(f"Missing ML training table: {training_path}")
    if not forecast_path.exists():
        raise FileNotFoundError(f"Missing capacity forecast table: {forecast_path}")

    training = pd.read_parquet(training_path)
    forecast = pd.read_parquet(forecast_path)
    backtest = pd.read_parquet(backtest_path) if backtest_path.exists() else pd.DataFrame()
    return training, forecast, backtest


def prepare_plot_tables(
    training: pd.DataFrame,
    forecast: pd.DataFrame,
    backtest: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize timestamp and numeric columns used by the dashboard."""
    historical = training.copy()
    future = forecast.copy()

    historical["end_anchor_time"] = pd.to_datetime(
        historical["end_anchor_time"],
        errors="coerce",
        utc=True,
    )
    future["forecast_timestamp"] = pd.to_datetime(
        future["forecast_timestamp"],
        errors="coerce",
        utc=True,
    )
    if "usage_rate_model" in future.columns:
        future["plot_label"] = future["usage_rate_model"].fillna(future["scenario"])
    else:
        future["plot_label"] = future["scenario"]
    if "usage_rate_model" in future.columns and "usage_multiplier" in future.columns:
        unique_scenarios_per_mode = future.groupby("plot_label")["scenario"].nunique()
        if unique_scenarios_per_mode.max() > 1:
            future = future[pd.to_numeric(future["usage_multiplier"], errors="coerce").eq(1.0)]

    numeric_columns = [
        "event_discharge_id",
        "capacity_ah",
        "measured_soh_pct",
        "cumulative_all_discharge_ah",
        "cumulative_all_discharge_wh",
        "predicted_capacity_ah",
        "predicted_capacity_ah_lower_95",
        "predicted_capacity_ah_upper_95",
        "predicted_future_soh_pct",
        "equivalent_full_cycles",
        "projected_soh_error_pct_slope",
        "projected_soh_error_pct_final",
        "usage_error_daily_ah_slope",
    ]
    for column in numeric_columns:
        if column in historical.columns:
            historical[column] = pd.to_numeric(historical[column], errors="coerce")
        if column in future.columns:
            future[column] = pd.to_numeric(future[column], errors="coerce")

    historical = historical.dropna(
        subset=["end_anchor_time", "cumulative_all_discharge_ah"],
    )
    future = future.dropna(
        subset=["forecast_timestamp", "plot_label", "cumulative_all_discharge_ah"],
    )
    historical = historical.sort_values("end_anchor_time", kind="mergesort")
    future = future.sort_values(["plot_label", "forecast_timestamp"], kind="mergesort")
    return historical, future


def prepare_backtest_table(backtest: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize the optional backtest table for plotting."""
    if backtest is None or backtest.empty:
        return pd.DataFrame()

    work = backtest.copy()
    work["end_anchor_time"] = pd.to_datetime(work["end_anchor_time"], errors="coerce", utc=True)
    numeric_columns = [
        "event_discharge_id",
        "actual_capacity_ah",
        "predicted_capacity_ah",
        "capacity_error_ah",
        "abs_capacity_error_ah",
        "actual_soh_pct",
        "predicted_soh_pct",
        "soh_error_pct",
        "abs_soh_error_pct",
    ]
    for column in numeric_columns:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    work = work.dropna(subset=["end_anchor_time", "actual_capacity_ah", "predicted_capacity_ah"])
    return work.sort_values("end_anchor_time", kind="mergesort").reset_index(drop=True)


def _downsample(df: pd.DataFrame, max_points: int | None) -> pd.DataFrame:
    if max_points is None or max_points <= 0 or len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def _infer_nominal_capacity_ah(historical: pd.DataFrame, future: pd.DataFrame) -> float | None:
    ratios: list[pd.Series] = []
    if {"capacity_ah", "measured_soh_pct"}.issubset(historical.columns):
        measured = historical["measured_soh_pct"].replace(0, pd.NA)
        ratios.append(historical["capacity_ah"] / measured * 100.0)
    if {"predicted_capacity_ah", "predicted_future_soh_pct"}.issubset(future.columns):
        predicted = future["predicted_future_soh_pct"].replace(0, pd.NA)
        ratios.append(future["predicted_capacity_ah"] / predicted * 100.0)

    if not ratios:
        return None

    values = pd.concat(ratios, ignore_index=True)
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def _capacity_axis_range(historical: pd.DataFrame, future: pd.DataFrame) -> tuple[float, float] | None:
    columns = [
        (historical, "capacity_ah"),
        (future, "predicted_capacity_ah"),
        (future, "predicted_capacity_ah_lower_95"),
        (future, "predicted_capacity_ah_upper_95"),
    ]
    values = []
    for frame, column in columns:
        if column in frame.columns:
            values.append(pd.to_numeric(frame[column], errors="coerce"))
    if not values:
        return None

    all_values = pd.concat(values, ignore_index=True).dropna()
    if all_values.empty:
        return None

    lower = float(all_values.min())
    upper = float(all_values.max())
    padding = max((upper - lower) * 0.08, 0.5)
    return max(0.0, lower - padding), upper + padding


def plot_capacity_usage_forecast(
    training: pd.DataFrame,
    forecast: pd.DataFrame,
    backtest: pd.DataFrame | None = None,
    *,
    output_path: Path,
    title: str,
    max_history_points: int | None = 50_000,
) -> None:
    """Write an interactive HTML dashboard for usage and capacity forecast."""
    historical, future = prepare_plot_tables(training, forecast)
    backtest_plot = prepare_backtest_table(backtest)
    if historical.empty:
        raise ValueError("No historical ML rows available to plot.")
    if future.empty:
        raise ValueError("No forecast rows available to plot.")

    historical_plot = _downsample(historical, max_history_points)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_backtest = not backtest_plot.empty
    row_count = 3 if has_backtest else 2
    subplot_titles = [
        "Cumulative discharge usage",
        "Measured and predicted capacity / synced SOH",
    ]
    specs = [[{"secondary_y": False}], [{"secondary_y": True}]]
    if has_backtest:
        subplot_titles.append("Backtest: hidden historical capacity prediction error")
        specs.append([{"secondary_y": True}])

    fig = make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=tuple(subplot_titles),
        specs=specs,
    )

    fig.add_trace(
        go.Scattergl(
            x=historical_plot["end_anchor_time"],
            y=historical_plot["cumulative_all_discharge_ah"],
            mode="lines+markers",
            name="historical cumulative Ah",
            line={"width": 2, "color": "#2563eb"},
            marker={"size": 4},
            customdata=historical_plot[
                ["event_discharge_id", "capacity_ah", "measured_soh_pct"]
            ],
            hovertemplate=(
                "time=%{x}<br>"
                "cumulative Ah=%{y:.2f}<br>"
                "event=%{customdata[0]}<br>"
                "capacity Ah=%{customdata[1]:.3f}<br>"
                "measured SOH=%{customdata[2]:.2f}%<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    scenario_colors = {
        "low_usage": "#16a34a",
        "normal_usage": "#f59e0b",
        "high_usage": "#dc2626",
        "historical_mean": "#2563eb",
        "historical_median": "#f59e0b",
        "recent_median": "#dc2626",
    }
    for scenario, group in future.groupby("plot_label", sort=False):
        color = scenario_colors.get(str(scenario), None)
        fig.add_trace(
            go.Scatter(
                x=group["forecast_timestamp"],
                y=group["cumulative_all_discharge_ah"],
                mode="lines",
                name=f"{scenario} cumulative Ah",
                line={"width": 2, "dash": "dash", "color": color},
                hovertemplate=(
                    "time=%{x}<br>"
                    "usage model="
                    + str(scenario)
                    + "<br>future cumulative Ah=%{y:.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scattergl(
            x=historical_plot["end_anchor_time"],
            y=historical_plot["capacity_ah"],
            mode="markers",
            name="measured capacity Ah",
            marker={"size": 5, "color": "#334155", "opacity": 0.65},
            customdata=historical_plot[["measured_soh_pct"]],
            hovertemplate=(
                "time=%{x}<br>"
                "measured capacity=%{y:.3f} Ah<br>"
                "measured SOH=%{customdata[0]:.2f}%<extra></extra>"
            ),
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    for scenario, group in future.groupby("plot_label", sort=False):
        group = group.copy()
        color = scenario_colors.get(str(scenario), None)
        hover_columns = [
            "predicted_future_soh_pct",
            "projected_soh_error_pct_slope",
            "projected_soh_error_pct_final",
            "equivalent_full_cycles",
            "usage_error_daily_ah_slope",
        ]
        for column in hover_columns:
            if column not in group.columns:
                group[column] = float("nan")
        fig.add_trace(
            go.Scatter(
                x=group["forecast_timestamp"],
                y=group["predicted_capacity_ah"],
                mode="lines",
                name=f"{scenario} predicted capacity Ah",
                line={"width": 2, "color": color},
                customdata=group[hover_columns],
                hovertemplate=(
                    "time=%{x}<br>"
                    "usage model="
                    + str(scenario)
                    + "<br>predicted capacity=%{y:.3f} Ah<br>"
                    "future SOH=%{customdata[0]:.2f}%<br>"
                    "SOH uncertainty, slope=+/- %{customdata[1]:.2f}%<br>"
                    "SOH uncertainty, final=+/- %{customdata[2]:.2f}%<br>"
                    "EFC=%{customdata[3]:.0f}<br>"
                    "daily Ah error slope=%{customdata[4]:.2f} Ah/day<extra></extra>"
                ),
            ),
            row=2,
            col=1,
            secondary_y=False,
        )

    if {
        "predicted_capacity_ah",
        "predicted_capacity_ah_lower_95",
        "predicted_capacity_ah_upper_95",
        "predicted_future_soh_pct",
    }.issubset(future.columns):
        band_label = "normal_usage" if future["plot_label"].eq("normal_usage").any() else "historical_median"
        normal = future[future["plot_label"].eq(band_label)].copy()
        if not normal.empty:
            fig.add_trace(
                go.Scatter(
                    x=normal["forecast_timestamp"],
                    y=normal["predicted_capacity_ah_upper_95"],
                    mode="lines",
                    name="normal upper 95%",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=normal["forecast_timestamp"],
                    y=normal["predicted_capacity_ah_lower_95"],
                    mode="lines",
                    name="normal 95% band",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor="rgba(245, 158, 11, 0.16)",
                    hovertemplate="normal 95% band<extra></extra>",
                ),
                row=2,
                col=1,
                secondary_y=False,
            )

    if has_backtest:
        holdout = backtest_plot[backtest_plot["row_type"].eq("holdout")]
        train = backtest_plot[backtest_plot["row_type"].eq("train")]

        fig.add_trace(
            go.Scattergl(
                x=train["end_anchor_time"],
                y=train["actual_capacity_ah"],
                mode="markers",
                name="backtest train actual",
                marker={"size": 4, "color": "#94a3b8", "opacity": 0.45},
                hovertemplate="time=%{x}<br>train actual=%{y:.3f} Ah<extra></extra>",
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scattergl(
                x=holdout["end_anchor_time"],
                y=holdout["actual_capacity_ah"],
                mode="markers",
                name="holdout actual capacity",
                marker={"size": 7, "color": "#2563eb"},
                hovertemplate="time=%{x}<br>holdout actual=%{y:.3f} Ah<extra></extra>",
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=backtest_plot["end_anchor_time"],
                y=backtest_plot["predicted_capacity_ah"],
                mode="lines",
                name="backtest predicted capacity",
                line={"width": 2, "color": "#f97316"},
                hovertemplate="time=%{x}<br>predicted=%{y:.3f} Ah<extra></extra>",
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(
                x=holdout["end_anchor_time"],
                y=holdout["capacity_error_ah"],
                name="holdout error Ah",
                marker={"color": "#dc2626", "opacity": 0.55},
                hovertemplate=(
                    "time=%{x}<br>"
                    "actual - predicted=%{y:.3f} Ah<extra></extra>"
                ),
            ),
            row=3,
            col=1,
            secondary_y=True,
        )

    first_time = historical["end_anchor_time"].min()
    last_history_time = historical["end_anchor_time"].max()
    last_forecast_time = future["forecast_timestamp"].max()
    backtest_summary = ""
    if has_backtest:
        holdout = backtest_plot[backtest_plot["row_type"].eq("holdout")]
        if not holdout.empty:
            backtest_summary = (
                f" | holdout MAE={holdout['abs_capacity_error_ah'].mean():.3f} Ah"
                f" / {holdout['abs_soh_error_pct'].mean():.2f}% SOH"
            )
    summary = (
        f"history rows={len(historical):,} | "
        f"history={first_time.date()} to {last_history_time.date()} | "
        f"forecast to {last_forecast_time.date()}"
        f"{backtest_summary}"
    )

    fig.update_layout(
        title=f"{title}<br><sup>{summary}</sup>",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.24, "x": 0},
        margin={"l": 70, "r": 70, "t": 95, "b": 110},
    )
    fig.update_xaxes(rangeslider_visible=True, row=row_count, col=1)
    fig.update_yaxes(title_text="Cumulative discharge Ah", row=1, col=1)
    nominal_capacity_ah = _infer_nominal_capacity_ah(historical, future)
    capacity_range = _capacity_axis_range(historical, future)
    if capacity_range is not None:
        fig.update_yaxes(
            title_text="Capacity Ah",
            range=list(capacity_range),
            row=2,
            col=1,
            secondary_y=False,
        )
        if nominal_capacity_ah is not None and nominal_capacity_ah > 0:
            soh_range = [
                capacity_range[0] / nominal_capacity_ah * 100.0,
                capacity_range[1] / nominal_capacity_ah * 100.0,
            ]
            fig.add_trace(
                go.Scatter(
                    x=[first_time, last_forecast_time],
                    y=soh_range,
                    mode="lines",
                    name="SOH axis scale",
                    line={"color": "rgba(0,0,0,0)", "width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=2,
                col=1,
                secondary_y=True,
            )
            fig.update_yaxes(
                title_text="SOH %",
                range=soh_range,
                row=2,
                col=1,
                secondary_y=True,
            )
    else:
        fig.update_yaxes(title_text="Capacity Ah", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="SOH %", row=2, col=1, secondary_y=True)
    if has_backtest:
        fig.update_yaxes(title_text="Capacity Ah", row=3, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Error Ah", row=3, col=1, secondary_y=True)

    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    logger.info("Wrote capacity usage forecast graph to %s", output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot capacity ML historical usage and future usage forecasts.",
    )
    parser.add_argument("--serial", default=None, help="Serial to plot when --ml-run-dir is omitted.")
    parser.add_argument("--ml-run-dir", type=Path, default=None, help="Specific ml_run folder to plot.")
    parser.add_argument("--capacity-ml-dir", type=Path, default=None, help="Override capacity ML base directory.")
    parser.add_argument("--output", type=Path, default=None, help="HTML output path.")
    parser.add_argument("--env-file", type=Path, default=None, help="Path to a .env file.")
    parser.add_argument("--title", default=None, help="Plot title.")
    parser.add_argument(
        "--max-history-points",
        type=int,
        default=50_000,
        help="Maximum historical points to plot. Use 0 to plot all rows.",
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
            capacity_ml_dir=args.capacity_ml_dir,
        )

        if args.ml_run_dir is not None:
            ml_run_dir = args.ml_run_dir
        elif args.serial is not None:
            ml_run_dir = find_latest_ml_run(storage_config.capacity_ml_dir, args.serial)
        else:
            raise ValueError("Provide either --serial or --ml-run-dir.")

        training, forecast, backtest = load_ml_outputs(ml_run_dir)
        serial = args.serial or str(training["serial"].iloc[0])
        output_path = (
            _non_overwriting_path(args.output)
            if args.output
            else default_output_path(
                data_dir=storage_config.data_dir,
                serial=serial,
                ml_run_dir=ml_run_dir,
            )
        )
        plot_capacity_usage_forecast(
            training,
            forecast,
            backtest,
            output_path=output_path,
            title=args.title or f"Serial {serial} capacity usage forecast",
            max_history_points=None if args.max_history_points == 0 else args.max_history_points,
        )
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Capacity forecast plotting failed.")
        else:
            logger.error("Capacity forecast plotting failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
