from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.capacity_ml import DEFAULT_SCENARIOS, add_model_features, parse_scenarios
from src.config import CapacityModelConfig, StorageConfig
from src.logging_utils import configure_logging
from src.plot_capacity_forecast import find_latest_ml_run


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
    model_name: str,
) -> Path:
    """Build the default non-overwriting HTML path for the selected model."""
    run_label = ml_run_dir.name.removeprefix("ml_run=")
    filename = (
        f"best_capacity_model_forecast_serial_{_safe_label(serial)}_"
        f"model_{_safe_label(model_name)}_ml_run_{_safe_label(run_label)}.html"
    )
    return _non_overwriting_path(
        data_dir / "validation_plots" / f"serial={_safe_label(serial)}" / filename
    )


def load_ml_run_inputs(ml_run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the training table and JSON summary for one ML run."""
    training_path = ml_run_dir / "ml_training_table.parquet"
    summary_path = ml_run_dir / "model_summary.json"

    if not training_path.exists():
        raise FileNotFoundError(f"Missing ML training table: {training_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing model summary JSON: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    return pd.read_parquet(training_path), summary


def selected_capacity_ah_model(summary: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Return the selected best capacity-Ah model name and feature columns."""
    model = summary.get("best_capacity_ah_model")
    if not model:
        raise ValueError("model_summary.json does not contain best_capacity_ah_model.")

    model_name = str(model["model_name"])
    feature_columns = tuple(model["feature_columns"])
    if not feature_columns:
        raise ValueError(f"Selected model has no feature columns: {model_name}")
    return model_name, feature_columns


def prepare_training_table(training: pd.DataFrame) -> pd.DataFrame:
    """Return valid rows with model features ready for refitting."""
    required = ["end_anchor_time", "capacity_ah", "event_discharge_id"]
    missing = [column for column in required if column not in training.columns]
    if missing:
        raise ValueError(f"Training table is missing required columns: {missing}")

    work = training.copy()
    work["end_anchor_time"] = pd.to_datetime(
        work["end_anchor_time"],
        errors="coerce",
        utc=True,
    )
    numeric_columns = [
        "event_discharge_id",
        "capacity_ah",
        "capacity_wh",
        "measured_soh_pct",
        "calendar_age_days",
        "cumulative_all_discharge_ah",
        "cumulative_all_discharge_wh",
    ]
    for column in numeric_columns:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    if "valid_training_row" in work.columns:
        work = work[work["valid_training_row"].fillna(False).astype(bool)]

    work = add_model_features(work)
    work = work.dropna(subset=["end_anchor_time", "capacity_ah"])
    work = work.sort_values("end_anchor_time", kind="mergesort").reset_index(drop=True)
    if len(work) < 2:
        raise ValueError("Need at least two valid capacity rows to fit the selected model.")
    return work


def fit_selected_model_on_all_rows(
    training: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    """Fit the chosen capacity model shape on all valid training rows."""
    required = ["capacity_ah", *feature_columns]
    fit_rows = training.dropna(subset=required).copy()
    if len(fit_rows) < 2:
        raise ValueError("Not enough rows have the selected model's required features.")

    x = fit_rows[list(feature_columns)].astype(float).to_numpy()
    design = np.column_stack([np.ones(len(fit_rows)), x])
    y = fit_rows["capacity_ah"].astype(float).to_numpy()
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)

    intercept = float(coefficients[0])
    slopes = tuple(float(value) for value in coefficients[1:])
    fitted = intercept + x @ np.asarray(slopes, dtype=float)
    residual = y - fitted

    return {
        "intercept": intercept,
        "coefficients": slopes,
        "fit_rows": fit_rows,
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "residual_std": float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0,
    }


def predict_from_fit(
    df: pd.DataFrame,
    *,
    intercept: float,
    coefficients: Sequence[float],
    feature_columns: Sequence[str],
) -> pd.Series:
    """Predict capacity Ah from fitted linear coefficients."""
    x = df[list(feature_columns)].astype(float).to_numpy()
    return pd.Series(
        intercept + x @ np.asarray(coefficients, dtype=float),
        index=df.index,
    )


def usage_rates_from_summary_or_training(
    summary: dict[str, Any],
    training: pd.DataFrame,
) -> tuple[float, float]:
    """Use saved usage rates when present, otherwise estimate from training rows."""
    usage_rates = summary.get("usage_rates") or {}
    ah_per_day = float(usage_rates.get("historical_ah_per_day", np.nan))
    wh_per_day = float(usage_rates.get("historical_wh_per_day", np.nan))

    if np.isfinite(ah_per_day) and ah_per_day > 0 and np.isfinite(wh_per_day) and wh_per_day > 0:
        return ah_per_day, wh_per_day

    first = training.iloc[0]
    last = training.iloc[-1]
    elapsed_days = max(
        (last["end_anchor_time"] - first["end_anchor_time"]).total_seconds() / 86400.0,
        1e-9,
    )
    ah_per_day = (
        float(last["cumulative_all_discharge_ah"])
        - float(first["cumulative_all_discharge_ah"])
    ) / elapsed_days
    wh_per_day = (
        float(last["cumulative_all_discharge_wh"])
        - float(first["cumulative_all_discharge_wh"])
    ) / elapsed_days
    return ah_per_day, wh_per_day


def build_future_feature_table(
    training: pd.DataFrame,
    *,
    summary: dict[str, Any],
    scenarios: dict[str, float],
    future_days: int,
    step_days: int,
) -> pd.DataFrame:
    """Build future rows with all feature columns used by candidate Ah models."""
    if future_days <= 0:
        raise ValueError("future_days must be positive.")
    if step_days <= 0:
        raise ValueError("step_days must be positive.")

    ah_per_day, wh_per_day = usage_rates_from_summary_or_training(summary, training)
    first_time = training["end_anchor_time"].min()
    last = training.iloc[-1]
    latest_capacity_ah = max(float(last["capacity_ah"]), 1e-9)

    rows: list[dict[str, Any]] = []
    for scenario, multiplier in scenarios.items():
        for day in range(step_days, future_days + 1, step_days):
            forecast_time = last["end_anchor_time"] + pd.Timedelta(days=day)
            future_cumulative_ah = (
                float(last["cumulative_all_discharge_ah"]) + ah_per_day * multiplier * day
            )
            future_cumulative_wh = (
                float(last["cumulative_all_discharge_wh"]) + wh_per_day * multiplier * day
            )
            event_increment = (
                future_cumulative_ah - float(last["cumulative_all_discharge_ah"])
            ) / latest_capacity_ah
            rows.append(
                {
                    "scenario": scenario,
                    "usage_multiplier": multiplier,
                    "forecast_days_after_last_measurement": day,
                    "forecast_timestamp": forecast_time,
                    "event_discharge_id": float(last["event_discharge_id"]) + event_increment,
                    "calendar_age_days": (forecast_time - first_time).total_seconds() / 86400.0,
                    "cumulative_all_discharge_ah": future_cumulative_ah,
                    "cumulative_all_discharge_wh": future_cumulative_wh,
                }
            )

    return add_model_features(pd.DataFrame(rows))


def build_forecast(
    future: pd.DataFrame,
    *,
    fit: dict[str, Any],
    feature_columns: Sequence[str],
    nominal_capacity_ah: float,
) -> pd.DataFrame:
    """Predict future capacity and SOH from the selected all-data model."""
    forecast = future.copy()
    forecast["predicted_capacity_ah"] = predict_from_fit(
        forecast,
        intercept=fit["intercept"],
        coefficients=fit["coefficients"],
        feature_columns=feature_columns,
    )
    forecast["predicted_future_soh_pct"] = (
        forecast["predicted_capacity_ah"] / nominal_capacity_ah * 100.0
    )
    residual = float(fit["residual_std"])
    forecast["predicted_capacity_ah_lower_95"] = forecast["predicted_capacity_ah"] - 1.96 * residual
    forecast["predicted_capacity_ah_upper_95"] = forecast["predicted_capacity_ah"] + 1.96 * residual
    return forecast


def plot_best_capacity_model_forecast(
    training: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    fit: dict[str, Any],
    model_name: str,
    feature_columns: Sequence[str],
    nominal_capacity_ah: float,
    output_path: Path,
    title: str,
) -> None:
    """Write an HTML plot for the all-data best capacity-Ah model forecast."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    historical = training.copy()
    historical["fitted_capacity_ah"] = predict_from_fit(
        historical,
        intercept=fit["intercept"],
        coefficients=fit["coefficients"],
        feature_columns=feature_columns,
    )
    historical["measured_soh_pct"] = historical["capacity_ah"] / nominal_capacity_ah * 100.0
    historical["fitted_soh_pct"] = historical["fitted_capacity_ah"] / nominal_capacity_ah * 100.0

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.11,
        subplot_titles=(
            f"Historical fit: {model_name}",
            f"10-year forecast from all-data {model_name}",
        ),
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
    )

    x_feature = feature_columns[0]
    fig.add_trace(
        go.Scattergl(
            x=historical[x_feature],
            y=historical["capacity_ah"],
            mode="markers",
            name="measured capacity Ah",
            marker={"size": 5, "color": "#334155", "opacity": 0.65},
            customdata=historical[["end_anchor_time", "event_discharge_id", "measured_soh_pct"]],
            hovertemplate=(
                f"{x_feature}=%{{x:.3f}}<br>"
                "time=%{customdata[0]}<br>"
                "event=%{customdata[1]:.0f}<br>"
                "capacity=%{y:.3f} Ah<br>"
                "measured SOH=%{customdata[2]:.2f}%<extra></extra>"
            ),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=historical[x_feature],
            y=historical["fitted_capacity_ah"],
            mode="lines",
            name=f"{model_name} fit on all data",
            line={"width": 2, "color": "#2563eb"},
            hovertemplate=f"{x_feature}=%{{x:.3f}}<br>fit=%{{y:.3f}} Ah<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=historical[x_feature],
            y=historical["fitted_soh_pct"],
            mode="lines",
            name="fitted SOH axis",
            line={"width": 0, "color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    scenario_colors = {
        "low_usage": "#16a34a",
        "normal_usage": "#f59e0b",
        "high_usage": "#dc2626",
    }
    for scenario, group in forecast.groupby("scenario", sort=False):
        color = scenario_colors.get(str(scenario))
        fig.add_trace(
            go.Scatter(
                x=group["forecast_timestamp"],
                y=group["predicted_capacity_ah"],
                mode="lines",
                name=f"{scenario} predicted capacity Ah",
                line={"width": 2, "color": color},
                customdata=group[
                    [
                        "event_discharge_id",
                        "cumulative_all_discharge_ah",
                        "predicted_future_soh_pct",
                        "usage_multiplier",
                    ]
                ],
                hovertemplate=(
                    "time=%{x}<br>"
                    "future event=%{customdata[0]:.1f}<br>"
                    "future cumulative Ah=%{customdata[1]:.2f}<br>"
                    "capacity=%{y:.3f} Ah<br>"
                    "SOH=%{customdata[2]:.2f}%<br>"
                    "scenario multiplier=%{customdata[3]:.2f}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=group["forecast_timestamp"],
                y=group["predicted_future_soh_pct"],
                mode="lines",
                name=f"{scenario} SOH axis",
                line={"width": 0, "color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=1,
            secondary_y=True,
        )

    normal = forecast[forecast["scenario"].eq("normal_usage")]
    if not normal.empty:
        fig.add_trace(
            go.Scatter(
                x=normal["forecast_timestamp"],
                y=normal["predicted_capacity_ah_upper_95"],
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
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
                name="normal 95% residual band",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(245, 158, 11, 0.16)",
                hovertemplate="normal residual band<extra></extra>",
            ),
            row=2,
            col=1,
            secondary_y=False,
        )

    coefficients = ", ".join(f"{value:.6g}" for value in fit["coefficients"])
    summary = (
        f"selected by previous holdout comparison, refit on all valid rows={len(fit['fit_rows']):,} | "
        f"features={','.join(feature_columns)} | "
        f"intercept={fit['intercept']:.4f} | coefficients=[{coefficients}] | "
        f"MAE={fit['mae']:.3f} Ah | RMSE={fit['rmse']:.3f} Ah"
    )
    fig.update_layout(
        title=f"{title}<br><sup>{summary}</sup>",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25, "x": 0},
        margin={"l": 75, "r": 75, "t": 105, "b": 120},
    )
    fig.update_xaxes(title_text=x_feature, row=1, col=1)
    fig.update_xaxes(title_text="Forecast time", rangeslider_visible=True, row=2, col=1)
    fig.update_yaxes(title_text="Capacity Ah", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="SOH %", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Capacity Ah", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="SOH %", row=2, col=1, secondary_y=True)

    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    logger.info("Wrote best capacity-model forecast graph to %s", output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a separate forecast from the selected best capacity-Ah model.",
    )
    parser.add_argument("--serial", default=None, help="Serial to plot when --ml-run-dir is omitted.")
    parser.add_argument("--ml-run-dir", type=Path, default=None, help="Specific ml_run folder to plot.")
    parser.add_argument("--capacity-ml-dir", type=Path, default=None, help="Override capacity ML base directory.")
    parser.add_argument("--output", type=Path, default=None, help="HTML output path.")
    parser.add_argument("--env-file", type=Path, default=None, help="Path to a .env file.")
    parser.add_argument("--title", default=None, help="Plot title.")
    parser.add_argument("--future-days", type=int, default=3650, help="Forecast horizon in days.")
    parser.add_argument("--step-days", type=int, default=30, help="Spacing between future forecast points.")
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Scenario multipliers as label=value,label=value. Defaults to low/normal/high.",
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
        model_config = CapacityModelConfig.from_env(args.env_file)
        scenarios = parse_scenarios(args.scenarios) if args.scenarios else dict(DEFAULT_SCENARIOS)

        if args.ml_run_dir is not None:
            ml_run_dir = args.ml_run_dir
        elif args.serial is not None:
            ml_run_dir = find_latest_ml_run(storage_config.capacity_ml_dir, args.serial)
        else:
            raise ValueError("Provide either --serial or --ml-run-dir.")

        raw_training, summary = load_ml_run_inputs(ml_run_dir)
        nominal_capacity_ah = (
            model_config.nominal_capacity_ah
            if model_config.nominal_capacity_ah is not None
            else float(summary["nominal_capacity_ah"])
        )
        model_name, feature_columns = selected_capacity_ah_model(summary)
        training = prepare_training_table(raw_training)
        fit = fit_selected_model_on_all_rows(training, feature_columns=feature_columns)
        future_features = build_future_feature_table(
            training,
            summary=summary,
            scenarios=scenarios,
            future_days=args.future_days,
            step_days=args.step_days,
        )
        forecast = build_forecast(
            future_features,
            fit=fit,
            feature_columns=feature_columns,
            nominal_capacity_ah=nominal_capacity_ah,
        )

        serial = args.serial or str(training["serial"].iloc[0])
        output_path = (
            _non_overwriting_path(args.output)
            if args.output
            else default_output_path(
                data_dir=storage_config.data_dir,
                serial=serial,
                ml_run_dir=ml_run_dir,
                model_name=model_name,
            )
        )
        plot_best_capacity_model_forecast(
            training,
            forecast,
            fit=fit,
            model_name=model_name,
            feature_columns=feature_columns,
            nominal_capacity_ah=nominal_capacity_ah,
            output_path=output_path,
            title=args.title or f"Serial {serial} best capacity-Ah model forecast",
        )
        logger.info("Selected capacity Ah model: %s", model_name)
        logger.info("Refit rows: %s", len(fit["fit_rows"]))
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Best capacity-model forecast plotting failed.")
        else:
            logger.error("Best capacity-model forecast plotting failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
