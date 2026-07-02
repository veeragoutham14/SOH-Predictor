from __future__ import annotations

import base64
import sys
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_runner import (  # noqa: E402
    STAGES,
    STAGES_BY_KEY,
    build_command,
    discover_serials,
    get_storage_config,
    latest_matching_file,
    latest_run_dir,
    read_json,
    run_stage,
)


WORKFLOWS = {
    "forecast": {
        "label": "Forecast From Latest Capacity Data",
        "description": "Train the SOH forecast, export the interactive plot, and export the report image.",
        "stages": ("build_capacity_ml", "plot_forecast", "plot_report"),
    },
    "local_processing": {
        "label": "Process Existing Raw Data",
        "description": "Validate local raw Parquet, build events, build capacity trend, then forecast.",
        "stages": (
            "validate_raw",
            "build_events",
            "build_capacity_trend",
            "build_capacity_ml",
            "plot_forecast",
            "plot_report",
        ),
    },
    "database_full": {
        "label": "Extract From Database And Forecast",
        "description": "Extract telemetry from TimescaleDB, then run the complete local processing pipeline.",
        "stages": (
            "extract_raw",
            "validate_raw",
            "build_events",
            "build_capacity_trend",
            "build_capacity_ml",
            "plot_forecast",
            "plot_report",
        ),
    },
}


def _blank_to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _format_datetime(day: date | None, clock: time | None) -> str | None:
    if day is None:
        return None
    value = datetime.combine(day, clock or time.min, tzinfo=timezone.utc)
    return value.isoformat()


def _safe_discover_serials(env_file: str | None) -> list[str]:
    try:
        return discover_serials(env_file)
    except Exception:
        return []


def _safe_storage(env_file: str | None):
    try:
        return get_storage_config(env_file)
    except Exception as exc:
        st.error(f"Storage configuration failed: {exc}")
        return get_storage_config(None)


def _latest_outputs(storage, serial: str) -> dict[str, Path | None]:
    validation_dir = storage.data_dir / "validation_plots"
    return {
        "capacity_run": latest_run_dir(storage.capacity_trend_dir, serial, "capacity_run"),
        "ml_run": latest_run_dir(storage.capacity_ml_dir, serial, "ml_run"),
        "html": latest_matching_file(
            validation_dir,
            f"capacity_usage_forecast_serial_{serial}*.html",
        ),
        "png": latest_matching_file(
            validation_dir,
            f"capacity_forecast_report_serial_{serial}*.png",
        ),
    }


def _metric_value(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def _quote_command(command: tuple[str, ...]) -> str:
    parts = []
    for part in command:
        parts.append(f'"{part}"' if any(char.isspace() for char in part) else part)
    return " ".join(parts)


def _stage_labels(stage_keys: tuple[str, ...] | list[str]) -> list[str]:
    return [STAGES_BY_KEY[stage_key].label for stage_key in stage_keys]


def _run_selected_stages(stage_keys: tuple[str, ...] | list[str], options: dict[str, Any]) -> None:
    st.session_state["pipeline_results"] = []
    progress = st.progress(0.0)
    status_box = st.empty()

    for index, stage_key in enumerate(stage_keys, start=1):
        stage = STAGES_BY_KEY[stage_key]
        status_box.info(f"Running: {stage.label}")
        result = run_stage(stage_key, options)
        st.session_state["pipeline_results"].append(result)
        progress.progress(index / len(stage_keys))
        if not result.success:
            status_box.error(f"{stage.label} failed with exit code {result.returncode}.")
            return

    status_box.success("Workflow completed.")


def _workflow_card(workflow_key: str, options: dict[str, Any]) -> None:
    workflow = WORKFLOWS[workflow_key]
    with st.container(border=True):
        st.subheader(workflow["label"])
        st.caption(workflow["description"])
        st.write(" -> ".join(_stage_labels(workflow["stages"])))
        disabled = workflow_key == "database_full" and (
            not options.get("start") or not options.get("end")
        )
        if st.button(
            f"Run {workflow['label']}",
            key=f"run_{workflow_key}",
            type="primary" if workflow_key == "forecast" else "secondary",
            disabled=disabled,
            use_container_width=True,
        ):
            _run_selected_stages(workflow["stages"], options)


st.set_page_config(page_title="SOH Pipeline HMI", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.25rem;}
    div[data-testid="stMetric"] {
        background: #f7f7f8;
        border: 1px solid #e5e7eb;
        padding: 0.8rem;
        border-radius: 0.5rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 0.5rem;
    }
    code {white-space: pre-wrap;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("SOH Pipeline HMI")

default_env = PROJECT_ROOT / ".env"
with st.sidebar:
    st.header("Battery")
    env_file = st.text_input(
        "Configuration",
        value=str(default_env) if default_env.exists() else "",
        help="Optional .env file. Leave blank to use defaults.",
    )
    env_file_value = _blank_to_none(env_file)
    serials = _safe_discover_serials(env_file_value)
    detected_serial = st.selectbox(
        "Battery serial",
        [""] + serials,
        index=1 if serials else 0,
    )
    manual_serial = st.text_input(
        "Manual serial",
        value="" if detected_serial else "300000172",
        help="Use only when the serial is not detected from local data.",
    ).strip()
    serial = detected_serial or manual_serial

    st.header("Database Window")
    extraction_start_date = st.date_input("Start date", value=None)
    extraction_start_time = st.time_input("Start time", value=time.min)
    extraction_end_date = st.date_input("End date", value=None)
    extraction_end_time = st.time_input("End time", value=time.min)

    st.header("Forecast")
    forecast_horizon_years = st.slider("Forecast horizon years", 1, 15, 10)
    step_days = st.select_slider("Forecast step", options=[7, 14, 30, 60, 90], value=30)
    degradation_model = st.selectbox(
        "Degradation model",
        ["usage_linear", "usage_calendar_constrained"],
        index=0,
    )
    usage_profile = st.selectbox(
        "Usage profile",
        ["All profiles", "Historical mean", "Historical median", "Recent median"],
        index=0,
    )
    threshold_soh_pct = st.slider("SOH threshold", 50.0, 95.0, 80.0, step=1.0)

    with st.expander("Advanced Settings", expanded=False):
        log_level = st.selectbox("Log level", ["INFO", "DEBUG", "WARNING", "ERROR"], index=0)
        compression = st.selectbox("Parquet compression", ["snappy", "zstd", "gzip", "none"], index=0)
        chunk_size = st.number_input("Extraction chunk size", min_value=1, value=100000, step=25000)
        limit = st.number_input("Extraction row limit", min_value=0, value=0, step=10000)
        rest_current_threshold_a = st.number_input(
            "Rest current threshold A",
            min_value=0.0,
            value=0.5,
            step=0.1,
        )
        future_events = st.number_input("Future events", min_value=0, value=20, step=1)
        training_cutoff = st.text_input("Training cutoff", value="")
        validation_end = st.text_input("Validation end", value="")
        scenarios = st.text_input("Scenario multipliers", value="")
        raw_input = st.text_input("Raw input override", value="")
        event_input = st.text_input("Event input override", value="")
        capacity_run_dir = st.text_input("Capacity run override", value="")
        ml_run_dir = st.text_input("ML run override", value="")
        raw_dir = st.text_input("Raw directory override", value="")
        classified_dir = st.text_input("Classified directory override", value="")
        event_dir = st.text_input("Event directory override", value="")
        capacity_trend_dir = st.text_input("Capacity trend directory override", value="")
        capacity_ml_dir = st.text_input("Capacity ML directory override", value="")

storage = _safe_storage(env_file_value)
latest = _latest_outputs(storage, serial) if serial else {}

usage_rate_mode_map = {
    "Historical mean": "historical_mean",
    "Historical median": "historical_median",
    "Recent median": "recent_median",
}
usage_rate_mode = usage_rate_mode_map.get(usage_profile)
usage_rate_modes = (
    None
    if usage_rate_mode
    else "historical_mean,historical_median,recent_median"
)

options = {
    "serial": serial,
    "env_file": env_file_value,
    "log_level": log_level,
    "start": _format_datetime(extraction_start_date, extraction_start_time),
    "end": _format_datetime(extraction_end_date, extraction_end_time),
    "limit": limit if limit else None,
    "chunk_size": chunk_size,
    "column_profile": "soh_core",
    "compression": compression,
    "raw_input": _blank_to_none(raw_input),
    "event_input": _blank_to_none(event_input),
    "capacity_run_dir": _blank_to_none(capacity_run_dir),
    "ml_run_dir": _blank_to_none(ml_run_dir),
    "raw_dir": _blank_to_none(raw_dir),
    "raw_output_dir": _blank_to_none(raw_dir),
    "event_dir": _blank_to_none(event_dir),
    "capacity_trend_dir": _blank_to_none(capacity_trend_dir),
    "capacity_ml_dir": _blank_to_none(capacity_ml_dir),
    "classified_output_dir": _blank_to_none(classified_dir),
    "event_output_dir": _blank_to_none(event_dir),
    "rest_current_threshold_a": rest_current_threshold_a,
    "future_events": future_events,
    "details": False,
    "max_detail_rows": None,
    "degradation_model": degradation_model,
    "training_cutoff": _blank_to_none(training_cutoff),
    "validation_end": _blank_to_none(validation_end),
    "future_days": int(forecast_horizon_years * 365),
    "step_days": step_days,
    "usage_rate_mode": usage_rate_mode,
    "usage_rate_modes": usage_rate_modes,
    "scenarios": _blank_to_none(scenarios),
    "threshold_soh_pct": threshold_soh_pct,
    "max_history_points": 50000,
}

run_tab, outputs_tab, audit_tab = st.tabs(["Operate", "Outputs", "Audit"])

with run_tab:
    status_cols = st.columns(4)
    status_cols[0].metric("Battery serial", serial or "Not selected")
    status_cols[1].metric("Latest capacity run", latest.get("capacity_run").name if latest.get("capacity_run") else "None")
    status_cols[2].metric("Latest ML run", latest.get("ml_run").name if latest.get("ml_run") else "None")
    status_cols[3].metric("Forecast horizon", f"{forecast_horizon_years} years")

    st.subheader("Operator Workflows")
    workflow_cols = st.columns(3)
    with workflow_cols[0]:
        _workflow_card("forecast", options)
    with workflow_cols[1]:
        _workflow_card("local_processing", options)
    with workflow_cols[2]:
        _workflow_card("database_full", options)

    st.subheader("Run Status")
    results = st.session_state.get("pipeline_results", [])
    if not results:
        st.info("No workflow has been run in this session.")
    for result in results:
        stage = STAGES_BY_KEY[result.stage_key]
        status = "Completed" if result.success else "Failed"
        with st.expander(f"{status}: {stage.label} ({result.elapsed_seconds:.1f}s)", expanded=not result.success):
            st.code(result.combined_output or "No output.", language="text")

with outputs_tab:
    ml_run = latest.get("ml_run")
    summary = read_json(ml_run / "model_summary.json") if ml_run else {}
    metric_cols = st.columns(4)
    degradation = summary.get("degradation_forecast_model", {})
    metric_cols[0].metric("Nominal capacity", _metric_value(summary.get("nominal_capacity_ah"), " Ah"))
    metric_cols[1].metric("Training rows", _metric_value(summary.get("training_rows")))
    metric_cols[2].metric("Forecast rows", _metric_value(summary.get("forecast_rows")))
    metric_cols[3].metric("Model", degradation.get("model_name", "n/a"))

    html_plot = latest.get("html")
    if html_plot:
        st.subheader("Interactive Forecast")
        html_data = base64.b64encode(html_plot.read_bytes()).decode("ascii")
        st.iframe(f"data:text/html;base64,{html_data}", height=720, width="stretch")
        st.download_button(
            "Download HTML Report",
            data=html_plot.read_bytes(),
            file_name=html_plot.name,
            mime="text/html",
        )

    png_report = latest.get("png")
    if png_report:
        st.subheader("Report Image")
        st.image(str(png_report), width="stretch")
        st.download_button(
            "Download PNG Report",
            data=png_report.read_bytes(),
            file_name=png_report.name,
            mime="image/png",
        )

    if not html_plot and not png_report:
        st.info("No exported reports found for this serial yet.")

with audit_tab:
    st.subheader("Execution Trace")
    st.write("Commands are shown here only for audit and debugging. Operators do not need to type them.")
    for workflow_key, workflow in WORKFLOWS.items():
        with st.expander(workflow["label"], expanded=False):
            for stage_key in workflow["stages"]:
                st.code(_quote_command(build_command(stage_key, options)), language="powershell")

    st.subheader("Storage")
    st.json(
        {
            "project_root": str(PROJECT_ROOT),
            "data_dir": str(storage.data_dir),
            "raw_parquet_dir": str(storage.raw_parquet_dir),
            "classified_parquet_dir": str(storage.classified_parquet_dir),
            "event_parquet_dir": str(storage.event_parquet_dir),
            "capacity_trend_dir": str(storage.capacity_trend_dir),
            "capacity_ml_dir": str(storage.capacity_ml_dir),
        }
    )

    st.subheader("Available Stages")
    st.table(
        [
            {"Stage": stage.label, "Module": stage.module}
            for stage in STAGES
        ]
    )
