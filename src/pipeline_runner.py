from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config import StorageConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PipelineStage:
    key: str
    label: str
    module: str


@dataclass(frozen=True)
class PipelineRunResult:
    stage_key: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    started_at: datetime
    completed_at: datetime
    cwd: Path

    @property
    def success(self) -> bool:
        return self.returncode == 0

    @property
    def elapsed_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def combined_output(self) -> str:
        return "\n".join(part for part in (self.stdout, self.stderr) if part.strip())


STAGES: tuple[PipelineStage, ...] = (
    PipelineStage("extract_raw", "Extract raw telemetry", "src.extract_db"),
    PipelineStage("validate_raw", "Validate raw Parquet", "src.validate_raw_parquet"),
    PipelineStage("build_events", "Build classified rows and events", "src.build_events"),
    PipelineStage(
        "build_capacity_trend",
        "Build capacity trend",
        "src.discharge_cycle_builder",
    ),
    PipelineStage("build_capacity_ml", "Build SOH forecast", "src.capacity_ml"),
    PipelineStage("plot_forecast", "Export interactive forecast", "src.plot_capacity_forecast"),
    PipelineStage(
        "plot_report",
        "Export report image",
        "src.plot_capacity_forecast_report_image",
    ),
)
STAGES_BY_KEY = {stage.key: stage for stage in STAGES}


def _has_value(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _append_value(args: list[str], flag: str, value: Any) -> None:
    if _has_value(value):
        args.extend([flag, str(value)])


def _append_flag(args: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        args.append(flag)


def _append_common_args(args: list[str], options: Mapping[str, Any]) -> None:
    _append_value(args, "--env-file", options.get("env_file"))
    _append_value(args, "--log-level", options.get("log_level"))


def build_stage_args(stage_key: str, options: Mapping[str, Any]) -> list[str]:
    """Build CLI arguments for one pipeline stage from HMI options."""
    if stage_key not in STAGES_BY_KEY:
        raise KeyError(f"Unknown pipeline stage: {stage_key}")

    args: list[str] = []
    serial = options.get("serial")

    if stage_key == "extract_raw":
        _append_value(args, "--serial", serial)
        _append_value(args, "--start", options.get("start"))
        _append_value(args, "--end", options.get("end"))
        _append_value(args, "--limit", options.get("limit"))
        _append_value(args, "--chunk-size", options.get("chunk_size"))
        _append_value(args, "--column-profile", options.get("column_profile"))
        _append_value(args, "--output-dir", options.get("raw_output_dir"))
        _append_value(args, "--compression", options.get("compression"))
        _append_flag(args, "--no-month-split", bool(options.get("no_month_split")))

    elif stage_key == "validate_raw":
        _append_value(args, "--input", options.get("raw_input"))
        _append_value(args, "--serial", serial)
        _append_value(args, "--raw-dir", options.get("raw_dir"))
        _append_flag(args, "--details", bool(options.get("details")))
        _append_value(args, "--max-detail-rows", options.get("max_detail_rows"))

    elif stage_key == "build_events":
        _append_value(args, "--input", options.get("raw_input"))
        _append_value(args, "--serial", serial)
        _append_value(args, "--start", options.get("start"))
        _append_value(args, "--end", options.get("end"))
        _append_value(args, "--raw-dir", options.get("raw_dir"))
        _append_value(args, "--classified-output-dir", options.get("classified_output_dir"))
        _append_value(args, "--event-output-dir", options.get("event_output_dir"))
        _append_value(args, "--rest-current-threshold-a", options.get("rest_current_threshold_a"))
        _append_value(args, "--compression", options.get("compression"))

    elif stage_key == "build_capacity_trend":
        _append_value(args, "--input", options.get("event_input"))
        _append_value(args, "--serial", serial)
        _append_value(args, "--event-run-id", options.get("event_run_id"))
        _append_value(args, "--event-dir", options.get("event_dir"))
        _append_value(args, "--output-dir", options.get("capacity_trend_dir"))
        _append_value(args, "--future-events", options.get("future_events"))
        _append_value(args, "--compression", options.get("compression"))

    elif stage_key == "build_capacity_ml":
        _append_value(args, "--serial", serial)
        _append_value(args, "--capacity-run-dir", options.get("capacity_run_dir"))
        _append_value(args, "--capacity-trend-dir", options.get("capacity_trend_dir"))
        _append_value(args, "--output-dir", options.get("capacity_ml_dir"))
        _append_value(args, "--nominal-capacity-ah", options.get("nominal_capacity_ah"))
        _append_value(args, "--bms-soh-reliable-after", options.get("bms_soh_reliable_after"))
        _append_value(args, "--min-capacity-ah", options.get("min_capacity_ah"))
        _append_value(args, "--max-capacity-ah", options.get("max_capacity_ah"))
        _append_flag(
            args,
            "--exclude-statistical-outliers",
            bool(options.get("exclude_statistical_outliers")),
        )
        _append_value(args, "--future-days", options.get("future_days"))
        _append_value(args, "--step-days", options.get("step_days"))
        _append_value(args, "--training-cutoff", options.get("training_cutoff"))
        _append_value(args, "--validation-end", options.get("validation_end"))
        _append_value(args, "--internal-holdout-fraction", options.get("internal_holdout_fraction"))
        _append_value(args, "--degradation-model", options.get("degradation_model"))
        _append_value(
            args,
            "--degradation-baseline-capacity-ah",
            options.get("degradation_baseline_capacity_ah"),
        )
        _append_value(
            args,
            "--degradation-baseline-method",
            options.get("degradation_baseline_method"),
        )
        if _has_value(options.get("usage_rate_mode")):
            _append_value(args, "--usage-rate-mode", options.get("usage_rate_mode"))
        else:
            _append_value(args, "--usage-rate-modes", options.get("usage_rate_modes"))
        _append_value(args, "--recent-usage-days", options.get("recent_usage_days"))
        _append_value(args, "--scenarios", options.get("scenarios"))
        _append_value(args, "--compression", options.get("compression"))

    elif stage_key == "plot_forecast":
        _append_value(args, "--serial", serial)
        _append_value(args, "--ml-run-dir", options.get("ml_run_dir"))
        _append_value(args, "--capacity-ml-dir", options.get("capacity_ml_dir"))
        _append_value(args, "--output", options.get("forecast_output"))
        _append_value(args, "--title", options.get("forecast_title"))
        _append_value(args, "--max-history-points", options.get("max_history_points"))

    elif stage_key == "plot_report":
        _append_value(args, "--serial", serial)
        _append_value(args, "--ml-run-dir", options.get("ml_run_dir"))
        _append_value(args, "--capacity-ml-dir", options.get("capacity_ml_dir"))
        _append_value(args, "--output", options.get("report_output"))
        _append_value(args, "--title", options.get("report_title"))
        _append_value(args, "--threshold-soh-pct", options.get("threshold_soh_pct"))

    _append_common_args(args, options)
    return args


def build_command(stage_key: str, options: Mapping[str, Any]) -> tuple[str, ...]:
    stage = STAGES_BY_KEY[stage_key]
    return (sys.executable, "-m", stage.module, *build_stage_args(stage_key, options))


def run_stage(
    stage_key: str,
    options: Mapping[str, Any],
    *,
    cwd: Path = PROJECT_ROOT,
) -> PipelineRunResult:
    """Run one pipeline stage in a subprocess and capture console output."""
    command = build_command(stage_key, options)
    started_at = datetime.now(timezone.utc)
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    completed_at = datetime.now(timezone.utc)
    return PipelineRunResult(
        stage_key=stage_key,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        started_at=started_at,
        completed_at=completed_at,
        cwd=cwd,
    )


def run_pipeline(
    stage_keys: Sequence[str],
    options: Mapping[str, Any],
    *,
    cwd: Path = PROJECT_ROOT,
    stop_on_failure: bool = True,
) -> list[PipelineRunResult]:
    results: list[PipelineRunResult] = []
    for stage_key in stage_keys:
        result = run_stage(stage_key, options, cwd=cwd)
        results.append(result)
        if stop_on_failure and not result.success:
            break
    return results


def get_storage_config(env_file: str | Path | None = None) -> "StorageConfig":
    from src.config import StorageConfig

    return StorageConfig.from_env(env_file if _has_value(env_file) else None)


def _serials_under(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        child.name.removeprefix("serial=")
        for child in path.glob("serial=*")
        if child.is_dir()
    }


def discover_serials(env_file: str | Path | None = None) -> list[str]:
    storage = get_storage_config(env_file)
    serials: set[str] = set()
    for path in (
        storage.raw_parquet_dir,
        storage.classified_parquet_dir,
        storage.event_parquet_dir,
        storage.capacity_trend_dir,
        storage.capacity_ml_dir,
    ):
        serials.update(_serials_under(path))
    return sorted(serials)


def latest_run_dir(base_dir: Path, serial: str | int, run_prefix: str) -> Path | None:
    serial_dir = base_dir / f"serial={serial}"
    if not serial_dir.exists():
        return None
    runs = sorted(path for path in serial_dir.rglob(f"{run_prefix}=*") if path.is_dir())
    return runs[-1] if runs else None


def latest_matching_file(base_dir: Path, pattern: str) -> Path | None:
    if not base_dir.exists():
        return None
    files = sorted(path for path in base_dir.glob(pattern) if path.is_file())
    return files[-1] if files else None


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
