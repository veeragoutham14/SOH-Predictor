from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RAW_PARQUET_DIR = DEFAULT_DATA_DIR / "raw_parquet"
DEFAULT_CLASSIFIED_PARQUET_DIR = DEFAULT_DATA_DIR / "processed" / "classified_rows"
DEFAULT_EVENT_PARQUET_DIR = DEFAULT_DATA_DIR / "processed" / "events"
DEFAULT_CAPACITY_TREND_DIR = DEFAULT_DATA_DIR / "processed" / "capacity_trend"
DEFAULT_CAPACITY_ML_DIR = DEFAULT_DATA_DIR / "processed" / "capacity_ml"
DEFAULT_EXTRACTION_COLUMN_PROFILE = "soh_core"


def load_project_env(env_file: str | Path | None = None) -> None:
    """Load environment variables from a project .env file when available."""
    if env_file is not None:
        load_dotenv(Path(env_file))
        return

    project_env = PROJECT_ROOT / ".env"
    if project_env.exists():
        load_dotenv(project_env)
    else:
        load_dotenv()


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _env_path(name: str, default: Path) -> Path:
    raw_value = os.getenv(name)
    path = Path(raw_value) if raw_value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc


def _env_optional_float(name: str) -> float | None:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return None
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


@dataclass(frozen=True)
class DBConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    connect_timeout_s: int = 30
    application_name: str = "veera_soh_extractor"

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "DBConfig":
        load_project_env(env_file)

        port_value = _required_env("DB_PORT")
        try:
            port = int(port_value)
        except ValueError as exc:
            raise ValueError("DB_PORT must be an integer.") from exc

        try:
            timeout = int(os.getenv("DB_CONNECT_TIMEOUT_S", "30"))
        except ValueError as exc:
            raise ValueError("DB_CONNECT_TIMEOUT_S must be an integer.") from exc

        return cls(
            host=_required_env("DB_HOST"),
            port=port,
            database=_required_env("DB_NAME"),
            user=_required_env("DB_USER"),
            password=_required_env("DB_PASSWORD"),
            connect_timeout_s=timeout,
        )


@dataclass(frozen=True)
class StorageConfig:
    data_dir: Path
    raw_parquet_dir: Path
    classified_parquet_dir: Path
    event_parquet_dir: Path
    capacity_trend_dir: Path
    capacity_ml_dir: Path

    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = None,
        raw_parquet_dir: str | Path | None = None,
        classified_parquet_dir: str | Path | None = None,
        event_parquet_dir: str | Path | None = None,
        capacity_trend_dir: str | Path | None = None,
        capacity_ml_dir: str | Path | None = None,
    ) -> "StorageConfig":
        load_project_env(env_file)

        data_dir = _env_path("DATA_DIR", DEFAULT_DATA_DIR)
        raw_output_dir = Path(raw_parquet_dir) if raw_parquet_dir else _env_path(
            "RAW_PARQUET_DIR",
            DEFAULT_RAW_PARQUET_DIR,
        )
        if not raw_output_dir.is_absolute():
            raw_output_dir = PROJECT_ROOT / raw_output_dir

        classified_output_dir = (
            Path(classified_parquet_dir)
            if classified_parquet_dir
            else _env_path("CLASSIFIED_PARQUET_DIR", DEFAULT_CLASSIFIED_PARQUET_DIR)
        )
        if not classified_output_dir.is_absolute():
            classified_output_dir = PROJECT_ROOT / classified_output_dir

        event_output_dir = (
            Path(event_parquet_dir)
            if event_parquet_dir
            else _env_path("EVENT_PARQUET_DIR", DEFAULT_EVENT_PARQUET_DIR)
        )
        if not event_output_dir.is_absolute():
            event_output_dir = PROJECT_ROOT / event_output_dir

        capacity_output_dir = (
            Path(capacity_trend_dir)
            if capacity_trend_dir
            else _env_path("CAPACITY_TREND_DIR", DEFAULT_CAPACITY_TREND_DIR)
        )
        if not capacity_output_dir.is_absolute():
            capacity_output_dir = PROJECT_ROOT / capacity_output_dir

        capacity_ml_output_dir = (
            Path(capacity_ml_dir)
            if capacity_ml_dir
            else _env_path("CAPACITY_ML_DIR", DEFAULT_CAPACITY_ML_DIR)
        )
        if not capacity_ml_output_dir.is_absolute():
            capacity_ml_output_dir = PROJECT_ROOT / capacity_ml_output_dir

        return cls(
            data_dir=data_dir,
            raw_parquet_dir=raw_output_dir,
            classified_parquet_dir=classified_output_dir,
            event_parquet_dir=event_output_dir,
            capacity_trend_dir=capacity_output_dir,
            capacity_ml_dir=capacity_ml_output_dir,
        )

    def ensure_dirs(self) -> None:
        self.raw_parquet_dir.mkdir(parents=True, exist_ok=True)
        self.classified_parquet_dir.mkdir(parents=True, exist_ok=True)
        self.event_parquet_dir.mkdir(parents=True, exist_ok=True)
        self.capacity_trend_dir.mkdir(parents=True, exist_ok=True)
        self.capacity_ml_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CapacityModelConfig:
    nominal_capacity_ah: float | None = None
    bms_soh_reliable_after: str = "2025-03-21T18:45:00+00:00"
    min_reasonable_capacity_ah: float = 30.0
    max_reasonable_capacity_ah: float = 55.0
    usage_rate_mode: str = "historical_mean"
    usage_rate_modes: str = "historical_mean,historical_median,recent_median"
    recent_usage_days: int = 90

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "CapacityModelConfig":
        load_project_env(env_file)
        return cls(
            nominal_capacity_ah=_env_optional_float("NOMINAL_CAPACITY_AH"),
            bms_soh_reliable_after=os.getenv(
                "BMS_SOH_RELIABLE_AFTER",
                cls.bms_soh_reliable_after,
            ),
            min_reasonable_capacity_ah=_env_float(
                "MIN_REASONABLE_CAPACITY_AH",
                cls.min_reasonable_capacity_ah,
            ),
            max_reasonable_capacity_ah=_env_float(
                "MAX_REASONABLE_CAPACITY_AH",
                cls.max_reasonable_capacity_ah,
            ),
            usage_rate_mode=os.getenv(
                "USAGE_RATE_MODE",
                cls.usage_rate_mode,
            ),
            usage_rate_modes=os.getenv(
                "USAGE_RATE_MODES",
                cls.usage_rate_modes,
            ),
            recent_usage_days=_env_int(
                "RECENT_USAGE_DAYS",
                cls.recent_usage_days,
            ),
        )


@dataclass(frozen=True)
class SignalColumnConfig:
    timestamp_col: str = "time"
    serial_col: str = "serial"
    current_col: str = "bspi2_current_a"
    voltage_col: str = "bspi2_voltage_v"
    soc_col: str = "bspi2_soc_pct"
    soh_col: str = "bspi2_soh_pct"
    mode_col: str = "operating_mode"
    event_id_col: str = "event_id"
    rest_current_threshold_a: float = 0.5
    expected_sample_interval_seconds: float = 1.0
    missing_gap_threshold_seconds: float = 60.0

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "SignalColumnConfig":
        load_project_env(env_file)
        return cls(
            timestamp_col=os.getenv("TIMESTAMP_COL", cls.timestamp_col),
            serial_col=os.getenv("SERIAL_COL", cls.serial_col),
            current_col=os.getenv("CURRENT_COL", cls.current_col),
            voltage_col=os.getenv("VOLTAGE_COL", cls.voltage_col),
            soc_col=os.getenv("SOC_COL", cls.soc_col),
            soh_col=os.getenv("SOH_COL", cls.soh_col),
            mode_col=os.getenv("MODE_COL", cls.mode_col),
            event_id_col=os.getenv("EVENT_ID_COL", cls.event_id_col),
            rest_current_threshold_a=_env_float(
                "REST_CURRENT_THRESHOLD_A",
                cls.rest_current_threshold_a,
            ),
            expected_sample_interval_seconds=_env_float(
                "EXPECTED_SAMPLE_INTERVAL_SECONDS",
                cls.expected_sample_interval_seconds,
            ),
            missing_gap_threshold_seconds=_env_float(
                "MISSING_GAP_THRESHOLD_SECONDS",
                cls.missing_gap_threshold_seconds,
            ),
        )


EXTRACTION_COLUMN_PROFILES: dict[str, tuple[str, ...]] = {
    # Current phase: enough for SOH, SOC state, current/voltage integration,
    # row-level mode classification, and event-level Ah/Wh summaries.
    "soh_core": (
        "timestamp_col",
        "serial_col",
        "soh_col",
        "soc_col",
        "current_col",
        "voltage_col",
    ),
    # Future profiles can be added here without changing extraction logic, for
    # example a thermal anomaly profile with configured temperature columns.
}


def get_extraction_column_profile(profile: str | None = None) -> str:
    """Resolve and validate the configured extraction column profile name."""
    profile_name = (
        profile
        or os.getenv("EXTRACTION_COLUMN_PROFILE", DEFAULT_EXTRACTION_COLUMN_PROFILE)
    ).strip().lower()

    if profile_name not in EXTRACTION_COLUMN_PROFILES:
        available = ", ".join(sorted(EXTRACTION_COLUMN_PROFILES))
        raise ValueError(
            f"Unknown extraction column profile: {profile_name}. "
            f"Available profiles: {available}"
        )
    return profile_name


def get_extraction_columns(
    signal_columns: SignalColumnConfig,
    *,
    profile: str | None = None,
) -> tuple[str, ...]:
    """Return physical database column names for a configured pipeline profile."""
    profile_name = get_extraction_column_profile(profile)
    logical_names = EXTRACTION_COLUMN_PROFILES[profile_name]
    physical_names = [getattr(signal_columns, name) for name in logical_names]
    return tuple(dict.fromkeys(physical_names))
