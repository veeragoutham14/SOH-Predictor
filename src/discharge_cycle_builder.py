from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import SignalColumnConfig, StorageConfig
from src.io_utils import collect_parquet_files, write_parquet_chunk
from src.logging_utils import configure_logging


logger = logging.getLogger(__name__)

MODE_DISCHARGING = "discharging"
MODE_CHARGING = "charging"
MODE_REST = "rest"
MODE_MISSING = "missing"
PARTITION_RE = re.compile(r"([^=]+)=(.+)")
FULL_SOC_ANCHOR = 100.0
EMPTY_SOC_ANCHOR = 0.0


@dataclass(frozen=True)
class CapacityTrendResult:
    usage_events: int
    discharge_episodes: int
    complete_discharges: int
    output_dir: Path


def _as_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    work = df.copy()
    for column in columns:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    return work


def _safe_float(value: Any, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def _add_capacity_duration_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add rest/discharge duration quality signals for full capacity rows."""
    work = df.copy()
    discharge_seconds = pd.to_numeric(
        work["discharge_duration_seconds"],
        errors="coerce",
    )
    rest_seconds = pd.to_numeric(
        work["rest_duration_seconds"],
        errors="coerce",
    )
    total_seconds = discharge_seconds + rest_seconds

    work["rest_to_discharge_ratio"] = np.where(
        discharge_seconds.gt(0),
        rest_seconds / discharge_seconds,
        np.nan,
    )
    work["rest_fraction"] = np.where(
        total_seconds.gt(0),
        rest_seconds / total_seconds,
        np.nan,
    )
    work["capacity_measurement_duration_valid"] = (
        discharge_seconds.gt(0) & rest_seconds.ge(0)
    )
    work["long_rest_capacity_measurement"] = (
        pd.Series(work["rest_to_discharge_ratio"], index=work.index)
        .gt(1.0)
        .fillna(False)
    )
    work["capacity_measurement_quality"] = np.select(
        [
            ~work["capacity_measurement_duration_valid"],
            work["long_rest_capacity_measurement"],
        ],
        [
            "invalid_duration",
            "long_rest_interrupted",
        ],
        default="clean_anchor_100_to_0",
    )
    return work


def _path_partition_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for part in path.parts:
        match = PARTITION_RE.fullmatch(part)
        if match is not None:
            values[match.group(1)] = match.group(2)
    return values


def _select_event_files(
    files: Sequence[Path],
    *,
    event_run_id: str | None = None,
) -> list[Path]:
    """Keep one event run per month unless a specific run id is requested."""
    if event_run_id:
        selected = [
            file for file in files if _path_partition_values(file).get("event_run") == event_run_id
        ]
        if not selected:
            raise FileNotFoundError(f"No event files found for event_run={event_run_id}")
        return selected

    grouped: dict[tuple[str, str, str], list[tuple[str, Path]]] = {}
    unpartitioned: list[Path] = []
    for file in files:
        values = _path_partition_values(file)
        run_id = values.get("event_run")
        serial = values.get("serial")
        year = values.get("year")
        month = values.get("month")
        if run_id is None or serial is None or year is None or month is None:
            unpartitioned.append(file)
            continue
        grouped.setdefault((serial, year, month), []).append((run_id, file))

    selected = list(unpartitioned)
    for candidates in grouped.values():
        latest_run = max(run_id for run_id, _ in candidates)
        selected.extend(file for run_id, file in candidates if run_id == latest_run)
    return sorted(selected)


def load_event_table(
    input_path: Path,
    *,
    serial: str | int | None = None,
    event_run_id: str | None = None,
) -> pd.DataFrame:
    """Load event Parquet files and return a sorted, de-duplicated event table."""
    files = _select_event_files(collect_parquet_files(input_path), event_run_id=event_run_id)
    logger.info("Found %s event Parquet files under %s", len(files), input_path)

    parts = [pd.read_parquet(file) for file in files]
    events = pd.concat(parts, ignore_index=True)
    if events.empty:
        return events

    if serial is not None:
        events = events[events["serial"].astype(str).eq(str(serial))]

    events["start_timestamp"] = pd.to_datetime(
        events["start_timestamp"],
        errors="coerce",
        utc=True,
    )
    events["end_timestamp"] = pd.to_datetime(
        events["end_timestamp"],
        errors="coerce",
        utc=True,
    )
    events = events.dropna(subset=["serial", "mode", "start_timestamp", "end_timestamp"])
    events = _as_numeric(
        events,
        [
            "duration_seconds",
            "num_rows",
            "start_soc",
            "end_soc",
            "mean_soc",
            "start_soh",
            "end_soh",
            "mean_soh",
            "throughput_ah",
            "energy_wh",
        ],
    )

    events = events.drop_duplicates(
        subset=["serial", "mode", "start_timestamp", "end_timestamp"],
        keep="last",
    )
    events = events.sort_values(["serial", "start_timestamp"], kind="mergesort")
    events = events.reset_index(drop=True)
    events["_event_row_id"] = range(len(events))
    return events


def build_discharge_usage_ledger(events: pd.DataFrame) -> pd.DataFrame:
    """Create an all-discharge usage ledger with cumulative Ah/Wh throughput."""
    if events.empty:
        return pd.DataFrame()

    discharge = events[events["mode"].eq(MODE_DISCHARGING)].copy()
    if discharge.empty:
        return pd.DataFrame()

    discharge = discharge.sort_values(["serial", "start_timestamp"], kind="mergesort")
    discharge["discharge_usage_id"] = discharge.groupby("serial").cumcount() + 1
    discharge["discharge_ah"] = discharge["throughput_ah"].fillna(0.0)
    discharge["discharge_wh"] = discharge["energy_wh"].fillna(0.0)
    discharge["soc_drop"] = discharge["start_soc"] - discharge["end_soc"]
    discharge["cumulative_all_discharge_ah"] = discharge.groupby("serial")[
        "discharge_ah"
    ].cumsum()
    discharge["cumulative_all_discharge_wh"] = discharge.groupby("serial")[
        "discharge_wh"
    ].cumsum()

    return discharge[
        [
            "discharge_usage_id",
            "_event_row_id",
            "event_id",
            "serial",
            "start_timestamp",
            "end_timestamp",
            "duration_seconds",
            "start_soc",
            "end_soc",
            "soc_drop",
            "start_soh",
            "end_soh",
            "discharge_ah",
            "discharge_wh",
            "cumulative_all_discharge_ah",
            "cumulative_all_discharge_wh",
        ]
    ].reset_index(drop=True)


def _start_episode(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "serial": event["serial"],
        "start_timestamp": event["start_timestamp"],
        "end_timestamp": event["end_timestamp"],
        "start_soc": event["start_soc"],
        "end_soc": event["end_soc"],
        "start_soh": event["start_soh"],
        "end_soh": event["end_soh"],
        "discharge_ah": 0.0,
        "discharge_wh": 0.0,
        "discharge_duration_seconds": 0.0,
        "elapsed_duration_seconds": 0.0,
        "num_discharge_events": 0,
        "rest_events_inside": 0,
        "rest_duration_seconds_inside": 0.0,
        "source_event_ids": [],
        "last_discharge_event_row_id": None,
    }


def _add_discharge_event(episode: dict[str, Any], event: dict[str, Any]) -> None:
    episode["end_timestamp"] = event["end_timestamp"]
    episode["end_soc"] = event["end_soc"]
    episode["end_soh"] = event["end_soh"]
    episode["discharge_ah"] += float(event.get("throughput_ah") or 0.0)
    episode["discharge_wh"] += float(event.get("energy_wh") or 0.0)
    episode["discharge_duration_seconds"] += float(event.get("duration_seconds") or 0.0)
    episode["elapsed_duration_seconds"] = max(
        0.0,
        (
            pd.Timestamp(episode["end_timestamp"])
            - pd.Timestamp(episode["start_timestamp"])
        ).total_seconds(),
    )
    episode["num_discharge_events"] += 1
    episode["source_event_ids"].append(str(event.get("event_id", "")))
    episode["last_discharge_event_row_id"] = event["_event_row_id"]


def _add_rest_event(episode: dict[str, Any], event: dict[str, Any]) -> None:
    episode["rest_events_inside"] += 1
    episode["rest_duration_seconds_inside"] += float(event.get("duration_seconds") or 0.0)


def build_discharge_episodes(events: pd.DataFrame, usage_ledger: pd.DataFrame) -> pd.DataFrame:
    """Group discharging events separated by rest into discharge episodes."""
    if events.empty:
        return pd.DataFrame()

    cumulative_by_event_row = {}
    if not usage_ledger.empty:
        cumulative_by_event_row = usage_ledger.set_index("_event_row_id")[
            ["cumulative_all_discharge_ah", "cumulative_all_discharge_wh"]
        ].to_dict(orient="index")

    episodes: list[dict[str, Any]] = []
    active_by_serial: dict[str, dict[str, Any]] = {}

    def close_episode(serial_key: str) -> None:
        active = active_by_serial.pop(serial_key, None)
        if active is None or active["num_discharge_events"] == 0:
            return

        cumulative = cumulative_by_event_row.get(active["last_discharge_event_row_id"], {})
        active["cumulative_all_discharge_ah"] = cumulative.get(
            "cumulative_all_discharge_ah",
            np.nan,
        )
        active["cumulative_all_discharge_wh"] = cumulative.get(
            "cumulative_all_discharge_wh",
            np.nan,
        )
        active["soc_drop"] = active["start_soc"] - active["end_soc"]
        active["source_event_ids"] = ",".join(active["source_event_ids"])
        episodes.append(active)

    for event in events.to_dict(orient="records"):
        serial_key = str(event["serial"])
        mode = event["mode"]

        if mode == MODE_DISCHARGING:
            active = active_by_serial.get(serial_key)
            if active is None:
                active = _start_episode(event)
                active_by_serial[serial_key] = active
            _add_discharge_event(active, event)
            continue

        if mode == MODE_REST and serial_key in active_by_serial:
            _add_rest_event(active_by_serial[serial_key], event)
            continue

        # Charging, missing, and unknown modes end the current discharge episode.
        close_episode(serial_key)

    for serial_key in list(active_by_serial):
        close_episode(serial_key)

    if not episodes:
        return pd.DataFrame()

    episode_df = pd.DataFrame(episodes)
    episode_df = episode_df.sort_values(["serial", "start_timestamp"], kind="mergesort")
    episode_df["discharge_episode_id"] = episode_df.groupby("serial").cumcount() + 1
    ordered = ["discharge_episode_id"] + [
        column for column in episode_df.columns if column != "discharge_episode_id"
    ]
    return episode_df[ordered].reset_index(drop=True)


def _full_soc_anchor(
    event: dict[str, Any],
) -> tuple[pd.Timestamp, float] | None:
    """Return the best available time/SOH for an exact 100% SOC anchor."""
    start_soc = _safe_float(event.get("start_soc"), np.nan)
    end_soc = _safe_float(event.get("end_soc"), np.nan)
    start_ok = pd.notna(start_soc) and start_soc == FULL_SOC_ANCHOR
    end_ok = pd.notna(end_soc) and end_soc == FULL_SOC_ANCHOR

    mode = event.get("mode")
    if mode == MODE_CHARGING:
        if end_ok:
            return pd.Timestamp(event["end_timestamp"]), _safe_float(event.get("end_soh"), np.nan)
        if start_ok:
            return pd.Timestamp(event["start_timestamp"]), _safe_float(event.get("start_soh"), np.nan)
        return None

    if start_ok:
        return pd.Timestamp(event["start_timestamp"]), _safe_float(event.get("start_soh"), np.nan)
    if end_ok:
        return pd.Timestamp(event["end_timestamp"]), _safe_float(event.get("end_soh"), np.nan)
    return None


def _empty_soc_anchor(
    event: dict[str, Any],
) -> tuple[pd.Timestamp, float] | None:
    """Return the best available time/SOH for an exact 0% SOC anchor."""
    start_soc = _safe_float(event.get("start_soc"), np.nan)
    end_soc = _safe_float(event.get("end_soc"), np.nan)
    start_ok = pd.notna(start_soc) and start_soc == EMPTY_SOC_ANCHOR
    end_ok = pd.notna(end_soc) and end_soc == EMPTY_SOC_ANCHOR

    if end_ok:
        return pd.Timestamp(event["end_timestamp"]), _safe_float(event.get("end_soh"), np.nan)
    if start_ok:
        return pd.Timestamp(event["start_timestamp"]), _safe_float(event.get("start_soh"), np.nan)
    return None


def _start_capacity_window(
    event: dict[str, Any],
    *,
    start_anchor_time: pd.Timestamp,
    bms_soh_start: float,
) -> dict[str, Any]:
    return {
        "serial": event["serial"],
        "start_anchor_time": start_anchor_time,
        "capacity_ah": 0.0,
        "capacity_wh": 0.0,
        "bms_soh_start": bms_soh_start,
        "discharge_duration_seconds": 0.0,
        "rest_duration_seconds": 0.0,
        "num_discharge_events": 0,
        "source_event_ids": [],
        "last_discharge_event_row_id": None,
    }


def _add_capacity_discharge_event(window: dict[str, Any], event: dict[str, Any]) -> None:
    window["capacity_ah"] += _safe_float(event.get("throughput_ah"))
    window["capacity_wh"] += _safe_float(event.get("energy_wh"))
    window["discharge_duration_seconds"] += _safe_float(event.get("duration_seconds"))
    window["num_discharge_events"] += 1
    window["source_event_ids"].append(str(event.get("event_id", "")))
    window["last_discharge_event_row_id"] = event["_event_row_id"]


def _add_capacity_rest_event(window: dict[str, Any], event: dict[str, Any]) -> None:
    window["rest_duration_seconds"] += _safe_float(event.get("duration_seconds"))


def build_capacity_measurements(
    events: pd.DataFrame,
    usage_ledger: pd.DataFrame,
) -> pd.DataFrame:
    """Build anchor-based full-discharge capacity measurements.

    A valid measurement starts at exact 100% SOC, allows rest, sums only
    discharging throughput, and closes only at exact 0% SOC. Intermediate SOC
    values are not used for capacity scaling because the BMS SOC can drift.
    """
    if events.empty:
        return pd.DataFrame()

    cumulative_by_event_row = {}
    if not usage_ledger.empty:
        cumulative_by_event_row = usage_ledger.set_index("_event_row_id")[
            ["cumulative_all_discharge_ah", "cumulative_all_discharge_wh"]
        ].to_dict(orient="index")

    measurements: list[dict[str, Any]] = []
    active_by_serial: dict[str, dict[str, Any]] = {}

    def start_or_replace_window(serial_key: str, event: dict[str, Any]) -> bool:
        anchor = _full_soc_anchor(event)
        if anchor is None:
            return False
        anchor_time, anchor_soh = anchor
        active_by_serial[serial_key] = _start_capacity_window(
            event,
            start_anchor_time=anchor_time,
            bms_soh_start=anchor_soh,
        )
        return True

    def close_window(
        serial_key: str,
        *,
        end_anchor_time: pd.Timestamp,
        bms_soh_end: float,
    ) -> None:
        active = active_by_serial.pop(serial_key, None)
        if active is None or active["num_discharge_events"] == 0:
            return

        cumulative = cumulative_by_event_row.get(active["last_discharge_event_row_id"], {})
        measurements.append(
            {
                "serial": active["serial"],
                "start_anchor_time": active["start_anchor_time"],
                "end_anchor_time": end_anchor_time,
                "capacity_ah": active["capacity_ah"],
                "capacity_wh": active["capacity_wh"],
                "cumulative_all_discharge_ah": cumulative.get(
                    "cumulative_all_discharge_ah",
                    np.nan,
                ),
                "cumulative_all_discharge_wh": cumulative.get(
                    "cumulative_all_discharge_wh",
                    np.nan,
                ),
                "bms_soh_start": active["bms_soh_start"],
                "bms_soh_end": bms_soh_end,
                "discharge_duration_seconds": active["discharge_duration_seconds"],
                "rest_duration_seconds": active["rest_duration_seconds"],
                "num_discharge_events": active["num_discharge_events"],
                "source_event_ids": ",".join(active["source_event_ids"]),
                "quality_flag": "full_anchor_100_to_0",
            }
        )

    for event in events.to_dict(orient="records"):
        serial_key = str(event["serial"])
        mode = event["mode"]

        active = active_by_serial.get(serial_key)
        if active is None:
            start_or_replace_window(serial_key, event)
            active = active_by_serial.get(serial_key)
            if active is None:
                continue

        if mode == MODE_MISSING:
            active_by_serial.pop(serial_key, None)
            continue

        if mode == MODE_CHARGING:
            if not start_or_replace_window(serial_key, event):
                active_by_serial.pop(serial_key, None)
            continue

        if mode == MODE_REST:
            if active["num_discharge_events"] == 0:
                start_or_replace_window(serial_key, event)
                active = active_by_serial.get(serial_key, active)
            _add_capacity_rest_event(active, event)

            empty_anchor = _empty_soc_anchor(
                event,
            )
            if empty_anchor is not None:
                end_anchor_time, end_anchor_soh = empty_anchor
                close_window(
                    serial_key,
                    end_anchor_time=end_anchor_time,
                    bms_soh_end=end_anchor_soh,
                )
            continue

        if mode == MODE_DISCHARGING:
            _add_capacity_discharge_event(active, event)
            empty_anchor = _empty_soc_anchor(
                event,
            )
            if empty_anchor is not None:
                end_anchor_time, end_anchor_soh = empty_anchor
                close_window(
                    serial_key,
                    end_anchor_time=end_anchor_time,
                    bms_soh_end=end_anchor_soh,
                )
            continue

        active_by_serial.pop(serial_key, None)

    if not measurements:
        return pd.DataFrame()

    complete = pd.DataFrame(measurements)
    complete = complete.sort_values(["serial", "start_anchor_time"], kind="mergesort")
    complete["event_discharge_id"] = complete.groupby("serial").cumcount() + 1
    complete = _add_capacity_duration_quality_columns(complete)

    columns = [
        "event_discharge_id",
        "serial",
        "start_anchor_time",
        "end_anchor_time",
        "capacity_ah",
        "capacity_wh",
        "cumulative_all_discharge_ah",
        "cumulative_all_discharge_wh",
        "bms_soh_start",
        "bms_soh_end",
        "discharge_duration_seconds",
        "rest_duration_seconds",
        "rest_to_discharge_ratio",
        "rest_fraction",
        "capacity_measurement_duration_valid",
        "long_rest_capacity_measurement",
        "capacity_measurement_quality",
        "num_discharge_events",
        "source_event_ids",
        "quality_flag",
    ]
    return complete[columns].reset_index(drop=True)


def _fit_linear_trend(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 2 or valid["x"].nunique() < 2:
        return np.nan, np.nan

    slope, intercept = np.polyfit(valid["x"].astype(float), valid["y"].astype(float), 1)
    return float(intercept), float(slope)


def _predict(intercept: float, slope: float, x: pd.Series) -> pd.Series:
    if np.isnan(intercept) or np.isnan(slope):
        return pd.Series(np.nan, index=x.index)
    return intercept + slope * x.astype(float)


def build_capacity_trend_table(
    measurements: pd.DataFrame,
    *,
    future_events: int = 20,
) -> pd.DataFrame:
    """Add measured trend values and simple scenario-based future predictions."""
    if measurements.empty:
        return pd.DataFrame()

    measured = measurements.copy()
    measured["row_type"] = "measured"

    ah_event_intercept, ah_event_slope = _fit_linear_trend(
        measured["event_discharge_id"],
        measured["capacity_ah"],
    )
    ah_usage_intercept, ah_usage_slope = _fit_linear_trend(
        measured["cumulative_all_discharge_ah"],
        measured["capacity_ah"],
    )
    wh_event_intercept, wh_event_slope = _fit_linear_trend(
        measured["event_discharge_id"],
        measured["capacity_wh"],
    )
    wh_usage_intercept, wh_usage_slope = _fit_linear_trend(
        measured["cumulative_all_discharge_wh"],
        measured["capacity_wh"],
    )

    measured["capacity_ah_trend_by_event_id"] = _predict(
        ah_event_intercept,
        ah_event_slope,
        measured["event_discharge_id"],
    )
    measured["capacity_ah_trend_by_cumulative_ah"] = _predict(
        ah_usage_intercept,
        ah_usage_slope,
        measured["cumulative_all_discharge_ah"],
    )
    measured["capacity_wh_trend_by_event_id"] = _predict(
        wh_event_intercept,
        wh_event_slope,
        measured["event_discharge_id"],
    )
    measured["capacity_wh_trend_by_cumulative_wh"] = _predict(
        wh_usage_intercept,
        wh_usage_slope,
        measured["cumulative_all_discharge_wh"],
    )

    if future_events <= 0:
        return measured

    future_parts = []
    for serial, group in measured.groupby("serial", sort=False):
        last = group.sort_values("event_discharge_id").iloc[-1]
        latest_capacity_ah = float(last["capacity_ah"])
        latest_capacity_wh = float(last["capacity_wh"])

        for step in range(1, future_events + 1):
            future_event_id = int(last["event_discharge_id"]) + step
            future_cumulative_ah = (
                float(last["cumulative_all_discharge_ah"]) + latest_capacity_ah * step
            )
            future_cumulative_wh = (
                float(last["cumulative_all_discharge_wh"]) + latest_capacity_wh * step
            )
            future_parts.append(
                {
                    "event_discharge_id": future_event_id,
                    "serial": serial,
                    "start_anchor_time": pd.NaT,
                    "end_anchor_time": pd.NaT,
                    "capacity_ah": np.nan,
                    "capacity_wh": np.nan,
                    "cumulative_all_discharge_ah": future_cumulative_ah,
                    "cumulative_all_discharge_wh": future_cumulative_wh,
                    "bms_soh_start": np.nan,
                    "bms_soh_end": np.nan,
                    "discharge_duration_seconds": np.nan,
                    "rest_duration_seconds": np.nan,
                    "num_discharge_events": np.nan,
                    "source_event_ids": "",
                    "quality_flag": "predicted_scenario",
                    "row_type": "predicted",
                    "capacity_ah_trend_by_event_id": (
                        ah_event_intercept + ah_event_slope * future_event_id
                        if not np.isnan(ah_event_slope)
                        else np.nan
                    ),
                    "capacity_ah_trend_by_cumulative_ah": (
                        ah_usage_intercept + ah_usage_slope * future_cumulative_ah
                        if not np.isnan(ah_usage_slope)
                        else np.nan
                    ),
                    "capacity_wh_trend_by_event_id": (
                        wh_event_intercept + wh_event_slope * future_event_id
                        if not np.isnan(wh_event_slope)
                        else np.nan
                    ),
                    "capacity_wh_trend_by_cumulative_wh": (
                        wh_usage_intercept + wh_usage_slope * future_cumulative_wh
                        if not np.isnan(wh_usage_slope)
                        else np.nan
                    ),
                }
            )

    future = pd.DataFrame(future_parts)
    return pd.concat([measured, future], ignore_index=True)


def write_capacity_outputs(
    *,
    usage_ledger: pd.DataFrame,
    episodes: pd.DataFrame,
    measurements: pd.DataFrame,
    trend_table: pd.DataFrame,
    output_dir: Path,
    compression: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet_chunk(usage_ledger, output_dir / "discharge_usage_ledger.parquet", compression=compression)
    write_parquet_chunk(episodes, output_dir / "discharge_episodes.parquet", compression=compression)
    write_parquet_chunk(measurements, output_dir / "capacity_measurements.parquet", compression=compression)
    write_parquet_chunk(trend_table, output_dir / "capacity_trend.parquet", compression=compression)


def build_discharge_capacity_trend(
    *,
    event_input_path: Path,
    output_base_dir: Path,
    serial: str | int | None = None,
    event_run_id: str | None = None,
    future_events: int = 20,
    compression: str | None = "snappy",
    run_id: str | None = None,
) -> CapacityTrendResult:
    """Build discharge usage, complete-discharge capacity, and trend outputs."""
    events = load_event_table(event_input_path, serial=serial, event_run_id=event_run_id)
    if events.empty:
        raise ValueError("No event rows found for capacity trend building.")

    usage_ledger = build_discharge_usage_ledger(events)
    episodes = build_discharge_episodes(events, usage_ledger)
    measurements = build_capacity_measurements(
        events,
        usage_ledger,
    )
    trend_table = build_capacity_trend_table(measurements, future_events=future_events)

    serial_label = str(serial or events["serial"].iloc[0])
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_base_dir / f"serial={serial_label}" / f"capacity_run={run_id}"
    write_capacity_outputs(
        usage_ledger=usage_ledger,
        episodes=episodes,
        measurements=measurements,
        trend_table=trend_table,
        output_dir=output_dir,
        compression=compression,
    )

    logger.info("Discharge usage events: %s", len(usage_ledger))
    logger.info("Discharge episodes: %s", len(episodes))
    logger.info("Complete exact 100-to-0 discharges: %s", len(measurements))
    logger.info("Wrote capacity trend outputs to: %s", output_dir)

    return CapacityTrendResult(
        usage_events=len(usage_ledger),
        discharge_episodes=len(episodes),
        complete_discharges=len(measurements),
        output_dir=output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build complete-discharge capacity trend tables from event Parquet.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Event Parquet file or directory.")
    parser.add_argument("--serial", default=None, help="Serial to process when --input is omitted.")
    parser.add_argument("--event-run-id", default=None, help="Optional event_run id to use.")
    parser.add_argument("--event-dir", type=Path, default=None, help="Override event Parquet base directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override capacity trend output directory.")
    parser.add_argument("--env-file", type=Path, default=None, help="Path to a .env file.")
    parser.add_argument("--future-events", type=int, default=20, help="Number of future full discharge events to estimate.")
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=("snappy", "zstd", "gzip", "none"),
        help="Parquet compression codec.",
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
            event_parquet_dir=args.event_dir,
            capacity_trend_dir=args.output_dir,
        )
        SignalColumnConfig.from_env(args.env_file)

        if args.input is not None:
            event_input_path = args.input
        elif args.serial is not None:
            event_input_path = storage_config.event_parquet_dir / f"serial={args.serial}"
        else:
            raise ValueError("Provide either --input or --serial.")

        compression = None if args.compression == "none" else args.compression
        build_discharge_capacity_trend(
            event_input_path=event_input_path,
            output_base_dir=storage_config.capacity_trend_dir,
            serial=args.serial,
            event_run_id=args.event_run_id,
            future_events=args.future_events,
            compression=compression,
        )
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Capacity trend build failed.")
        else:
            logger.error("Capacity trend build failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
