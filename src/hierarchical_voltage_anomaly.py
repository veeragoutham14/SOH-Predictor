from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from psycopg2 import sql

from src.cell_signal_audit import _qualified_identifier
from src.config import DBConfig, StorageConfig
from src.db import DEFAULT_SERIAL_COL, DEFAULT_TABLE, DEFAULT_TIMESTAMP_COL, open_db_connection
from src.electrothermal_contract import expected_hierarchical_signals
from src.io_utils import write_parquet_chunk
from src.logging_utils import configure_logging

logger = logging.getLogger(__name__)

SEVERITY_BANDS = (
    (10.0, "excellent"),
    (20.0, "acceptable"),
    (30.0, "monitor"),
    (50.0, "anomaly_candidate"),
    (100.0, "engineering_review"),
    (200.0, "serious"),
    (float("inf"), "severe"),
)
SEVERITY_ORDER = {name: index for index, (_, name) in enumerate(SEVERITY_BANDS)}


def engineering_severity(spread_mv: float | int | None) -> str:
    if spread_mv is None or not np.isfinite(float(spread_mv)):
        return "unknown"
    value = max(0.0, float(spread_mv))
    return next(label for upper, label in SEVERITY_BANDS if value < upper)


def _mean_column(normalized_name: str, unit: str) -> str:
    suffix = "mv" if unit == "mV" else "c"
    return f"{normalized_name.removesuffix(f'_{suffix}')}_mean_{suffix}"


def _robust_location_scale(values: np.ndarray, floor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    location = np.nanmedian(values, axis=0)
    upper = np.clip(values - location, 0.0, None)
    scale = np.maximum(1.4826 * np.nanmedian(upper, axis=0), floor)
    return location, scale


def score_hierarchical_windows(
    context: pd.DataFrame,
    voltage: pd.DataFrame,
    system: pd.DataFrame,
    channel_map: pd.DataFrame,
    *,
    baseline_days: int = 90,
    low_rank_components: int = 3,
    robust_z_threshold: float = 6.0,
    ewma_alpha: float = 0.1,
    minimum_coverage: float = 0.95,
    rest_current_threshold_a: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["window_start_utc", "serial"]
    merged = context.merge(system, on=keys, suffixes=("", "_system"), validate="one_to_one")
    for column in (
        "system_temperature_mean_c",
        "system_temperature_min_c",
        "system_temperature_max_c",
    ):
        if column not in merged:
            merged[column] = np.nan
    voltage_channels = channel_map[channel_map["measurement_type"] == "cell_voltage_channel"].copy()
    mean_columns = [
        _mean_column(str(item.normalized_name), str(item.unit))
        for item in voltage_channels.itertuples(index=False)
    ]
    missing = sorted(set(mean_columns) - set(voltage.columns))
    if missing:
        raise ValueError("Voltage feature dataset is missing channels: " + ", ".join(missing))
    merged = merged.merge(voltage[[*keys, *mean_columns]], on=keys, validate="one_to_one")
    merged["window_start_utc"] = pd.to_datetime(merged["window_start_utc"], utc=True)
    merged = merged.sort_values("window_start_utc").reset_index(drop=True)
    current = pd.to_numeric(merged["bms_current_mean_a"], errors="coerce")
    merged["operating_state"] = np.select(
        [current > rest_current_threshold_a, current < -rest_current_threshold_a],
        ["charging", "discharging"],
        default="rest",
    )
    coverage = pd.to_numeric(merged["coverage_fraction"], errors="coerce")
    values = merged[mean_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    finite_fraction = np.isfinite(values).mean(axis=1)
    quality_valid = coverage.ge(minimum_coverage).to_numpy() & (finite_fraction >= minimum_coverage)
    merged["quality_valid"] = quality_valid
    merged["voltage_channel_coverage_fraction"] = finite_fraction

    channel_medians = np.nanmedian(values, axis=0)
    filled = np.where(np.isfinite(values), values, channel_medians)
    peer_residual = np.zeros_like(filled)
    for (module, pack), group in voltage_channels.groupby(["module_index", "pack_index"]):
        indexes = [mean_columns.index(_mean_column(str(item.normalized_name), str(item.unit))) for item in group.itertuples(index=False)]
        pack_values = filled[:, indexes]
        peer_residual[:, indexes] = np.abs(pack_values - np.median(pack_values, axis=1, keepdims=True))

    baseline_end = merged["window_start_utc"].min() + pd.Timedelta(days=baseline_days)
    baseline_mask = quality_valid & merged["window_start_utc"].le(baseline_end).to_numpy()
    if baseline_mask.sum() < max(100, len(mean_columns) * 2):
        baseline_mask = quality_valid
    if baseline_mask.sum() < max(20, len(mean_columns)):
        raise ValueError("Insufficient quality-valid windows for hierarchical baseline")
    peer_location, peer_scale = _robust_location_scale(peer_residual[baseline_mask])
    peer_z = np.clip((peer_residual - peer_location) / peer_scale, 0.0, None)

    baseline_values = filled[baseline_mask]
    center = np.median(baseline_values, axis=0)
    centered_baseline = baseline_values - center
    _, _, right = np.linalg.svd(centered_baseline, full_matrices=False)
    rank = max(1, min(low_rank_components, right.shape[0], right.shape[1]))
    components = right[:rank]
    centered_all = filled - center
    reconstructed = (centered_all @ components.T) @ components
    low_rank_residual = np.abs(centered_all - reconstructed)
    low_rank_location, low_rank_scale = _robust_location_scale(low_rank_residual[baseline_mask])
    low_rank_z = np.clip((low_rank_residual - low_rank_location) / low_rank_scale, 0.0, None)
    temporal_score = pd.DataFrame(low_rank_z).ewm(alpha=ewma_alpha, adjust=False).mean().to_numpy()

    channel_keys = voltage_channels["channel_key"].astype(str).tolist()
    max_peer_index = np.argmax(peer_z, axis=1)
    max_low_rank_index = np.argmax(low_rank_z, axis=1)
    merged["maximum_peer_residual_mv"] = np.max(peer_residual, axis=1)
    merged["maximum_peer_z"] = np.max(peer_z, axis=1)
    merged["maximum_low_rank_residual_mv"] = np.max(low_rank_residual, axis=1)
    merged["maximum_low_rank_z"] = np.max(low_rank_z, axis=1)
    merged["maximum_temporal_ewma_score"] = np.max(temporal_score, axis=1)
    merged["peer_affected_channel_key"] = [channel_keys[index] for index in max_peer_index]
    merged["model_affected_channel_key"] = [channel_keys[index] for index in max_low_rank_index]
    system_spread = pd.to_numeric(
        merged["system_voltage_spread_mv"], errors="coerce"
    ).to_numpy(float)
    reported_spread = pd.to_numeric(
        merged["bms_reported_voltage_spread_max_mv"], errors="coerce"
    ).to_numpy(float)
    reported_spread_plausible = np.isfinite(reported_spread) & (
        (reported_spread >= 0.0) & (reported_spread <= 3000.0)
    )
    analysis_spread = system_spread
    merged["analysis_voltage_spread_mv"] = analysis_spread
    merged["bms_reported_voltage_spread_plausible"] = reported_spread_plausible
    merged["bms_reported_voltage_spread_delta_mv"] = reported_spread - system_spread
    merged["engineering_severity"] = [
        engineering_severity(value) if valid else "unavailable"
        for value, valid in zip(analysis_spread, quality_valid, strict=True)
    ]
    merged["engineering_candidate"] = quality_valid & (analysis_spread >= 30.0)
    merged["contextual_candidate"] = quality_valid & (
        (merged["maximum_peer_z"] >= robust_z_threshold)
        | (merged["maximum_low_rank_z"] >= robust_z_threshold)
        | (merged["maximum_temporal_ewma_score"] >= robust_z_threshold)
    )
    merged["point_anomaly_candidate"] = quality_valid & (
        merged["engineering_candidate"]
        | ((analysis_spread >= 20.0) & merged["contextual_candidate"])
    )
    merged["detection_reason"] = np.select(
        [
            ~merged["quality_valid"],
            merged["engineering_candidate"] & merged["contextual_candidate"],
            merged["engineering_candidate"],
            merged["contextual_candidate"],
        ],
        [
            "quality_invalid",
            "engineering_and_contextual",
            "engineering_band",
            "contextual_peer_or_model",
        ],
        default="not_flagged",
    )
    model = {
        "baseline_start": merged.loc[baseline_mask, "window_start_utc"].min().isoformat(),
        "baseline_end": merged.loc[baseline_mask, "window_start_utc"].max().isoformat(),
        "baseline_rows": int(baseline_mask.sum()),
        "low_rank_components": rank,
        "robust_z_threshold": robust_z_threshold,
        "ewma_alpha": ewma_alpha,
        "peer_baseline_method": "per-channel upper semimad within physical pack peers",
        "low_rank_method": "training-baseline SVD reconstruction with robust upper-tail residual scaling",
        "spread_basis": "hierarchical_channel_means_only",
        "bms_global_spread_role": "reconciliation diagnostic only; never used for anomaly scoring",
        "bms_global_spread_plausible_range_mv": [0, 3000],
        "fixed_engineering_bands_mv": {
            "excellent": [0, 10],
            "acceptable": [10, 20],
            "monitor": [20, 30],
            "anomaly_candidate": [30, 50],
            "engineering_review": [50, 100],
            "serious": [100, 200],
            "severe": [200, None],
        },
    }
    return merged, model


def build_episodes(
    scored: pd.DataFrame,
    *,
    aggregation_minutes: int,
    minimum_episode_bins: int = 3,
    episode_gap_minutes: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = scored.copy()
    result["episode_id"] = pd.NA
    candidates = result[result["point_anomaly_candidate"]].copy()
    if candidates.empty:
        return result, pd.DataFrame(
            columns=(
                "episode_id", "start_time_utc", "end_time_utc", "duration_minutes",
                "candidate_bins", "severity", "maximum_spread_mv", "median_spread_mv",
                "maximum_peer_z", "maximum_low_rank_z", "maximum_temporal_ewma_score",
                "affected_channel_key", "affected_module_index", "affected_pack_index",
                "affected_channel_index", "operating_states", "mean_soc_pct",
                "mean_current_a", "mean_temperature_c", "minimum_temperature_c",
                "maximum_temperature_c", "detection_reasons", "explanation",
            )
        )
    gap = candidates["window_start_utc"].diff().dt.total_seconds().div(60)
    candidates["_group"] = gap.gt(episode_gap_minutes).fillna(True).cumsum()
    episodes: list[dict[str, Any]] = []
    for _, frame in candidates.groupby("_group", sort=True):
        if len(frame) < minimum_episode_bins:
            continue
        start = frame["window_start_utc"].iloc[0]
        end = frame["window_start_utc"].iloc[-1] + pd.Timedelta(minutes=aggregation_minutes)
        episode_id = f"episode_{len(episodes) + 1:05d}"
        result.loc[frame.index, "episode_id"] = episode_id
        severity = max(
            frame["engineering_severity"],
            key=lambda value: SEVERITY_ORDER.get(str(value), -1),
        )
        affected = frame["model_affected_channel_key"].mode()
        channel_key = str(affected.iloc[0]) if not affected.empty else None
        parts = channel_key.split("_") if channel_key else []
        episodes.append(
            {
                "episode_id": episode_id,
                "start_time_utc": start,
                "end_time_utc": end,
                "duration_minutes": float((end - start).total_seconds() / 60),
                "candidate_bins": int(len(frame)),
                "severity": severity,
                "maximum_spread_mv": float(frame["analysis_voltage_spread_mv"].max()),
                "median_spread_mv": float(frame["analysis_voltage_spread_mv"].median()),
                "maximum_peer_z": float(frame["maximum_peer_z"].max()),
                "maximum_low_rank_z": float(frame["maximum_low_rank_z"].max()),
                "maximum_temporal_ewma_score": float(frame["maximum_temporal_ewma_score"].max()),
                "affected_channel_key": channel_key,
                "affected_module_index": int(parts[0][1:]) if len(parts) == 3 else None,
                "affected_pack_index": int(parts[1][1:]) if len(parts) == 3 else None,
                "affected_channel_index": int(parts[2][2:]) if len(parts) == 3 else None,
                "operating_states": ",".join(sorted(frame["operating_state"].unique())),
                "mean_soc_pct": float(frame["bms_soc_mean_pct"].mean()),
                "mean_current_a": float(frame["bms_current_mean_a"].mean()),
                "mean_temperature_c": float(frame["system_temperature_mean_c"].mean()),
                "minimum_temperature_c": float(frame["system_temperature_min_c"].min()),
                "maximum_temperature_c": float(frame["system_temperature_max_c"].max()),
                "detection_reasons": ",".join(sorted(frame["detection_reason"].unique())),
                "explanation": (
                    f"{len(frame)} persistent {aggregation_minutes}-minute bins; "
                    f"maximum spread {frame['analysis_voltage_spread_mv'].max():.1f} mV; "
                    f"dominant channel {channel_key or 'unknown'}"
                ),
            }
        )
    return result, pd.DataFrame(episodes)


def daily_summary(scored: pd.DataFrame, aggregation_minutes: int) -> pd.DataFrame:
    frame = scored.copy()
    frame["day_utc"] = frame["window_start_utc"].dt.floor("D")
    valid = frame["quality_valid"].fillna(False).astype(bool)
    frame["_valid_analysis_spread_mv"] = frame["analysis_voltage_spread_mv"].where(valid)
    for column in ("maximum_peer_z", "maximum_low_rank_z", "maximum_temporal_ewma_score"):
        frame[f"_valid_{column}"] = frame[column].where(valid)
    sustained = frame["episode_id"].notna()
    grouped = frame.groupby("day_utc", as_index=False).agg(
        feature_windows=("window_start_utc", "size"),
        quality_valid_windows=("quality_valid", "sum"),
        quality_invalid_windows=("quality_valid", lambda values: int((~values.astype(bool)).sum())),
        bms_global_spread_implausible_windows=(
            "bms_reported_voltage_spread_plausible",
            lambda values: int((~values.astype(bool)).sum()),
        ),
        system_spread_p50_mv=("_valid_analysis_spread_mv", "median"),
        system_spread_p95_mv=("_valid_analysis_spread_mv", lambda values: values.quantile(0.95)),
        system_spread_max_mv=("_valid_analysis_spread_mv", "max"),
        maximum_peer_z=("_valid_maximum_peer_z", "max"),
        maximum_low_rank_z=("_valid_maximum_low_rank_z", "max"),
        maximum_temporal_ewma_score=("_valid_maximum_temporal_ewma_score", "max"),
        point_candidate_windows=("point_anomaly_candidate", "sum"),
        soc_mean_pct=("bms_soc_mean_pct", "mean"),
        current_mean_a=("bms_current_mean_a", "mean"),
        temperature_mean_c=("system_temperature_mean_c", "mean"),
    )
    episode_counts = frame.loc[sustained].groupby("day_utc")["episode_id"].nunique()
    grouped["sustained_episode_count"] = grouped["day_utc"].map(episode_counts).fillna(0).astype(int)
    sustained_bins = frame.loc[sustained].groupby("day_utc").size()
    grouped["sustained_episode_minutes"] = (
        grouped["day_utc"].map(sustained_bins).fillna(0) * aggregation_minutes
    )
    for state in ("charging", "discharging", "rest"):
        state_bins = frame.loc[sustained & frame["operating_state"].eq(state)].groupby("day_utc").size()
        grouped[f"sustained_{state}_minutes"] = (
            grouped["day_utc"].map(state_bins).fillna(0) * aggregation_minutes
        )
    grouped["sustained_mixed_minutes"] = 0.0
    return grouped


def _read_feature_run(feature_root: Path, serial: str) -> tuple[Path, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = sorted((feature_root / f"serial={serial}").glob("feature_run=*/model_summary.json"))
    if not summaries:
        raise ValueError(f"No hierarchical feature run found for serial {serial}")
    run_summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summaries]
    run_dir = summaries[-1].parent
    summary = run_summaries[-1]
    aggregation = int(summary["aggregation_minutes"])
    module_count = int(summary["module_count"])
    compatible = [
        path.parent
        for path, item in zip(summaries, run_summaries, strict=True)
        if int(item.get("aggregation_minutes") or 0) == aggregation
        and int(item.get("module_count") or 0) == module_count
    ]
    suffix = f"{aggregation}min.parquet"
    def read_named(prefix: str) -> pd.DataFrame:
        paths = [
            path
            for compatible_run in compatible
            for path in sorted(compatible_run.glob("year=*/month=*/" + prefix + suffix))
        ]
        if not paths:
            raise ValueError(f"Feature run is missing {prefix}{suffix}")
        combined = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
        return (
            combined.sort_values("window_start_utc")
            .drop_duplicates(["window_start_utc", "serial"], keep="last")
            .reset_index(drop=True)
        )
    summary["compatible_feature_run_ids"] = [path.name for path in compatible]
    return (
        run_dir,
        summary,
        read_named("operating_context_"),
        read_named("voltage_channel_features_"),
        read_named("system_features_"),
        pd.read_parquet(run_dir / "channel_map.parquet"),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_evidence(
    connection: Any,
    episodes: pd.DataFrame,
    *,
    serial: str,
    module_count: int,
    output_dir: Path,
    padding_minutes: int,
    maximum_episodes: int,
    compression: str | None,
    table: str,
    timestamp_col: str,
    serial_col: str,
) -> list[dict[str, Any]]:
    if episodes.empty or maximum_episodes <= 0:
        return []
    signals = expected_hierarchical_signals(module_count)
    columns = [timestamp_col, serial_col, *[item.source_signal_name for item in signals], "bspi2_soc_pct", "bspi2_soh_pct", "bspi2_current_a", "bspi2_voltage_v", "bspi2_power_w", *[name for name in ("bspi2_cellvoltagemin_v", "bspi2_cellvoltagemax_v", "bspi2_celltemperaturemin_c", "bspi2_celltemperaturemax_c")]]
    query = sql.SQL(
        "SELECT {columns} FROM {table} WHERE {serial_col} = %s AND {timestamp} >= %s AND {timestamp} < %s ORDER BY {timestamp}"
    ).format(
        columns=sql.SQL(", ").join(sql.Identifier(name) for name in columns),
        table=_qualified_identifier(table),
        serial_col=sql.Identifier(serial_col),
        timestamp=sql.Identifier(timestamp_col),
    )
    selected = episodes.sort_values("maximum_spread_mv", ascending=False).head(maximum_episodes)
    manifest: list[dict[str, Any]] = []
    for item in selected.itertuples(index=False):
        start = pd.Timestamp(item.start_time_utc).to_pydatetime() - timedelta(minutes=padding_minutes)
        end = pd.Timestamp(item.end_time_utc).to_pydatetime() + timedelta(minutes=padding_minutes)
        with connection.cursor() as cursor:
            cursor.execute(query, (serial, start, end))
            rows = cursor.fetchall()
            names = [column[0] for column in cursor.description]
        if not rows:
            continue
        frame = pd.DataFrame.from_records(rows, columns=names)
        frame.insert(0, "episode_id", item.episode_id)
        path = output_dir / f"episode_id={item.episode_id}" / "episode_evidence_1s.parquet"
        write_parquet_chunk(frame, path, compression=compression)
        manifest.append({"episode_id": item.episode_id, "rows": len(frame), "path": str(path.relative_to(output_dir.parent)), "sha256": _sha256(path), "start_utc": start.isoformat(), "end_utc": end.isoformat()})
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect hierarchical BMS voltage anomalies.")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--baseline-days", type=int, default=90)
    parser.add_argument("--low-rank-components", type=int, default=3)
    parser.add_argument("--robust-z-threshold", type=float, default=6.0)
    parser.add_argument("--ewma-alpha", type=float, default=0.1)
    parser.add_argument("--minimum-coverage", type=float, default=0.95)
    parser.add_argument("--rest-current-threshold-a", type=float, default=0.5)
    parser.add_argument("--minimum-episode-bins", type=int, default=3)
    parser.add_argument("--episode-gap-minutes", type=int, default=15)
    parser.add_argument("--evidence-padding-minutes", type=int, default=30)
    parser.add_argument("--maximum-evidence-episodes", type=int, default=50)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--timestamp-col", default=DEFAULT_TIMESTAMP_COL)
    parser.add_argument("--serial-col", default=DEFAULT_SERIAL_COL)
    parser.add_argument("--compression", default="zstd", choices=("snappy", "zstd", "gzip", "none"))
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        storage = StorageConfig.from_env(args.env_file)
        feature_root = args.feature_root or storage.data_dir / "processed" / "bms_hierarchical_features"
        feature_run, feature_summary, context, voltage, system, channel_map = _read_feature_run(feature_root, str(args.serial))
        aggregation = int(feature_summary["aggregation_minutes"])
        scored, model = score_hierarchical_windows(
            context,
            voltage,
            system,
            channel_map,
            baseline_days=args.baseline_days,
            low_rank_components=args.low_rank_components,
            robust_z_threshold=args.robust_z_threshold,
            ewma_alpha=args.ewma_alpha,
            minimum_coverage=args.minimum_coverage,
            rest_current_threshold_a=args.rest_current_threshold_a,
        )
        scored, episodes = build_episodes(
            scored,
            aggregation_minutes=aggregation,
            minimum_episode_bins=args.minimum_episode_bins,
            episode_gap_minutes=args.episode_gap_minutes,
        )
        daily = daily_summary(scored, aggregation)
        output_root = args.output_root or storage.data_dir / "processed" / "hierarchical_cell_anomaly"
        run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = output_root / f"serial={args.serial}" / f"anomaly_run={run_stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        compression = None if args.compression == "none" else args.compression
        write_parquet_chunk(scored, run_dir / "anomaly_windows.parquet", compression=compression)
        write_parquet_chunk(episodes, run_dir / "anomaly_episodes.parquet", compression=compression)
        write_parquet_chunk(daily, run_dir / "daily_anomaly_summary.parquet", compression=compression)
        evidence_manifest: list[dict[str, Any]] = []
        if args.maximum_evidence_episodes > 0 and not episodes.empty:
            db_config = DBConfig.from_env(args.env_file)
            with open_db_connection(db_config) as connection:
                evidence_manifest = _extract_evidence(
                    connection,
                    episodes,
                    serial=str(args.serial),
                    module_count=int(feature_summary["module_count"]),
                    output_dir=run_dir / "evidence",
                    padding_minutes=args.evidence_padding_minutes,
                    maximum_episodes=args.maximum_evidence_episodes,
                    compression=compression,
                    table=args.table,
                    timestamp_col=args.timestamp_col,
                    serial_col=args.serial_col,
                )
        valid_scored = scored.loc[scored["quality_valid"]]
        reported_spread = pd.to_numeric(
            scored["bms_reported_voltage_spread_max_mv"], errors="coerce"
        )
        summary = {
            "model_family": "hierarchical_cell_voltage_anomaly",
            "serial": str(args.serial),
            "module_count": int(feature_summary["module_count"]),
            "analysis_start": scored["window_start_utc"].min().isoformat(),
            "analysis_end": scored["window_start_utc"].max().isoformat(),
            "feature_rows": len(scored),
            "quality_valid_rows": int(scored["quality_valid"].sum()),
            "quality_valid_pct": float(100.0 * scored["quality_valid"].mean()),
            "point_anomaly_windows": int(scored["point_anomaly_candidate"].sum()),
            "sustained_anomaly_episode_count": len(episodes),
            "anomaly_days": int((daily["sustained_episode_count"] > 0).sum()),
            "p95_analysis_spread_mv": float(valid_scored["analysis_voltage_spread_mv"].quantile(0.95)),
            "maximum_analysis_spread_mv": float(valid_scored["analysis_voltage_spread_mv"].max()),
            "maximum_bms_reported_voltage_spread_mv": float(reported_spread.max()),
            "bms_global_spread_implausible_windows": int(
                (~scored["bms_reported_voltage_spread_plausible"]).sum()
            ),
            "spread_basis": "hierarchical_channel_means_only",
            "maximum_peer_z": float(valid_scored["maximum_peer_z"].max()),
            "maximum_low_rank_z": float(valid_scored["maximum_low_rank_z"].max()),
            "maximum_temporal_ewma_score": float(valid_scored["maximum_temporal_ewma_score"].max()),
            "first_episode_start": (
                episodes["start_time_utc"].min().isoformat() if not episodes.empty else None
            ),
            "latest_episode_end": (
                episodes["end_time_utc"].max().isoformat() if not episodes.empty else None
            ),
            "episode_severity_counts": (
                episodes["severity"].value_counts().sort_index().to_dict()
                if not episodes.empty else {}
            ),
            "source_feature_run_id": feature_run.name,
            "source_feature_run_ids": feature_summary.get("compatible_feature_run_ids", [feature_run.name]),
            "detector": model,
            "evidence_episode_count": len(evidence_manifest),
            "evidence_policy": "One-second BMS database telemetry is retained only around the highest-spread sustained episodes.",
            "context_value_semantics": "SOC, current, and temperature are five-minute aggregates of BMS-reported database values for each detection window.",
            "ground_truth_status": "No independent cell-voltage instrumentation label",
            "claim_limit": "Anomalies describe BMS-reported channel inconsistency, not a confirmed physical cell defect.",
        }
        manifest = {
            "manifest_version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "source_feature_run": feature_run.name,
            "detector": model,
            "evidence": evidence_manifest,
        }
        (run_dir / "model_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (run_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        logger.info("Hierarchical anomaly analysis complete: %s", run_dir)
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Hierarchical anomaly analysis failed")
        else:
            logger.error("Hierarchical anomaly analysis failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
