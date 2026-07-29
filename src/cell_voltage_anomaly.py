from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.cell_health_features import add_module_configuration_epochs
from src.config import StorageConfig
from src.io_utils import write_parquet_chunk
from src.logging_utils import configure_logging

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "time",
    "serial",
    "sample_count",
    "valid_cell_voltage_sample_count",
    "soc_mean_pct",
    "current_mean_a",
    "current_max_abs_a",
    "module_count",
    "cell_spread_p95_mv",
    "cell_spread_max_mv",
    "compensated_cell_spread_p95_mv",
    "compensated_cell_spread_max_mv",
    "cell_temperature_min_c",
    "cell_temperature_max_c",
}

BASELINE_LEVELS = (
    (
        "epoch_mode_soc_temperature",
        ("module_configuration_epoch", "operating_state", "soc_band_pct", "temperature_band_c"),
    ),
    ("epoch_mode_soc", ("module_configuration_epoch", "operating_state", "soc_band_pct")),
    ("epoch_mode", ("module_configuration_epoch", "operating_state")),
    ("epoch", ("module_configuration_epoch",)),
)


def latest_feature_path(feature_root: Path, serial: str) -> Path:
    candidates = sorted(
        (feature_root / f"serial={serial}").glob(
            "feature_run=*/cell_health_features.parquet"
        )
    )
    if not candidates:
        raise FileNotFoundError(f"No compact cell-health history found for {serial}")
    return candidates[-1]


def load_features(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Cell-health history is missing columns: {', '.join(missing)}")
    if "module_configuration_epoch" not in frame.columns:
        frame = add_module_configuration_epochs(frame)
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    return frame.sort_values("time").reset_index(drop=True)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def prepare_analysis_frame(
    features: pd.DataFrame,
    *,
    aggregation_minutes: int,
    rest_current_threshold_a: float,
    minimum_sample_coverage: float,
    minimum_cell_voltage_coverage: float,
) -> pd.DataFrame:
    if aggregation_minutes <= 0:
        raise ValueError("aggregation_minutes must be positive")
    if rest_current_threshold_a < 0:
        raise ValueError("rest_current_threshold_a must be nonnegative")
    for name, value in {
        "minimum_sample_coverage": minimum_sample_coverage,
        "minimum_cell_voltage_coverage": minimum_cell_voltage_coverage,
    }.items():
        if not 0 < value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    result = features.copy()
    expected_samples = aggregation_minutes * 60
    sample_count = _numeric(result, "sample_count")
    valid_cell_samples = _numeric(result, "valid_cell_voltage_sample_count")
    result["sample_coverage_pct"] = 100.0 * sample_count / expected_samples
    result["cell_voltage_coverage_pct"] = (
        100.0 * valid_cell_samples / sample_count.replace(0, np.nan)
    )

    current_mean = _numeric(result, "current_mean_a")
    current_max_abs = _numeric(result, "current_max_abs_a")
    operating_state = pd.Series("mixed", index=result.index, dtype="object")
    operating_state[current_max_abs <= rest_current_threshold_a] = "rest"
    operating_state[
        (current_max_abs > rest_current_threshold_a)
        & (current_mean > rest_current_threshold_a)
    ] = "charging"
    operating_state[
        (current_max_abs > rest_current_threshold_a)
        & (current_mean < -rest_current_threshold_a)
    ] = "discharging"
    result["operating_state"] = operating_state

    soc = _numeric(result, "soc_mean_pct")
    result["soc_band_pct"] = (np.floor(soc.clip(0, 100) / 10.0) * 10.0).clip(0, 90)
    temperature_mid = (
        _numeric(result, "cell_temperature_min_c")
        + _numeric(result, "cell_temperature_max_c")
    ) / 2.0
    result["temperature_mean_c"] = temperature_mid
    result["temperature_band_c"] = np.floor(temperature_mid / 10.0) * 10.0

    compensated = _numeric(result, "compensated_cell_spread_p95_mv")
    raw = _numeric(result, "cell_spread_p95_mv")
    result["analysis_spread_p95_mv"] = compensated.where(compensated.notna(), raw)
    module_count = _numeric(result, "module_count")
    epoch = _numeric(result, "module_configuration_epoch")
    spread = _numeric(result, "analysis_spread_p95_mv")
    quality_valid = (
        result["time"].notna()
        & module_count.between(1, 4)
        & epoch.notna()
        & soc.between(0, 100)
        & current_mean.notna()
        & temperature_mid.between(-40, 100)
        & spread.between(0, 1000)
        & (sample_count >= expected_samples * minimum_sample_coverage)
        & (valid_cell_samples >= sample_count * minimum_cell_voltage_coverage)
    )
    result["quality_valid"] = quality_valid
    return result


def _baseline_stats(
    values: pd.Series,
    scale_floor_mv: float,
    baseline_quantile: float,
) -> dict[str, float]:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    median = float(finite.median())
    # This is an upper-tail detector. A conventional two-sided MAD lets
    # harmless variation below the median inflate the high-spread threshold,
    # especially for quantized 10/20/30 mV signals. Measure only positive
    # deviations; the high quantile below still protects legitimate baseline
    # tail behavior.
    upper_mad = float((finite - median).clip(lower=0).median())
    scale = max(1.4826 * upper_mad, scale_floor_mv)
    return {
        "count": float(len(finite)),
        "median": median,
        "scale": scale,
        "high_quantile": float(finite.quantile(baseline_quantile)),
    }


def score_anomalies(
    prepared: pd.DataFrame,
    *,
    baseline_days: int,
    minimum_baseline_group_rows: int,
    baseline_quantile: float,
    robust_z_threshold: float,
    minimum_excess_mv: float,
    scale_floor_mv: float,
    absolute_threshold_floor_mv: float,
    absolute_critical_spread_mv: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if baseline_days <= 0 or minimum_baseline_group_rows <= 0:
        raise ValueError("baseline_days and minimum_baseline_group_rows must be positive")
    if not 0.5 < baseline_quantile < 1:
        raise ValueError("baseline_quantile must be between 0.5 and 1")
    if (
        robust_z_threshold <= 0
        or minimum_excess_mv < 0
        or scale_floor_mv <= 0
        or absolute_threshold_floor_mv < 0
        or absolute_critical_spread_mv <= absolute_threshold_floor_mv
    ):
        raise ValueError("absolute critical spread must exceed the positive detection floors")

    result = prepared.copy()
    valid = result[result["quality_valid"]].copy()
    epoch_start = valid.groupby("module_configuration_epoch")["time"].transform("min")
    valid["baseline_candidate"] = valid["time"] < (
        epoch_start + pd.to_timedelta(baseline_days, unit="D")
    )
    baseline = valid[valid["baseline_candidate"]]

    result["baseline_level"] = pd.Series(pd.NA, index=result.index, dtype="object")
    result["baseline_count"] = np.nan
    result["baseline_median_mv"] = np.nan
    result["baseline_scale_mv"] = np.nan
    result["baseline_high_quantile_mv"] = np.nan
    baseline_manifest: list[dict[str, Any]] = []

    for level_name, keys in BASELINE_LEVELS:
        groups: dict[tuple[Any, ...], dict[str, float]] = {}
        grouper: str | list[str] = list(keys) if len(keys) > 1 else keys[0]
        for key, frame in baseline.groupby(grouper, dropna=False, sort=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            stats = _baseline_stats(
                frame["analysis_spread_p95_mv"],
                scale_floor_mv,
                baseline_quantile,
            )
            if stats["count"] < minimum_baseline_group_rows:
                continue
            groups[key_tuple] = stats
            baseline_manifest.append(
                {
                    "level": level_name,
                    "key": [None if pd.isna(value) else value for value in key_tuple],
                    "rows": int(stats["count"]),
                    "median_mv": round(stats["median"], 6),
                    "scale_mv": round(stats["scale"], 6),
                    "high_quantile_mv": round(stats["high_quantile"], 6),
                }
            )

        unassigned = result["quality_valid"] & result["baseline_median_mv"].isna()
        for index, row in result.loc[unassigned, list(keys)].iterrows():
            stats = groups.get(tuple(row[key] for key in keys))
            if stats is None:
                continue
            result.at[index, "baseline_level"] = level_name
            result.at[index, "baseline_count"] = stats["count"]
            result.at[index, "baseline_median_mv"] = stats["median"]
            result.at[index, "baseline_scale_mv"] = stats["scale"]
            result.at[index, "baseline_high_quantile_mv"] = stats["high_quantile"]

    result["baseline_threshold_mv"] = pd.concat(
        [
            result["baseline_median_mv"]
            + robust_z_threshold * result["baseline_scale_mv"],
            result["baseline_high_quantile_mv"] + minimum_excess_mv,
            pd.Series(absolute_threshold_floor_mv, index=result.index),
        ],
        axis=1,
    ).max(axis=1)
    result["spread_excess_mv"] = (
        result["analysis_spread_p95_mv"] - result["baseline_median_mv"]
    )
    result["robust_z_score"] = (
        result["spread_excess_mv"] / result["baseline_scale_mv"]
    )
    result["anomaly_score"] = (
        robust_z_threshold
        + (result["analysis_spread_p95_mv"] - result["baseline_threshold_mv"])
        / result["baseline_scale_mv"]
    ).clip(lower=0, upper=25)
    result["contextual_anomaly_candidate"] = (
        result["quality_valid"]
        & result["baseline_median_mv"].notna()
        & (result["analysis_spread_p95_mv"] >= result["baseline_threshold_mv"])
        & (result["spread_excess_mv"] >= minimum_excess_mv)
    )
    result["absolute_spread_candidate"] = (
        result["quality_valid"]
        & (result["analysis_spread_p95_mv"] >= absolute_critical_spread_mv)
    )
    result["point_anomaly_candidate"] = (
        result["contextual_anomaly_candidate"] | result["absolute_spread_candidate"]
    )
    result["candidate_reason"] = np.select(
        [
            result["contextual_anomaly_candidate"] & result["absolute_spread_candidate"],
            result["absolute_spread_candidate"],
            result["contextual_anomaly_candidate"],
        ],
        ["contextual_and_absolute", "absolute_spread", "contextual"],
        default="none",
    )
    result["in_sustained_episode"] = False
    return result, baseline_manifest


def summarize_module_epoch_baselines(
    scored: pd.DataFrame,
    *,
    baseline_days: int,
    shift_threshold_mv: float,
) -> list[dict[str, Any]]:
    valid = scored[scored["quality_valid"]].copy()
    epochs: list[dict[str, Any]] = []
    previous_p95: float | None = None
    for epoch_id, frame in valid.groupby("module_configuration_epoch", sort=True):
        ordered = frame.sort_values("time")
        start = pd.Timestamp(ordered["time"].iloc[0])
        baseline = ordered[ordered["time"] < start + pd.to_timedelta(baseline_days, unit="D")]
        spread = pd.to_numeric(baseline["analysis_spread_p95_mv"], errors="coerce").dropna()
        if spread.empty:
            continue
        p95 = float(spread.quantile(0.95))
        shift = p95 - previous_p95 if previous_p95 is not None else None
        epochs.append(
            {
                "epoch": int(epoch_id),
                "module_count": int(ordered["module_count"].mode().iloc[0]),
                "start": start.isoformat(),
                "end": pd.Timestamp(ordered["time"].iloc[-1]).isoformat(),
                "baseline_rows": int(len(spread)),
                "baseline_median_mv": round(float(spread.median()), 6),
                "baseline_p95_mv": round(p95, 6),
                "change_from_previous_epoch_p95_mv": (
                    round(shift, 6) if shift is not None else None
                ),
                "elevated_shift_candidate": bool(
                    shift is not None and shift >= shift_threshold_mv
                ),
            }
        )
        previous_p95 = p95
    return epochs


def build_anomaly_episodes(
    scored: pd.DataFrame,
    *,
    aggregation_minutes: int,
    minimum_episode_bins: int,
    episode_gap_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if minimum_episode_bins <= 0 or episode_gap_minutes < aggregation_minutes:
        raise ValueError("episode settings are inconsistent with the aggregation interval")
    candidates = scored[scored["point_anomaly_candidate"]].sort_values("time").copy()
    episode_columns = [
        "episode_id",
        "serial",
        "start_time",
        "end_time",
        "duration_minutes",
        "candidate_bins",
        "module_configuration_epoch",
        "module_count",
        "operating_states",
        "detection_reasons",
        "contextual_candidate_bins",
        "absolute_candidate_bins",
        "maximum_anomaly_score",
        "maximum_spread_mv",
        "median_spread_mv",
        "median_baseline_mv",
        "mean_soc_pct",
        "mean_current_a",
        "mean_temperature_c",
        "severity",
    ]
    if candidates.empty:
        return scored, pd.DataFrame(columns=episode_columns)

    gap = candidates["time"].diff().dt.total_seconds().div(60.0)
    epoch_change = candidates["module_configuration_epoch"].ne(
        candidates["module_configuration_epoch"].shift()
    )
    candidates["_episode_group"] = (
        gap.isna() | gap.gt(episode_gap_minutes) | epoch_change
    ).cumsum()
    serial = str(candidates["serial"].iloc[0])
    episodes: list[dict[str, Any]] = []
    sustained_indexes: list[int] = []
    for _, frame in candidates.groupby("_episode_group", sort=True):
        if len(frame) < minimum_episode_bins:
            continue
        sustained_indexes.extend(frame.index.tolist())
        maximum_score = float(frame["anomaly_score"].max())
        maximum_spread = float(frame["analysis_spread_p95_mv"].max())
        absolute_bins = int(frame["absolute_spread_candidate"].sum())
        contextual_bins = int(frame["contextual_anomaly_candidate"].sum())
        severity = "critical" if absolute_bins or maximum_score >= 12 else "warning"
        start_time = pd.Timestamp(frame["time"].iloc[0])
        end_time = pd.Timestamp(frame["time"].iloc[-1])
        episodes.append(
            {
                "episode_id": f"{serial}-{start_time.strftime('%Y%m%dT%H%M%SZ')}",
                "serial": serial,
                "start_time": start_time,
                "end_time": end_time,
                "duration_minutes": float(
                    (end_time - start_time).total_seconds() / 60.0 + aggregation_minutes
                ),
                "candidate_bins": int(len(frame)),
                "module_configuration_epoch": int(
                    frame["module_configuration_epoch"].mode().iloc[0]
                ),
                "module_count": int(frame["module_count"].mode().iloc[0]),
                "operating_states": ",".join(sorted(frame["operating_state"].unique())),
                "detection_reasons": ",".join(sorted(set(frame["candidate_reason"]) - {"none"})),
                "contextual_candidate_bins": contextual_bins,
                "absolute_candidate_bins": absolute_bins,
                "maximum_anomaly_score": maximum_score,
                "maximum_spread_mv": maximum_spread,
                "median_spread_mv": float(frame["analysis_spread_p95_mv"].median()),
                "median_baseline_mv": float(frame["baseline_median_mv"].median()),
                "mean_soc_pct": float(frame["soc_mean_pct"].mean()),
                "mean_current_a": float(frame["current_mean_a"].mean()),
                "mean_temperature_c": float(frame["temperature_mean_c"].mean()),
                "severity": severity,
            }
        )
    scored.loc[sustained_indexes, "in_sustained_episode"] = True
    return scored, pd.DataFrame(episodes, columns=episode_columns)


def build_daily_summary(scored: pd.DataFrame, *, aggregation_minutes: int) -> pd.DataFrame:
    valid = scored[scored["quality_valid"]].copy()
    if valid.empty:
        return pd.DataFrame()
    valid["day"] = valid["time"].dt.floor("D")

    def summarize_day(frame: pd.DataFrame) -> pd.Series:
        def quantile(column: str, value: float) -> float:
            series = pd.to_numeric(frame[column], errors="coerce").dropna()
            return float(series.quantile(value)) if not series.empty else float("nan")

        sustained = frame["in_sustained_episode"].fillna(False).astype(bool)

        def sustained_minutes(state: str) -> float:
            matches = sustained & frame["operating_state"].eq(state)
            return float(matches.sum() * aggregation_minutes)

        module = pd.to_numeric(frame["module_count"], errors="coerce").dropna()
        return pd.Series(
            {
                "serial": str(frame["serial"].iloc[0]),
                "feature_rows": int(len(frame)),
                "source_sample_rows": int(frame["sample_count"].sum()),
                "module_count": int(module.mode().iloc[0]) if not module.empty else None,
                "module_configuration_epoch": int(
                    frame["module_configuration_epoch"].mode().iloc[0]
                ),
                "spread_p50_mv": quantile("analysis_spread_p95_mv", 0.50),
                "spread_p95_mv": quantile("analysis_spread_p95_mv", 0.95),
                "spread_max_mv": float(frame["analysis_spread_p95_mv"].max()),
                "raw_spread_p95_mv": quantile("cell_spread_p95_mv", 0.95),
                "compensated_spread_p95_mv": quantile(
                    "compensated_cell_spread_p95_mv", 0.95
                ),
                "baseline_threshold_median_mv": float(
                    frame["baseline_threshold_mv"].median()
                ),
                "baseline_threshold_p95_mv": quantile(
                    "baseline_threshold_mv", 0.95
                ),
                "point_candidate_bins": int(frame["point_anomaly_candidate"].sum()),
                "sustained_candidate_bins": int(frame["in_sustained_episode"].sum()),
                "sustained_episode_minutes": float(
                    frame["in_sustained_episode"].sum() * aggregation_minutes
                ),
                "sustained_charging_minutes": sustained_minutes("charging"),
                "sustained_discharging_minutes": sustained_minutes("discharging"),
                "sustained_rest_minutes": sustained_minutes("rest"),
                "sustained_mixed_minutes": sustained_minutes("mixed"),
                "maximum_anomaly_score": float(frame["anomaly_score"].max()),
                "soc_mean_pct": float(frame["soc_mean_pct"].mean()),
                "temperature_mean_c": float(frame["temperature_mean_c"].mean()),
            }
        )

    daily = valid.groupby("day", sort=True).apply(summarize_day, include_groups=False)
    return daily.reset_index().rename(columns={"day": "time"})


def summarize_model(
    scored: pd.DataFrame,
    episodes: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    serial: str,
    source_feature_path: Path,
    baseline_manifest: list[dict[str, Any]],
    epoch_baselines: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    valid_rows = int(scored["quality_valid"].sum())
    scored_rows = int((scored["quality_valid"] & scored["baseline_median_mv"].notna()).sum())
    sustained_rows = int(scored["in_sustained_episode"].sum())
    episode_days = (
        int(pd.to_datetime(episodes["start_time"]).dt.floor("D").nunique())
        if not episodes.empty
        else 0
    )
    finite_spread = pd.to_numeric(scored["analysis_spread_p95_mv"], errors="coerce").dropna()
    epoch_count = int(
        pd.to_numeric(scored["module_configuration_epoch"], errors="coerce").nunique()
    )
    return {
        "model_family": "nca_cell_voltage_anomaly",
        "chemistry": "NCA",
        "serial": serial,
        "source_feature_path": str(source_feature_path.resolve()),
        "source_feature_run_id": source_feature_path.parent.name,
        "analysis_start": pd.Timestamp(scored["time"].min()).isoformat(),
        "analysis_end": pd.Timestamp(scored["time"].max()).isoformat(),
        "feature_rows": int(len(scored)),
        "quality_valid_rows": valid_rows,
        "quality_valid_pct": 100.0 * valid_rows / len(scored) if len(scored) else 0.0,
        "scored_rows": scored_rows,
        "scored_pct_of_valid": 100.0 * scored_rows / valid_rows if valid_rows else 0.0,
        "baseline_group_count": len(baseline_manifest),
        "module_configuration_epoch_count": epoch_count,
        "module_epoch_baselines": epoch_baselines,
        "elevated_module_epoch_shift_count": sum(
            item["elevated_shift_candidate"] for item in epoch_baselines
        ),
        "maximum_module_epoch_baseline_shift_mv": max(
            (
                item["change_from_previous_epoch_p95_mv"]
                for item in epoch_baselines
                if item["change_from_previous_epoch_p95_mv"] is not None
            ),
            default=None,
        ),
        "point_anomaly_candidate_rows": int(scored["point_anomaly_candidate"].sum()),
        "contextual_anomaly_candidate_rows": int(
            scored["contextual_anomaly_candidate"].sum()
        ),
        "absolute_spread_candidate_rows": int(
            scored["absolute_spread_candidate"].sum()
        ),
        "sustained_anomaly_candidate_rows": sustained_rows,
        "sustained_anomaly_episode_count": int(len(episodes)),
        "anomaly_days": episode_days,
        "maximum_anomaly_score": (
            float(scored["anomaly_score"].max()) if scored_rows else None
        ),
        "p95_analysis_spread_mv": (
            float(finite_spread.quantile(0.95)) if not finite_spread.empty else None
        ),
        "maximum_analysis_spread_mv": (
            float(finite_spread.max()) if not finite_spread.empty else None
        ),
        "first_episode_start": (
            pd.Timestamp(episodes["start_time"].min()).isoformat()
            if not episodes.empty
            else None
        ),
        "latest_episode_end": (
            pd.Timestamp(episodes["end_time"].max()).isoformat()
            if not episodes.empty
            else None
        ),
        "daily_rows": int(len(daily)),
        "parameters": parameters,
        "baseline_policy": {
            "metric": "compensated cell-spread p95, with raw spread fallback",
            "method": "one-sided median and MAD baseline",
            "conditioning": [
                "module configuration epoch",
                "operating state",
                "SOC band",
                "temperature band",
            ],
            "fallback_order": [level[0] for level in BASELINE_LEVELS],
        },
        "evidence_status": "unsupervised anomaly candidates; no confirmed weak-cell labels",
        "claim_limit": (
            "Candidates indicate unusual system-level cell-voltage spread under comparable "
            "conditions. They do not identify a specific cell or prove capacity loss."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect conditional NCA cell-voltage spread anomaly candidates."
    )
    parser.add_argument("--serial", required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--aggregation-minutes", type=int, default=5)
    parser.add_argument("--baseline-days", type=int, default=90)
    parser.add_argument("--minimum-baseline-group-rows", type=int, default=100)
    parser.add_argument("--baseline-quantile", type=float, default=0.995)
    parser.add_argument("--robust-z-threshold", type=float, default=6.0)
    parser.add_argument("--minimum-excess-mv", type=float, default=5.0)
    parser.add_argument("--scale-floor-mv", type=float, default=1.0)
    parser.add_argument("--absolute-threshold-floor-mv", type=float, default=25.0)
    parser.add_argument("--absolute-critical-spread-mv", type=float, default=100.0)
    parser.add_argument("--module-epoch-shift-threshold-mv", type=float, default=20.0)
    parser.add_argument("--minimum-episode-bins", type=int, default=3)
    parser.add_argument("--episode-gap-minutes", type=int, default=15)
    parser.add_argument("--rest-current-threshold-a", type=float, default=0.5)
    parser.add_argument("--minimum-sample-coverage", type=float, default=0.5)
    parser.add_argument("--minimum-cell-voltage-coverage", type=float, default=0.95)
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=("snappy", "zstd", "gzip", "none"),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        storage = StorageConfig.from_env()
        feature_root = args.feature_root or storage.data_dir / "processed" / "cell_health"
        feature_path = args.features or latest_feature_path(feature_root, str(args.serial))
        output_root = args.output_root or storage.data_dir / "processed" / "cell_anomaly"
        features = load_features(feature_path)
        prepared = prepare_analysis_frame(
            features,
            aggregation_minutes=args.aggregation_minutes,
            rest_current_threshold_a=args.rest_current_threshold_a,
            minimum_sample_coverage=args.minimum_sample_coverage,
            minimum_cell_voltage_coverage=args.minimum_cell_voltage_coverage,
        )
        scored, baseline_manifest = score_anomalies(
            prepared,
            baseline_days=args.baseline_days,
            minimum_baseline_group_rows=args.minimum_baseline_group_rows,
            baseline_quantile=args.baseline_quantile,
            robust_z_threshold=args.robust_z_threshold,
            minimum_excess_mv=args.minimum_excess_mv,
            scale_floor_mv=args.scale_floor_mv,
            absolute_threshold_floor_mv=args.absolute_threshold_floor_mv,
            absolute_critical_spread_mv=args.absolute_critical_spread_mv,
        )
        scored, episodes = build_anomaly_episodes(
            scored,
            aggregation_minutes=args.aggregation_minutes,
            minimum_episode_bins=args.minimum_episode_bins,
            episode_gap_minutes=args.episode_gap_minutes,
        )
        daily = build_daily_summary(scored, aggregation_minutes=args.aggregation_minutes)
        epoch_baselines = summarize_module_epoch_baselines(
            scored,
            baseline_days=args.baseline_days,
            shift_threshold_mv=args.module_epoch_shift_threshold_mv,
        )
        parameters = {
            "aggregation_minutes": args.aggregation_minutes,
            "baseline_days": args.baseline_days,
            "baseline_scale_method": "upper_semimad",
            "minimum_baseline_group_rows": args.minimum_baseline_group_rows,
            "baseline_quantile": args.baseline_quantile,
            "robust_z_threshold": args.robust_z_threshold,
            "minimum_excess_mv": args.minimum_excess_mv,
            "scale_floor_mv": args.scale_floor_mv,
            "absolute_threshold_floor_mv": args.absolute_threshold_floor_mv,
            "absolute_critical_spread_mv": args.absolute_critical_spread_mv,
            "module_epoch_shift_threshold_mv": args.module_epoch_shift_threshold_mv,
            "minimum_episode_bins": args.minimum_episode_bins,
            "episode_gap_minutes": args.episode_gap_minutes,
            "rest_current_threshold_a": args.rest_current_threshold_a,
            "minimum_sample_coverage": args.minimum_sample_coverage,
            "minimum_cell_voltage_coverage": args.minimum_cell_voltage_coverage,
        }
        summary = summarize_model(
            scored,
            episodes,
            daily,
            serial=str(args.serial),
            source_feature_path=feature_path,
            baseline_manifest=baseline_manifest,
            epoch_baselines=epoch_baselines,
            parameters=parameters,
        )
        run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = output_root / f"serial={args.serial}" / f"anomaly_run={run_stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        compression = None if args.compression == "none" else args.compression
        score_columns = [
            "time",
            "serial",
            "module_configuration_epoch",
            "module_count",
            "operating_state",
            "soc_band_pct",
            "temperature_band_c",
            "sample_coverage_pct",
            "cell_voltage_coverage_pct",
            "quality_valid",
            "cell_spread_p95_mv",
            "compensated_cell_spread_p95_mv",
            "analysis_spread_p95_mv",
            "baseline_level",
            "baseline_count",
            "baseline_median_mv",
            "baseline_scale_mv",
            "baseline_high_quantile_mv",
            "baseline_threshold_mv",
            "spread_excess_mv",
            "robust_z_score",
            "anomaly_score",
            "contextual_anomaly_candidate",
            "absolute_spread_candidate",
            "candidate_reason",
            "point_anomaly_candidate",
            "in_sustained_episode",
        ]
        write_parquet_chunk(
            scored[score_columns],
            run_dir / "cell_voltage_anomaly_scores.parquet",
            compression=compression,
        )
        write_parquet_chunk(
            episodes,
            run_dir / "cell_voltage_anomaly_episodes.parquet",
            compression=compression,
        )
        write_parquet_chunk(
            daily,
            run_dir / "daily_cell_voltage_anomaly.parquet",
            compression=compression,
        )
        (run_dir / "model_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Cell-voltage anomaly analysis complete for serial=%s: %s episodes; %s",
            args.serial,
            f"{len(episodes):,}",
            run_dir,
        )
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Cell-voltage anomaly analysis failed")
        else:
            logger.error("Cell-voltage anomaly analysis failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
