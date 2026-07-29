from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from src.cell_voltage_anomaly import (
    build_anomaly_episodes,
    build_daily_summary,
    prepare_analysis_frame,
    score_anomalies,
)


def _features() -> pd.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for index in range(120):
        timestamp = start + timedelta(minutes=5 * index)
        spread = 10.0 + (index % 3 - 1) * 0.2
        rows.append(_row(timestamp, spread))
    for index in range(3):
        rows.append(_row(start + timedelta(days=2, minutes=5 * index), 32.0 + index))
    return pd.DataFrame(rows)


def _row(timestamp: datetime, spread: float) -> dict[str, object]:
    return {
        "time": timestamp,
        "serial": "300000083",
        "sample_count": 255,
        "valid_cell_voltage_sample_count": 255,
        "soc_mean_pct": 55.0,
        "current_mean_a": 0.0,
        "current_max_abs_a": 0.2,
        "module_count": 2,
        "module_configuration_epoch": 1,
        "cell_spread_p95_mv": spread + 1.0,
        "cell_spread_max_mv": spread + 3.0,
        "compensated_cell_spread_p95_mv": spread,
        "compensated_cell_spread_max_mv": spread + 2.0,
        "cell_temperature_min_c": 20.0,
        "cell_temperature_max_c": 22.0,
    }


def test_conditional_robust_model_detects_sustained_spread_episode() -> None:
    prepared = prepare_analysis_frame(
        _features(),
        aggregation_minutes=5,
        rest_current_threshold_a=0.5,
        minimum_sample_coverage=0.5,
        minimum_cell_voltage_coverage=0.95,
    )
    scored, manifest = score_anomalies(
        prepared,
        baseline_days=1,
        minimum_baseline_group_rows=20,
        baseline_quantile=0.995,
        robust_z_threshold=6.0,
        minimum_excess_mv=5.0,
        scale_floor_mv=1.0,
        absolute_threshold_floor_mv=25.0,
        absolute_critical_spread_mv=100.0,
    )
    scored, episodes = build_anomaly_episodes(
        scored,
        aggregation_minutes=5,
        minimum_episode_bins=2,
        episode_gap_minutes=15,
    )
    daily = build_daily_summary(scored, aggregation_minutes=5)

    assert manifest
    assert prepared["quality_valid"].all()
    assert scored["point_anomaly_candidate"].sum() == 3
    assert scored["in_sustained_episode"].sum() == 3
    assert len(episodes) == 1
    assert episodes.iloc[0]["duration_minutes"] == 15.0
    assert daily["sustained_candidate_bins"].sum() == 3
    assert daily["sustained_episode_minutes"].sum() == 15.0
    assert daily["sustained_rest_minutes"].sum() == 15.0
    assert daily["sustained_charging_minutes"].sum() == 0.0
    assert daily["sustained_discharging_minutes"].sum() == 0.0
    assert daily["sustained_mixed_minutes"].sum() == 0.0


def test_absolute_spread_cannot_be_normalized_away() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        _row(start + timedelta(minutes=5 * index), (120.0, 130.0, 140.0)[index % 3])
        for index in range(120)
    ]
    rows.extend(
        _row(start + timedelta(days=2, minutes=5 * index), 170.0)
        for index in range(3)
    )
    prepared = prepare_analysis_frame(
        pd.DataFrame(rows),
        aggregation_minutes=5,
        rest_current_threshold_a=0.5,
        minimum_sample_coverage=0.5,
        minimum_cell_voltage_coverage=0.95,
    )
    scored, _ = score_anomalies(
        prepared,
        baseline_days=1,
        minimum_baseline_group_rows=20,
        baseline_quantile=0.995,
        robust_z_threshold=6.0,
        minimum_excess_mv=5.0,
        scale_floor_mv=1.0,
        absolute_threshold_floor_mv=25.0,
        absolute_critical_spread_mv=150.0,
    )
    scored, episodes = build_anomaly_episodes(
        scored,
        aggregation_minutes=5,
        minimum_episode_bins=3,
        episode_gap_minutes=15,
    )

    retained = scored[scored["time"] >= start + timedelta(days=2)]
    assert retained["absolute_spread_candidate"].all()
    assert retained["point_anomaly_candidate"].all()
    assert len(episodes) == 1
    assert episodes.iloc[0]["severity"] == "critical"
    assert episodes.iloc[0]["absolute_candidate_bins"] == 3
    assert episodes.iloc[0]["detection_reasons"] == "contextual_and_absolute"


def test_low_coverage_rows_are_excluded_before_scoring() -> None:
    features = _features()
    features.loc[0, "sample_count"] = 10
    features.loc[0, "valid_cell_voltage_sample_count"] = 10

    prepared = prepare_analysis_frame(
        features,
        aggregation_minutes=5,
        rest_current_threshold_a=0.5,
        minimum_sample_coverage=0.5,
        minimum_cell_voltage_coverage=0.95,
    )

    assert bool(prepared.loc[0, "quality_valid"]) is False


def test_lower_tail_quantization_does_not_inflate_upper_threshold() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    baseline_spreads = [10.0] * 40 + [20.0] * 50 + [30.0] * 30
    rows = [
        _row(start + timedelta(minutes=5 * index), spread)
        for index, spread in enumerate(baseline_spreads)
    ]
    rows.append(_row(start + timedelta(days=2), 40.0))
    prepared = prepare_analysis_frame(
        pd.DataFrame(rows),
        aggregation_minutes=5,
        rest_current_threshold_a=0.5,
        minimum_sample_coverage=0.5,
        minimum_cell_voltage_coverage=0.95,
    )

    scored, _ = score_anomalies(
        prepared,
        baseline_days=1,
        minimum_baseline_group_rows=20,
        baseline_quantile=0.995,
        robust_z_threshold=6.0,
        minimum_excess_mv=5.0,
        scale_floor_mv=1.0,
        absolute_threshold_floor_mv=25.0,
        absolute_critical_spread_mv=100.0,
    )

    target = scored.loc[scored["time"] == start + timedelta(days=2)].iloc[0]
    assert target["baseline_scale_mv"] == 1.0
    assert target["baseline_threshold_mv"] == 35.0
    assert bool(target["contextual_anomaly_candidate"]) is True
