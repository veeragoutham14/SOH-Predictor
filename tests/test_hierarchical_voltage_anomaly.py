from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from src.bms_hierarchical_features import feature_column
from src.electrothermal_contract import expected_hierarchical_signals
from src.hierarchical_voltage_anomaly import (
    build_episodes,
    daily_summary,
    engineering_severity,
    score_hierarchical_windows,
)


def test_engineering_severity_bands_are_fixed_and_auditable() -> None:
    assert engineering_severity(5) == "excellent"
    assert engineering_severity(15) == "acceptable"
    assert engineering_severity(25) == "monitor"
    assert engineering_severity(40) == "anomaly_candidate"
    assert engineering_severity(75) == "engineering_review"
    assert engineering_severity(150) == "serious"
    assert engineering_severity(250) == "severe"


def test_hierarchical_detector_identifies_persistent_affected_channel() -> None:
    signals = expected_hierarchical_signals(1)
    voltage_signals = [item for item in signals if item.measurement_type == "cell_voltage_channel"]
    channel_map = pd.DataFrame(item.to_dict() for item in signals)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    context_rows = []
    voltage_rows = []
    system_rows = []
    for index in range(160):
        timestamp = start + timedelta(minutes=5 * index)
        context_rows.append(
            {
                "window_start_utc": timestamp,
                "serial": "TEST",
                "coverage_fraction": 0.0 if index == 20 else 1.0,
                "bms_current_mean_a": -2.0,
                "bms_soc_mean_pct": 50.0,
            }
        )
        voltage_row = {"window_start_utc": timestamp, "serial": "TEST"}
        for offset, signal in enumerate(voltage_signals):
            value = 3500.0 + offset * 0.05
            if index >= 150 and signal.channel_key == "m01_p01_cv00":
                value -= 45.0
            voltage_row[feature_column(signal, "mean")] = value
        voltage_rows.append(voltage_row)
        spread = 500.0 if index == 20 else 50.0 if index >= 150 else 5.0
        system_rows.append(
            {
                "window_start_utc": timestamp,
                "serial": "TEST",
                "system_voltage_spread_mv": spread,
                "bms_reported_voltage_spread_max_mv": 18960.0 if index == 10 else spread,
            }
        )
    scored, _ = score_hierarchical_windows(
        pd.DataFrame(context_rows),
        pd.DataFrame(voltage_rows),
        pd.DataFrame(system_rows),
        channel_map,
        baseline_days=1,
        low_rank_components=2,
    )
    scored, episodes = build_episodes(
        scored,
        aggregation_minutes=5,
        minimum_episode_bins=3,
        episode_gap_minutes=15,
    )

    assert len(episodes) == 1
    assert episodes.iloc[0]["affected_channel_key"] == "m01_p01_cv00"
    assert episodes.iloc[0]["severity"] == "engineering_review"
    assert scored["episode_id"].notna().sum() == 10
    assert scored["analysis_voltage_spread_mv"].max() == 500.0
    assert scored.loc[10, "analysis_voltage_spread_mv"] == 5.0
    assert scored.loc[10, "bms_reported_voltage_spread_plausible"] == False  # noqa: E712
    assert scored.loc[20, "engineering_severity"] == "unavailable"
    assert scored.loc[20, "point_anomaly_candidate"] == False  # noqa: E712

    daily = daily_summary(scored, aggregation_minutes=5)
    assert daily.iloc[0]["system_spread_max_mv"] == 50.0
    assert daily.iloc[0]["quality_invalid_windows"] == 1
    assert daily.iloc[0]["bms_global_spread_implausible_windows"] == 1
