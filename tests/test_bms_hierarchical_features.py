from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from src.bms_hierarchical_features import feature_column, split_feature_tables
from src.electrothermal_contract import expected_hierarchical_signals


def test_split_features_preserves_hierarchy_and_builds_system_spread() -> None:
    signals = expected_hierarchical_signals(1)
    row: dict[str, object] = {
        "window_start_utc": datetime(2026, 1, 1, tzinfo=UTC),
        "serial": "TEST",
        "first_sample_time": datetime(2026, 1, 1, tzinfo=UTC),
        "last_sample_time": datetime(2026, 1, 1, 0, 4, 59, tzinfo=UTC),
        "sample_count": 300,
        "distinct_timestamp_count": 300,
        "bms_soc_mean_pct": 50.0,
        "bms_soc_min_pct": 49.0,
        "bms_soc_max_pct": 51.0,
        "bms_soh_mean_pct": 98.0,
        "bms_current_mean_a": -2.0,
        "bms_current_min_a": -3.0,
        "bms_current_max_a": -1.0,
        "bms_current_max_abs_a": 3.0,
        "bms_stack_voltage_mean_v": 100.0,
        "bms_power_mean_w": -200.0,
        "bms_module_count_mode": 1,
        "bms_reported_voltage_spread_mean_mv": 20.0,
        "bms_reported_voltage_spread_max_mv": 30.0,
        "bms_reported_temperature_spread_mean_c": 2.0,
        "bms_reported_temperature_spread_max_c": 3.0,
    }
    voltage_offset = 0
    temperature_offset = 0
    for signal in signals:
        if signal.unit == "mV":
            value = 3_500.0 + voltage_offset
            voltage_offset += 1
        else:
            value = 20.0 + temperature_offset / 10
            temperature_offset += 1
        for statistic in ("mean", "min", "max", "std"):
            row[feature_column(signal, statistic)] = value
        row[feature_column(signal, "valid_count")] = 300

    context, voltage, temperature, system = split_feature_tables(
        pd.DataFrame([row]), signals, aggregation_minutes=5
    )

    assert context.iloc[0]["coverage_fraction"] == 1.0
    assert len([column for column in voltage if column.endswith("_mean_mv")]) == 28
    assert len([column for column in temperature if column.endswith("_mean_c")]) == 8
    assert system.iloc[0]["system_voltage_spread_mv"] == 27.0
    assert system.iloc[0]["lowest_voltage_channel_key"] == "m01_p01_cv00"


def test_split_features_keeps_windows_with_all_voltage_channels_missing() -> None:
    signals = expected_hierarchical_signals(1)
    row: dict[str, object] = {
        "window_start_utc": datetime(2026, 1, 1, tzinfo=UTC),
        "serial": "TEST",
        "first_sample_time": datetime(2026, 1, 1, tzinfo=UTC),
        "last_sample_time": datetime(2026, 1, 1, 0, 4, 59, tzinfo=UTC),
        "sample_count": 300,
        "distinct_timestamp_count": 300,
        "bms_soc_mean_pct": 50.0,
        "bms_soc_min_pct": 49.0,
        "bms_soc_max_pct": 51.0,
        "bms_soh_mean_pct": 98.0,
        "bms_current_mean_a": 0.0,
        "bms_current_min_a": 0.0,
        "bms_current_max_a": 0.0,
        "bms_current_max_abs_a": 0.0,
        "bms_stack_voltage_mean_v": 100.0,
        "bms_power_mean_w": 0.0,
        "bms_module_count_mode": 1,
        "bms_reported_voltage_spread_mean_mv": None,
        "bms_reported_voltage_spread_max_mv": None,
        "bms_reported_temperature_spread_mean_c": None,
        "bms_reported_temperature_spread_max_c": None,
    }
    for signal in signals:
        value = None if signal.unit == "mV" else 20.0
        for statistic in ("mean", "min", "max", "std"):
            row[feature_column(signal, statistic)] = value
        row[feature_column(signal, "valid_count")] = 0 if value is None else 300

    _, _, _, system = split_feature_tables(
        pd.DataFrame([row]), signals, aggregation_minutes=5
    )

    assert pd.isna(system.iloc[0]["lowest_voltage_channel_key"])
    assert pd.isna(system.iloc[0]["highest_voltage_channel_key"])
    assert pd.isna(system.iloc[0]["lowest_voltage_mean_mv"])
    assert pd.isna(system.iloc[0]["highest_voltage_mean_mv"])
