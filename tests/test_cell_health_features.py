from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from src.cell_health_features import (
    add_module_configuration_epochs,
    module_configuration_epochs,
    summarize_features,
)


def _feature_frame() -> pd.DataFrame:
    start = datetime(2026, 4, 24, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for day_offset, module_count in enumerate((2, 2, 2, 3, 3, 3)):
        for bin_offset in range(3):
            timestamp = start + timedelta(days=day_offset, minutes=5 * bin_offset)
            observed = 226.15 if day_offset == 1 and bin_offset == 0 else module_count
            rows.append(
                {
                    "time": timestamp,
                    "first_sample_time": timestamp,
                    "last_sample_time": timestamp + timedelta(minutes=4, seconds=59),
                    "sample_count": 250,
                    "valid_cell_voltage_sample_count": 249,
                    "module_count_bucket_mode": observed,
                    "valid_module_count_sample_count": 249,
                    "invalid_module_count_sample_count": int(observed > 4),
                    "cell_spread_max_mv": 20.0,
                    "compensated_cell_spread_max_mv": 18.0,
                    "cell_temperature_spread_max_c": 3.0,
                }
            )
    return pd.DataFrame(rows)


def test_module_count_cleaning_labels_stable_configuration_epochs() -> None:
    cleaned = add_module_configuration_epochs(_feature_frame())

    assert set(cleaned["module_count"].dropna().astype(int)) == {2, 3}
    assert cleaned["module_configuration_epoch"].nunique() == 2
    assert pd.isna(cleaned.loc[3, "module_count_bucket_mode"])

    epochs = module_configuration_epochs(cleaned)
    assert [item["module_count"] for item in epochs] == [2, 3]
    assert [item["feature_rows"] for item in epochs] == [9, 9]


def test_feature_summary_reports_latest_and_historical_module_counts() -> None:
    cleaned = add_module_configuration_epochs(_feature_frame())
    summary = summarize_features(
        cleaned,
        serial="300000093",
        start=datetime(2026, 4, 24, tzinfo=UTC),
        end=datetime(2026, 4, 30, tzinfo=UTC),
        aggregation_minutes=5,
    )

    assert summary["latest_module_count"] == 3
    assert summary["observed_module_count"] == 3
    assert summary["dominant_module_count"] == 2
    assert summary["module_configuration_changed"] is True
    assert len(summary["module_configuration_epochs"]) == 2
    assert summary["invalid_module_count_sample_rows"] == 1
