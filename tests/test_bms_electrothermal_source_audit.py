from __future__ import annotations

from src.bms_electrothermal_source_audit import SourceSignalQuality, quality_decision


def _record(measurement_type: str, *, ready: bool = True, frozen: bool = False) -> SourceSignalQuality:
    return SourceSignalQuality(
        source_signal_name="source",
        normalized_name="normalized",
        measurement_type=measurement_type,
        module_index=1,
        pack_index=1,
        source_channel_index=0,
        channel_key="key",
        unit="mV" if measurement_type == "cell_voltage_channel" else "C",
        present_in_schema=True,
        non_null_rows=100,
        valid_rows=100,
        distinct_values=10,
        non_null_coverage_pct=100.0,
        valid_coverage_pct=100.0,
        observed_min=1.0,
        observed_max=2.0,
        frozen_signal=frozen,
        ready=ready,
    )


def test_quality_gate_requires_ready_reconciled_hierarchical_signals() -> None:
    ready, reasons, status = quality_decision(
        [_record("cell_voltage_channel"), _record("pack_temperature_sensor")],
        {
            "voltage_min_p95_error_mv": 2.0,
            "voltage_max_p95_error_mv": 3.0,
            "temperature_min_p95_error_c": 0.2,
            "temperature_max_p95_error_c": 0.3,
        },
        duplicate_timestamp_pct=0.0,
    )

    assert ready is True
    assert reasons == []
    assert status == "ready"


def test_quality_gate_blocks_frozen_or_unreconciled_voltage() -> None:
    ready, reasons, status = quality_decision(
        [
            _record("cell_voltage_channel", frozen=True),
            _record("pack_temperature_sensor"),
        ],
        {
            "voltage_min_p95_error_mv": 20.0,
            "voltage_max_p95_error_mv": 20.0,
            "temperature_min_p95_error_c": 0.2,
            "temperature_max_p95_error_c": 0.3,
        },
        duplicate_timestamp_pct=0.0,
    )

    assert ready is False
    assert status == "blocked"
    assert "one_or_more_voltage_channels_frozen" in reasons
    assert "individual_voltage_channels_do_not_reconcile_with_bms_extrema" in reasons


def test_quality_gate_accepts_exact_zero_reconciliation_error() -> None:
    ready, reasons, status = quality_decision(
        [_record("cell_voltage_channel"), _record("pack_temperature_sensor")],
        {
            "voltage_min_p95_error_mv": 0.0,
            "voltage_max_p95_error_mv": 0.0,
            "temperature_min_p95_error_c": 0.0,
            "temperature_max_p95_error_c": 0.0,
        },
        duplicate_timestamp_pct=0.0,
    )

    assert ready is True
    assert reasons == []
    assert status == "ready"
