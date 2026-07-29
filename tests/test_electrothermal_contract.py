from __future__ import annotations

from src.electrothermal_contract import (
    contract_sha256,
    expected_hierarchical_signals,
    parse_hierarchical_signal,
)


def test_two_module_contract_has_stable_voltage_and_temperature_identity() -> None:
    signals = expected_hierarchical_signals(2)
    voltage = [item for item in signals if item.measurement_type == "cell_voltage_channel"]
    temperature = [
        item for item in signals if item.measurement_type == "pack_temperature_sensor"
    ]

    assert len(voltage) == 56
    assert len(temperature) == 16
    assert voltage[0].source_signal_name == (
        "bspi2_batterymodules_mod1_pack1_cellvoltage_mv_0"
    )
    assert voltage[0].normalized_name == (
        "module_01_pack_01_cell_voltage_channel_00_mv"
    )
    assert voltage[0].channel_key == "m01_p01_cv00"
    assert temperature[-1].normalized_name == (
        "module_02_pack_02_temperature_sensor_03_c"
    )
    assert len(contract_sha256(signals)) == 64


def test_database_signal_parser_preserves_zero_based_source_index() -> None:
    signal = parse_hierarchical_signal(
        "bspi2_batterymodules_mod2_pack1_cellvoltage_mv_13"
    )

    assert signal is not None
    assert signal.module_index == 2
    assert signal.pack_index == 1
    assert signal.source_channel_index == 13
    assert signal.channel_key == "m02_p01_cv13"
