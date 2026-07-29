from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Iterable

MAX_MODULE_COUNT = 4
PACKS_PER_MODULE = 2
VOLTAGE_CHANNELS_PER_PACK = 14
TEMPERATURE_SENSORS_PER_PACK = 4

VOLTAGE_SIGNAL_PATTERN = re.compile(
    r"^bspi2_batterymodules_mod(?P<module>[1-4])_pack(?P<pack>[1-2])_"
    r"cellvoltage_mv_(?P<index>\d+)$"
)
TEMPERATURE_SIGNAL_PATTERN = re.compile(
    r"^bspi2_batterymodules_mod(?P<module>[1-4])_pack(?P<pack>[1-2])_"
    r"temperature(?P<index>\d+)_c$"
)

GLOBAL_RECONCILIATION_SIGNALS = (
    "bspi2_cellvoltagemin_v",
    "bspi2_cellvoltagemax_v",
    "bspi2_celltemperaturemin_c",
    "bspi2_celltemperaturemax_c",
)
OPERATING_CONTEXT_SIGNALS = (
    "bspi2_soc_pct",
    "bspi2_soh_pct",
    "bspi2_current_a",
    "bspi2_voltage_v",
    "bspi2_power_w",
    "bspi2_modulecount",
)


@dataclass(frozen=True)
class HierarchicalSignal:
    source_signal_name: str
    measurement_type: str
    module_index: int
    pack_index: int
    source_channel_index: int
    channel_key: str
    normalized_name: str
    unit: str
    plausible_min: float
    plausible_max: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def voltage_signal(module: int, pack: int, index: int) -> HierarchicalSignal:
    _validate_hierarchy(module, pack)
    if not 0 <= index < VOLTAGE_CHANNELS_PER_PACK:
        raise ValueError("voltage channel index must be between 0 and 13")
    return HierarchicalSignal(
        source_signal_name=(
            f"bspi2_batterymodules_mod{module}_pack{pack}_cellvoltage_mv_{index}"
        ),
        measurement_type="cell_voltage_channel",
        module_index=module,
        pack_index=pack,
        source_channel_index=index,
        channel_key=f"m{module:02d}_p{pack:02d}_cv{index:02d}",
        normalized_name=(
            f"module_{module:02d}_pack_{pack:02d}_"
            f"cell_voltage_channel_{index:02d}_mv"
        ),
        unit="mV",
        plausible_min=2_000.0,
        plausible_max=5_000.0,
    )


def temperature_signal(module: int, pack: int, index: int) -> HierarchicalSignal:
    _validate_hierarchy(module, pack)
    if not 0 <= index < TEMPERATURE_SENSORS_PER_PACK:
        raise ValueError("temperature sensor index must be between 0 and 3")
    return HierarchicalSignal(
        source_signal_name=(
            f"bspi2_batterymodules_mod{module}_pack{pack}_temperature{index}_c"
        ),
        measurement_type="pack_temperature_sensor",
        module_index=module,
        pack_index=pack,
        source_channel_index=index,
        channel_key=f"m{module:02d}_p{pack:02d}_t{index:02d}",
        normalized_name=(
            f"module_{module:02d}_pack_{pack:02d}_temperature_sensor_{index:02d}_c"
        ),
        unit="C",
        plausible_min=-40.0,
        plausible_max=100.0,
    )


def expected_hierarchical_signals(module_count: int) -> list[HierarchicalSignal]:
    if not 1 <= module_count <= MAX_MODULE_COUNT:
        raise ValueError("module_count must be between 1 and 4")
    signals: list[HierarchicalSignal] = []
    for module in range(1, module_count + 1):
        for pack in range(1, PACKS_PER_MODULE + 1):
            signals.extend(
                voltage_signal(module, pack, index)
                for index in range(VOLTAGE_CHANNELS_PER_PACK)
            )
            signals.extend(
                temperature_signal(module, pack, index)
                for index in range(TEMPERATURE_SENSORS_PER_PACK)
            )
    return signals


def parse_hierarchical_signal(name: str) -> HierarchicalSignal | None:
    voltage_match = VOLTAGE_SIGNAL_PATTERN.fullmatch(name)
    if voltage_match:
        return voltage_signal(
            int(voltage_match.group("module")),
            int(voltage_match.group("pack")),
            int(voltage_match.group("index")),
        )
    temperature_match = TEMPERATURE_SIGNAL_PATTERN.fullmatch(name)
    if temperature_match:
        return temperature_signal(
            int(temperature_match.group("module")),
            int(temperature_match.group("pack")),
            int(temperature_match.group("index")),
        )
    return None


def contract_sha256(signals: Iterable[HierarchicalSignal]) -> str:
    payload = [signal.to_dict() for signal in signals]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_hierarchy(module: int, pack: int) -> None:
    if not 1 <= module <= MAX_MODULE_COUNT:
        raise ValueError("module index must be between 1 and 4")
    if not 1 <= pack <= PACKS_PER_MODULE:
        raise ValueError("pack index must be between 1 and 2")
