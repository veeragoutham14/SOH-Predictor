from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from psycopg2 import sql

from src.cell_signal_audit import (
    _available_columns,
    _observed_module_count,
    _qualified_identifier,
    _representative_recent_day,
)
from src.config import DBConfig, StorageConfig
from src.db import DEFAULT_SERIAL_COL, DEFAULT_TABLE, DEFAULT_TIMESTAMP_COL, open_db_connection
from src.electrothermal_contract import (
    GLOBAL_RECONCILIATION_SIGNALS,
    OPERATING_CONTEXT_SIGNALS,
    HierarchicalSignal,
    contract_sha256,
    expected_hierarchical_signals,
)
from src.io_utils import parse_timestamp, write_parquet_chunk
from src.logging_utils import configure_logging

logger = logging.getLogger(__name__)

VOLTAGE_RECONCILIATION_TOLERANCE_MV = 15.0
TEMPERATURE_RECONCILIATION_TOLERANCE_C = 2.0


@dataclass(frozen=True)
class SourceSignalQuality:
    source_signal_name: str
    normalized_name: str
    measurement_type: str
    module_index: int | None
    pack_index: int | None
    source_channel_index: int | None
    channel_key: str | None
    unit: str
    present_in_schema: bool
    non_null_rows: int
    valid_rows: int
    distinct_values: int
    non_null_coverage_pct: float
    valid_coverage_pct: float
    observed_min: float | None
    observed_max: float | None
    frozen_signal: bool
    ready: bool


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _schema_manifest(connection: Any, table: str) -> tuple[list[dict[str, str]], str]:
    schema_name, table_name = table.split(".", 1) if "." in table else ("public", table)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
            """,
            (schema_name, table_name),
        )
        columns = [
            {"name": str(name), "data_type": str(data_type), "nullable": str(nullable)}
            for name, data_type, nullable in cursor.fetchall()
        ]
    encoded = json.dumps(columns, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return columns, hashlib.sha256(encoded).hexdigest()


def _audit_query(
    signals: Sequence[HierarchicalSignal],
    present_names: set[str],
    *,
    table: str,
    timestamp_col: str,
    serial_col: str,
) -> tuple[sql.Composed, list[HierarchicalSignal]]:
    query_signals = [item for item in signals if item.source_signal_name in present_names]
    aggregates: list[sql.Composed] = [
        sql.SQL("COUNT(*) AS total_rows"),
        sql.SQL("COUNT(DISTINCT {timestamp}) AS distinct_timestamps").format(
            timestamp=sql.Identifier(timestamp_col)
        ),
        sql.SQL("MIN({timestamp}) AS first_timestamp").format(
            timestamp=sql.Identifier(timestamp_col)
        ),
        sql.SQL("MAX({timestamp}) AS last_timestamp").format(
            timestamp=sql.Identifier(timestamp_col)
        ),
    ]
    for index, item in enumerate(query_signals):
        signal = sql.Identifier(item.source_signal_name)
        lower = sql.Literal(item.plausible_min)
        upper = sql.Literal(item.plausible_max)
        aggregates.extend(
            (
                sql.SQL("COUNT({signal}) AS {alias}").format(
                    signal=signal, alias=sql.Identifier(f"non_null_{index}")
                ),
                sql.SQL(
                    "COUNT(*) FILTER (WHERE {signal} BETWEEN {lower} AND {upper}) "
                    "AS {alias}"
                ).format(
                    signal=signal,
                    lower=lower,
                    upper=upper,
                    alias=sql.Identifier(f"valid_{index}"),
                ),
                sql.SQL("COUNT(DISTINCT {signal}) AS {alias}").format(
                    signal=signal, alias=sql.Identifier(f"distinct_{index}")
                ),
                sql.SQL("MIN({signal}) AS {alias}").format(
                    signal=signal, alias=sql.Identifier(f"min_{index}")
                ),
                sql.SQL("MAX({signal}) AS {alias}").format(
                    signal=signal, alias=sql.Identifier(f"max_{index}")
                ),
            )
        )
    query = sql.SQL(
        """
        SELECT {aggregates}
        FROM {table}
        WHERE {serial_col} = %s
          AND {timestamp} >= %s
          AND {timestamp} < %s
        """
    ).format(
        aggregates=sql.SQL(", ").join(aggregates),
        table=_qualified_identifier(table),
        serial_col=sql.Identifier(serial_col),
        timestamp=sql.Identifier(timestamp_col),
    )
    return query, query_signals


def _reconciliation_query(
    hierarchical: Sequence[HierarchicalSignal],
    available: set[str],
    *,
    table: str,
    timestamp_col: str,
    serial_col: str,
) -> sql.Composed | None:
    voltage = [
        item.source_signal_name
        for item in hierarchical
        if item.measurement_type == "cell_voltage_channel"
        and item.source_signal_name in available
    ]
    temperature = [
        item.source_signal_name
        for item in hierarchical
        if item.measurement_type == "pack_temperature_sensor"
        and item.source_signal_name in available
    ]
    required = set(GLOBAL_RECONCILIATION_SIGNALS)
    if len(voltage) == 0 or len(temperature) == 0 or not required.issubset(available):
        return None
    voltage_min = sql.SQL("LEAST({})").format(
        sql.SQL(", ").join(sql.Identifier(name) for name in voltage)
    )
    voltage_max = sql.SQL("GREATEST({})").format(
        sql.SQL(", ").join(sql.Identifier(name) for name in voltage)
    )
    temperature_min = sql.SQL("LEAST({})").format(
        sql.SQL(", ").join(sql.Identifier(name) for name in temperature)
    )
    temperature_max = sql.SQL("GREATEST({})").format(
        sql.SQL(", ").join(sql.Identifier(name) for name in temperature)
    )
    complete_voltage = sql.SQL(" + ").join(
        sql.SQL("CASE WHEN {name} IS NOT NULL THEN 1 ELSE 0 END").format(
            name=sql.Identifier(name)
        )
        for name in voltage
    )
    complete_temperature = sql.SQL(" + ").join(
        sql.SQL("CASE WHEN {name} IS NOT NULL THEN 1 ELSE 0 END").format(
            name=sql.Identifier(name)
        )
        for name in temperature
    )
    return sql.SQL(
        """
        WITH compared AS (
          SELECT
            CASE WHEN ({complete_voltage}) = {voltage_count}
              THEN ABS(({voltage_min}) - bspi2_cellvoltagemin_v * 1000.0) END
              AS voltage_min_error_mv,
            CASE WHEN ({complete_voltage}) = {voltage_count}
              THEN ABS(({voltage_max}) - bspi2_cellvoltagemax_v * 1000.0) END
              AS voltage_max_error_mv,
            CASE WHEN ({complete_temperature}) = {temperature_count}
              THEN ABS(({temperature_min}) - bspi2_celltemperaturemin_c) END
              AS temperature_min_error_c,
            CASE WHEN ({complete_temperature}) = {temperature_count}
              THEN ABS(({temperature_max}) - bspi2_celltemperaturemax_c) END
              AS temperature_max_error_c
          FROM {table}
          WHERE {serial_col} = %s
            AND {timestamp} >= %s
            AND {timestamp} < %s
        )
        SELECT
          COUNT(voltage_min_error_mv) AS voltage_compared_rows,
          AVG(voltage_min_error_mv) AS voltage_min_mae_mv,
          AVG(voltage_max_error_mv) AS voltage_max_mae_mv,
          PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY voltage_min_error_mv)
            AS voltage_min_p95_error_mv,
          PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY voltage_max_error_mv)
            AS voltage_max_p95_error_mv,
          COUNT(temperature_min_error_c) AS temperature_compared_rows,
          AVG(temperature_min_error_c) AS temperature_min_mae_c,
          AVG(temperature_max_error_c) AS temperature_max_mae_c,
          PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY temperature_min_error_c)
            AS temperature_min_p95_error_c,
          PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY temperature_max_error_c)
            AS temperature_max_p95_error_c
        FROM compared
        """
    ).format(
        complete_voltage=complete_voltage,
        voltage_count=sql.Literal(len(voltage)),
        voltage_min=voltage_min,
        voltage_max=voltage_max,
        complete_temperature=complete_temperature,
        temperature_count=sql.Literal(len(temperature)),
        temperature_min=temperature_min,
        temperature_max=temperature_max,
        table=_qualified_identifier(table),
        serial_col=sql.Identifier(serial_col),
        timestamp=sql.Identifier(timestamp_col),
    )


def quality_decision(
    records: Sequence[SourceSignalQuality],
    reconciliation: dict[str, Any],
    *,
    duplicate_timestamp_pct: float,
) -> tuple[bool, list[str], str]:
    def reconciliation_value(key: str) -> float:
        value = reconciliation.get(key)
        return float(value) if value is not None else float("inf")

    reasons: list[str] = []
    voltage = [item for item in records if item.measurement_type == "cell_voltage_channel"]
    temperature = [
        item for item in records if item.measurement_type == "pack_temperature_sensor"
    ]
    if not voltage or not all(item.ready for item in voltage):
        reasons.append("one_or_more_voltage_channels_not_ready")
    if not temperature or not all(item.ready for item in temperature):
        reasons.append("one_or_more_temperature_sensors_not_ready")
    if any(item.frozen_signal for item in voltage):
        reasons.append("one_or_more_voltage_channels_frozen")
    if duplicate_timestamp_pct > 0.1:
        reasons.append("duplicate_timestamp_rate_above_0_1_pct")
    voltage_error = max(
        reconciliation_value("voltage_min_p95_error_mv"),
        reconciliation_value("voltage_max_p95_error_mv"),
    )
    temperature_error = max(
        reconciliation_value("temperature_min_p95_error_c"),
        reconciliation_value("temperature_max_p95_error_c"),
    )
    if voltage_error > VOLTAGE_RECONCILIATION_TOLERANCE_MV:
        reasons.append("individual_voltage_channels_do_not_reconcile_with_bms_extrema")
    if temperature_error > TEMPERATURE_RECONCILIATION_TOLERANCE_C:
        reasons.append("temperature_sensors_do_not_reconcile_with_bms_extrema")
    ready = not reasons
    return ready, reasons, "ready" if ready else "blocked"


def audit_source(
    connection: Any,
    *,
    serial: str,
    start: datetime | None,
    end: datetime | None,
    module_count: int | None,
    readiness_threshold: float,
    table: str,
    timestamp_col: str,
    serial_col: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if (start is None) != (end is None):
        raise ValueError("start and end must be provided together")
    selected_automatically = start is None
    if start is None or end is None:
        start, end = _representative_recent_day(
            connection,
            serial=serial,
            table=table,
            timestamp_col=timestamp_col,
            serial_col=serial_col,
        )
    if start >= end:
        raise ValueError("start must be earlier than end")
    available = _available_columns(connection, table)
    observed_modules = module_count or _observed_module_count(
        connection,
        serial=serial,
        start=start,
        end=end,
        table=table,
        timestamp_col=timestamp_col,
        serial_col=serial_col,
        module_count_col="bspi2_modulecount",
    )
    hierarchical = expected_hierarchical_signals(observed_modules)
    query, queried = _audit_query(
        hierarchical,
        available,
        table=table,
        timestamp_col=timestamp_col,
        serial_col=serial_col,
    )
    with connection.cursor() as cursor:
        cursor.execute(query, (serial, start, end))
        values = dict(zip((item[0] for item in cursor.description), cursor.fetchone(), strict=True))
    total_rows = int(values["total_rows"] or 0)
    if total_rows == 0:
        raise ValueError(f"No rows found for serial {serial} in {start} to {end}")
    query_index = {item.source_signal_name: index for index, item in enumerate(queried)}
    records: list[SourceSignalQuality] = []
    for item in hierarchical:
        index = query_index.get(item.source_signal_name)
        present = index is not None
        non_null = int(values[f"non_null_{index}"] or 0) if present else 0
        valid = int(values[f"valid_{index}"] or 0) if present else 0
        distinct = int(values[f"distinct_{index}"] or 0) if present else 0
        non_null_pct = 100.0 * non_null / total_rows
        valid_pct = 100.0 * valid / total_rows
        frozen = present and non_null > 1 and distinct <= 1
        records.append(
            SourceSignalQuality(
                source_signal_name=item.source_signal_name,
                normalized_name=item.normalized_name,
                measurement_type=item.measurement_type,
                module_index=item.module_index,
                pack_index=item.pack_index,
                source_channel_index=item.source_channel_index,
                channel_key=item.channel_key,
                unit=item.unit,
                present_in_schema=present,
                non_null_rows=non_null,
                valid_rows=valid,
                distinct_values=distinct,
                non_null_coverage_pct=non_null_pct,
                valid_coverage_pct=valid_pct,
                observed_min=(float(values[f"min_{index}"]) if present and values[f"min_{index}"] is not None else None),
                observed_max=(float(values[f"max_{index}"]) if present and values[f"max_{index}"] is not None else None),
                frozen_signal=frozen,
                ready=present and valid / total_rows >= readiness_threshold and not frozen,
            )
        )
    reconciliation_query = _reconciliation_query(
        hierarchical,
        available,
        table=table,
        timestamp_col=timestamp_col,
        serial_col=serial_col,
    )
    reconciliation: dict[str, Any] = {}
    if reconciliation_query is not None:
        with connection.cursor() as cursor:
            cursor.execute(reconciliation_query, (serial, start, end))
            reconciliation = dict(
                zip((item[0] for item in cursor.description), cursor.fetchone(), strict=True)
            )
    distinct_timestamps = int(values["distinct_timestamps"] or 0)
    duplicate_pct = 100.0 * max(0, total_rows - distinct_timestamps) / total_rows
    ready, blocking_reasons, status = quality_decision(
        records,
        reconciliation,
        duplicate_timestamp_pct=duplicate_pct,
    )
    schema_columns, schema_hash = _schema_manifest(connection, table)
    context_present = sorted(set(OPERATING_CONTEXT_SIGNALS) & available)
    voltage_records = [
        item for item in records if item.measurement_type == "cell_voltage_channel"
    ]
    temperature_records = [
        item for item in records if item.measurement_type == "pack_temperature_sensor"
    ]
    observed_seconds = max(1.0, (end - start).total_seconds())
    voltage_reconciled = (
        reconciliation.get("voltage_min_p95_error_mv") is not None
        and reconciliation.get("voltage_max_p95_error_mv") is not None
        and max(
            float(reconciliation["voltage_min_p95_error_mv"]),
            float(reconciliation["voltage_max_p95_error_mv"]),
        ) <= VOLTAGE_RECONCILIATION_TOLERANCE_MV
    )
    temperature_reconciled = (
        reconciliation.get("temperature_min_p95_error_c") is not None
        and reconciliation.get("temperature_max_p95_error_c") is not None
        and max(
            float(reconciliation["temperature_min_p95_error_c"]),
            float(reconciliation["temperature_max_p95_error_c"]),
        ) <= TEMPERATURE_RECONCILIATION_TOLERANCE_C
    )
    summary = {
        "model_family": "bms_electrothermal_source_quality",
        "serial": serial,
        "sample_start": start.isoformat(),
        "sample_end": end.isoformat(),
        "sample_selected_automatically": selected_automatically,
        "module_count": observed_modules,
        "expected_voltage_channels": sum(
            item.measurement_type == "cell_voltage_channel" for item in records
        ),
        "ready_voltage_channels": sum(
            item.measurement_type == "cell_voltage_channel" and item.ready for item in records
        ),
        "expected_temperature_sensors": sum(
            item.measurement_type == "pack_temperature_sensor" for item in records
        ),
        "ready_temperature_sensors": sum(
            item.measurement_type == "pack_temperature_sensor" and item.ready for item in records
        ),
        "readiness_threshold_pct": readiness_threshold * 100.0,
        "total_rows": total_rows,
        "distinct_timestamps": distinct_timestamps,
        "duplicate_timestamp_pct": duplicate_pct,
        "nominal_1hz_coverage_pct": min(
            100.0, 100.0 * distinct_timestamps / observed_seconds
        ),
        "minimum_voltage_channel_valid_coverage_pct": min(
            (item.valid_coverage_pct for item in voltage_records), default=0.0
        ),
        "minimum_temperature_sensor_valid_coverage_pct": min(
            (item.valid_coverage_pct for item in temperature_records), default=0.0
        ),
        "voltage_extrema_reconciled": voltage_reconciled,
        "temperature_extrema_reconciled": temperature_reconciled,
        "first_timestamp": str(values["first_timestamp"]),
        "last_timestamp": str(values["last_timestamp"]),
        "context_signals_present": context_present,
        "source_quality_ready": ready,
        "quality_gate_status": status,
        "blocking_reasons": blocking_reasons,
        "reconciliation": {
            key: float(value) if value is not None else None
            for key, value in reconciliation.items()
        },
        "provenance": {
            "source_table": table,
            "timestamp_column": timestamp_col,
            "serial_column": serial_col,
            "database_schema_sha256": schema_hash,
            "database_schema_column_count": len(schema_columns),
            "hierarchical_contract_sha256": contract_sha256(hierarchical),
            "source_code_git_commit": _git_commit(),
            "timestamp_semantics": "UTC",
            "value_semantics": "BMS-reported telemetry; physical calibration not independently verified",
        },
    }
    reconciliation_frame = pd.DataFrame([summary["reconciliation"]])
    return summary, pd.DataFrame(asdict(item) for item in records), reconciliation_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit hierarchical BMS electrothermal signals.")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--module-count", type=int)
    parser.add_argument("--start", type=parse_timestamp)
    parser.add_argument("--end", type=parse_timestamp)
    parser.add_argument("--readiness-threshold", type=float, default=0.95)
    parser.add_argument("--column-catalog", type=Path)
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
        db_config = DBConfig.from_env(args.env_file)
        storage = StorageConfig.from_env(args.env_file)
        output_root = args.output_root or storage.data_dir / "processed" / "bms_electrothermal_source_quality"
        with open_db_connection(db_config) as connection:
            summary, signals, reconciliation = audit_source(
                connection,
                serial=str(args.serial),
                start=args.start,
                end=args.end,
                module_count=args.module_count,
                readiness_threshold=args.readiness_threshold,
                table=args.table,
                timestamp_col=args.timestamp_col,
                serial_col=args.serial_col,
            )
        run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = output_root / f"serial={args.serial}" / f"source_audit_run={run_stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        compression = None if args.compression == "none" else args.compression
        write_parquet_chunk(signals, run_dir / "signal_availability.parquet", compression=compression)
        write_parquet_chunk(reconciliation, run_dir / "signal_reconciliation.parquet", compression=compression)
        manifest = {
            "manifest_version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "serial": str(args.serial),
            "column_catalog_sha256": _sha256_file(args.column_catalog),
            "source_quality_ready": summary["source_quality_ready"],
            "provenance": summary["provenance"],
        }
        (run_dir / "model_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (run_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        logger.info("Electrothermal source audit complete for serial=%s: %s", args.serial, run_dir)
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Electrothermal source audit failed")
        else:
            logger.error("Electrothermal source audit failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
