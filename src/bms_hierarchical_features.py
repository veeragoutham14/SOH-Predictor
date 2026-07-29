from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from psycopg2 import sql

from src.cell_signal_audit import _available_columns, _qualified_identifier
from src.config import DBConfig, StorageConfig
from src.db import DEFAULT_SERIAL_COL, DEFAULT_TABLE, DEFAULT_TIMESTAMP_COL, open_db_connection
from src.electrothermal_contract import (
    GLOBAL_RECONCILIATION_SIGNALS,
    OPERATING_CONTEXT_SIGNALS,
    HierarchicalSignal,
    contract_sha256,
    expected_hierarchical_signals,
)
from src.io_utils import iter_month_ranges, normalize_timestamp_columns, parse_timestamp, write_parquet_chunk
from src.logging_utils import configure_logging

logger = logging.getLogger(__name__)

FEATURE_VERSION = "v1"
FEATURE_STATISTICS = ("mean", "min", "max", "std", "valid_count")


def feature_column(signal: HierarchicalSignal, statistic: str) -> str:
    if statistic not in FEATURE_STATISTICS:
        raise ValueError(f"Unsupported feature statistic: {statistic}")
    suffix = "mv" if signal.unit == "mV" else "c"
    base = signal.normalized_name.removesuffix(f"_{suffix}")
    return f"{base}_{statistic}_{suffix}" if statistic != "valid_count" else f"{base}_valid_count"


def _signal_aggregates(signal: HierarchicalSignal) -> list[sql.Composed]:
    source = sql.Identifier(signal.source_signal_name)
    valid = sql.SQL("CASE WHEN {source} BETWEEN {lower} AND {upper} THEN {source} END").format(
        source=source,
        lower=sql.Literal(signal.plausible_min),
        upper=sql.Literal(signal.plausible_max),
    )
    return [
        sql.SQL("AVG({valid}) AS {alias}").format(
            valid=valid, alias=sql.Identifier(feature_column(signal, "mean"))
        ),
        sql.SQL("MIN({valid}) AS {alias}").format(
            valid=valid, alias=sql.Identifier(feature_column(signal, "min"))
        ),
        sql.SQL("MAX({valid}) AS {alias}").format(
            valid=valid, alias=sql.Identifier(feature_column(signal, "max"))
        ),
        sql.SQL("STDDEV_POP({valid}) AS {alias}").format(
            valid=valid, alias=sql.Identifier(feature_column(signal, "std"))
        ),
        sql.SQL("COUNT({valid}) AS {alias}").format(
            valid=valid, alias=sql.Identifier(feature_column(signal, "valid_count"))
        ),
    ]


def build_feature_query(
    signals: Sequence[HierarchicalSignal],
    *,
    table: str = DEFAULT_TABLE,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
    serial_col: str = DEFAULT_SERIAL_COL,
) -> sql.Composed:
    aggregates: list[sql.Composed] = [
        sql.SQL("MIN({timestamp}) AS first_sample_time").format(timestamp=sql.Identifier(timestamp_col)),
        sql.SQL("MAX({timestamp}) AS last_sample_time").format(timestamp=sql.Identifier(timestamp_col)),
        sql.SQL("COUNT(*) AS sample_count"),
        sql.SQL("COUNT(DISTINCT {timestamp}) AS distinct_timestamp_count").format(
            timestamp=sql.Identifier(timestamp_col)
        ),
        sql.SQL("AVG(bspi2_soc_pct) AS bms_soc_mean_pct"),
        sql.SQL("MIN(bspi2_soc_pct) AS bms_soc_min_pct"),
        sql.SQL("MAX(bspi2_soc_pct) AS bms_soc_max_pct"),
        sql.SQL("AVG(bspi2_soh_pct) AS bms_soh_mean_pct"),
        sql.SQL("AVG(bspi2_current_a) AS bms_current_mean_a"),
        sql.SQL("MIN(bspi2_current_a) AS bms_current_min_a"),
        sql.SQL("MAX(bspi2_current_a) AS bms_current_max_a"),
        sql.SQL("MAX(ABS(bspi2_current_a)) AS bms_current_max_abs_a"),
        sql.SQL("AVG(bspi2_voltage_v) AS bms_stack_voltage_mean_v"),
        sql.SQL("AVG(bspi2_power_w) AS bms_power_mean_w"),
        sql.SQL("MODE() WITHIN GROUP (ORDER BY bspi2_modulecount) AS bms_module_count_mode"),
        sql.SQL("AVG((bspi2_cellvoltagemax_v - bspi2_cellvoltagemin_v) * 1000.0) AS bms_reported_voltage_spread_mean_mv"),
        sql.SQL("MAX((bspi2_cellvoltagemax_v - bspi2_cellvoltagemin_v) * 1000.0) AS bms_reported_voltage_spread_max_mv"),
        sql.SQL("AVG(bspi2_celltemperaturemax_c - bspi2_celltemperaturemin_c) AS bms_reported_temperature_spread_mean_c"),
        sql.SQL("MAX(bspi2_celltemperaturemax_c - bspi2_celltemperaturemin_c) AS bms_reported_temperature_spread_max_c"),
    ]
    for signal in signals:
        aggregates.extend(_signal_aggregates(signal))
    return sql.SQL(
        """
        SELECT
          date_bin(%s::interval, {timestamp}, TIMESTAMPTZ '2000-01-01 00:00:00+00') AS window_start_utc,
          {serial_col} AS serial,
          {aggregates}
        FROM {table}
        WHERE {serial_col} = %s
          AND {timestamp} >= %s
          AND {timestamp} < %s
        GROUP BY window_start_utc, {serial_col}
        ORDER BY window_start_utc
        """
    ).format(
        timestamp=sql.Identifier(timestamp_col),
        serial_col=sql.Identifier(serial_col),
        aggregates=sql.SQL(", ").join(aggregates),
        table=_qualified_identifier(table),
    )


def split_feature_tables(
    frame: pd.DataFrame,
    signals: Sequence[HierarchicalSignal],
    *,
    aggregation_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["window_start_utc", "serial", "first_sample_time", "last_sample_time"]
    voltage = [item for item in signals if item.measurement_type == "cell_voltage_channel"]
    temperature = [item for item in signals if item.measurement_type == "pack_temperature_sensor"]
    voltage_columns = [feature_column(item, stat) for item in voltage for stat in FEATURE_STATISTICS]
    temperature_columns = [feature_column(item, stat) for item in temperature for stat in FEATURE_STATISTICS]
    context_columns = [column for column in frame.columns if column not in set(voltage_columns + temperature_columns)]
    context = frame[context_columns].copy()
    context["expected_sample_count"] = aggregation_minutes * 60
    context["coverage_fraction"] = (
        pd.to_numeric(context["distinct_timestamp_count"], errors="coerce")
        / context["expected_sample_count"]
    ).clip(upper=1.0)
    voltage_frame = frame[[*keys, *voltage_columns]].copy()
    temperature_frame = frame[[*keys, *temperature_columns]].copy()
    system = frame[keys].copy()

    voltage_mean_columns: list[str] = []
    temperature_mean_columns: list[str] = []
    for module in sorted({item.module_index for item in signals}):
        module_voltage: list[str] = []
        module_temperature: list[str] = []
        for pack in (1, 2):
            pack_voltage = [
                feature_column(item, "mean")
                for item in voltage
                if item.module_index == module and item.pack_index == pack
            ]
            pack_temperature = [
                feature_column(item, "mean")
                for item in temperature
                if item.module_index == module and item.pack_index == pack
            ]
            module_voltage.extend(pack_voltage)
            module_temperature.extend(pack_temperature)
            system[f"module_{module:02d}_pack_{pack:02d}_voltage_spread_mv"] = (
                frame[pack_voltage].max(axis=1) - frame[pack_voltage].min(axis=1)
            )
            system[f"module_{module:02d}_pack_{pack:02d}_temperature_spread_c"] = (
                frame[pack_temperature].max(axis=1) - frame[pack_temperature].min(axis=1)
            )
        voltage_mean_columns.extend(module_voltage)
        temperature_mean_columns.extend(module_temperature)
        system[f"module_{module:02d}_voltage_spread_mv"] = (
            frame[module_voltage].max(axis=1) - frame[module_voltage].min(axis=1)
        )
        system[f"module_{module:02d}_temperature_spread_c"] = (
            frame[module_temperature].max(axis=1) - frame[module_temperature].min(axis=1)
        )
    system["system_voltage_spread_mv"] = (
        frame[voltage_mean_columns].max(axis=1) - frame[voltage_mean_columns].min(axis=1)
    )
    system["system_temperature_spread_c"] = (
        frame[temperature_mean_columns].max(axis=1) - frame[temperature_mean_columns].min(axis=1)
    )
    system["system_temperature_mean_c"] = frame[temperature_mean_columns].mean(axis=1)
    system["system_temperature_min_c"] = frame[temperature_mean_columns].min(axis=1)
    system["system_temperature_max_c"] = frame[temperature_mean_columns].max(axis=1)
    voltage_values = frame[voltage_mean_columns].apply(pd.to_numeric, errors="coerce")
    voltage_matrix = voltage_values.to_numpy(dtype=float, na_value=np.nan)
    valid_voltage = np.isfinite(voltage_matrix)
    has_valid_voltage = valid_voltage.any(axis=1)
    channel_keys = np.asarray([item.channel_key for item in voltage], dtype=object)

    minimum_matrix = np.where(valid_voltage, voltage_matrix, np.inf)
    maximum_matrix = np.where(valid_voltage, voltage_matrix, -np.inf)
    minimum_indices = minimum_matrix.argmin(axis=1)
    maximum_indices = maximum_matrix.argmax(axis=1)
    lowest_channel_keys = np.full(len(frame), None, dtype=object)
    highest_channel_keys = np.full(len(frame), None, dtype=object)
    lowest_channel_keys[has_valid_voltage] = channel_keys[
        minimum_indices[has_valid_voltage]
    ]
    highest_channel_keys[has_valid_voltage] = channel_keys[
        maximum_indices[has_valid_voltage]
    ]
    minimum_values = minimum_matrix.min(axis=1)
    maximum_values = maximum_matrix.max(axis=1)
    minimum_values[~has_valid_voltage] = np.nan
    maximum_values[~has_valid_voltage] = np.nan

    system["lowest_voltage_channel_key"] = lowest_channel_keys
    system["highest_voltage_channel_key"] = highest_channel_keys
    system["lowest_voltage_mean_mv"] = minimum_values
    system["highest_voltage_mean_mv"] = maximum_values
    for column in (
        "bms_reported_voltage_spread_mean_mv",
        "bms_reported_voltage_spread_max_mv",
        "bms_reported_temperature_spread_mean_c",
        "bms_reported_temperature_spread_max_c",
    ):
        system[column] = frame[column]
    return context, voltage_frame, temperature_frame, system


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _latest_ready_audit(output_root: Path, serial: str, module_count: int) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(
        (output_root / f"serial={serial}").glob("source_audit_run=*/model_summary.json")
    )
    if not candidates:
        raise ValueError(f"No electrothermal source audit exists for serial {serial}")
    path = candidates[-1]
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not summary.get("source_quality_ready"):
        reasons = ", ".join(summary.get("blocking_reasons") or ["unknown source-quality failure"])
        raise ValueError(f"Latest electrothermal source audit is blocked: {reasons}")
    if int(summary.get("module_count") or 0) != module_count:
        raise ValueError("Source-audit module count does not match requested feature module count")
    return path.parent, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build hierarchical five-minute BMS electrothermal features.")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--module-count", required=True, type=int)
    parser.add_argument("--start", required=True, type=parse_timestamp)
    parser.add_argument("--end", required=True, type=parse_timestamp)
    parser.add_argument("--aggregation-minutes", type=int, default=5, choices=(1, 5, 15))
    parser.add_argument("--source-audit-root", type=Path)
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
        if args.start >= args.end:
            raise ValueError("start must be earlier than end")
        signals = expected_hierarchical_signals(args.module_count)
        db_config = DBConfig.from_env(args.env_file)
        storage = StorageConfig.from_env(args.env_file)
        audit_root = args.source_audit_root or storage.data_dir / "processed" / "bms_electrothermal_source_quality"
        audit_run, audit_summary = _latest_ready_audit(audit_root, str(args.serial), args.module_count)
        output_root = args.output_root or storage.data_dir / "processed" / "bms_hierarchical_features"
        run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = output_root / f"serial={args.serial}" / f"feature_run={run_stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        compression = None if args.compression == "none" else args.compression
        channel_map = pd.DataFrame(item.to_dict() for item in signals)
        write_parquet_chunk(channel_map, run_dir / "channel_map.parquet", compression=compression)
        query = build_feature_query(
            signals,
            table=args.table,
            timestamp_col=args.timestamp_col,
            serial_col=args.serial_col,
        )
        partitions: list[dict[str, Any]] = []
        total_feature_rows = 0
        total_source_rows = 0
        with open_db_connection(db_config) as connection:
            available = _available_columns(connection, args.table)
            required = {
                *(item.source_signal_name for item in signals),
                *GLOBAL_RECONCILIATION_SIGNALS,
                *OPERATING_CONTEXT_SIGNALS,
            }
            missing = sorted(required - available)
            if missing:
                raise ValueError("Database is missing required hierarchical signals: " + ", ".join(missing))
            for interval_start, interval_end in iter_month_ranges(args.start, args.end):
                logger.info("Building hierarchical features for %s to %s", interval_start, interval_end)
                with connection.cursor() as cursor:
                    cursor.execute(
                        query,
                        (f"{args.aggregation_minutes} minutes", str(args.serial), interval_start, interval_end),
                    )
                    rows = cursor.fetchall()
                    names = [item[0] for item in cursor.description]
                if not rows:
                    continue
                frame = normalize_timestamp_columns(
                    pd.DataFrame.from_records(rows, columns=names),
                    timestamp_columns=("window_start_utc", "first_sample_time", "last_sample_time"),
                )
                context, voltage, temperature, system = split_feature_tables(
                    frame, signals, aggregation_minutes=args.aggregation_minutes
                )
                partition_dir = run_dir / f"year={interval_start.year:04d}" / f"month={interval_start.month:02d}"
                outputs = {
                    "operating_context_5min.parquet": context,
                    "voltage_channel_features_5min.parquet": voltage,
                    "temperature_sensor_features_5min.parquet": temperature,
                    "system_features_5min.parquet": system,
                }
                files: list[dict[str, Any]] = []
                for name, output in outputs.items():
                    path = partition_dir / name.replace("5min", f"{args.aggregation_minutes}min")
                    write_parquet_chunk(output, path, compression=compression)
                    files.append({"name": str(path.relative_to(run_dir)), "rows": len(output), "sha256": _sha256(path), "bytes": path.stat().st_size})
                source_rows = int(context["sample_count"].sum())
                total_source_rows += source_rows
                total_feature_rows += len(context)
                partitions.append(
                    {
                        "start": interval_start.isoformat(),
                        "end": interval_end.isoformat(),
                        "feature_rows": len(context),
                        "source_rows": source_rows,
                        "files": files,
                    }
                )
        if not partitions:
            raise ValueError("No hierarchical electrothermal data found in the requested range")
        manifest = {
            "manifest_version": "1.0.0",
            "feature_version": FEATURE_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "serial": str(args.serial),
            "module_count": args.module_count,
            "aggregation_minutes": args.aggregation_minutes,
            "hierarchical_contract_sha256": contract_sha256(signals),
            "source_audit_run": audit_run.name,
            "source_audit_contract_sha256": audit_summary["provenance"]["hierarchical_contract_sha256"],
            "partitions": partitions,
            "value_semantics": "BMS-reported telemetry aggregated in PostgreSQL; not independently calibrated ground truth",
        }
        summary = {
            "model_family": "bms_hierarchical_electrothermal_features",
            "serial": str(args.serial),
            "module_count": args.module_count,
            "start": args.start.isoformat(),
            "end": args.end.isoformat(),
            "aggregation_minutes": args.aggregation_minutes,
            "feature_rows": total_feature_rows,
            "source_sample_rows": total_source_rows,
            "voltage_channel_count": sum(item.measurement_type == "cell_voltage_channel" for item in signals),
            "temperature_sensor_count": sum(item.measurement_type == "pack_temperature_sensor" for item in signals),
            "source_quality_gate": "passed",
            "source_audit_run": audit_run.name,
            "partition_count": len(partitions),
            "claim_limit": "Features reproduce BMS database telemetry; physical sensor accuracy requires independent calibration evidence.",
        }
        (run_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (run_dir / "model_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        logger.info("Hierarchical electrothermal feature build complete: %s", run_dir)
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Hierarchical feature build failed")
        else:
            logger.error("Hierarchical feature build failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
