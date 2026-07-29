from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from psycopg2 import sql

from src.config import DBConfig, StorageConfig
from src.db import DEFAULT_SERIAL_COL, DEFAULT_TABLE, DEFAULT_TIMESTAMP_COL, open_db_connection
from src.io_utils import parse_timestamp, write_parquet_chunk
from src.logging_utils import configure_logging

logger = logging.getLogger(__name__)

CELL_COLUMN_PATTERN = re.compile(
    r"^bspi2_batterymodules_mod(?P<module>[1-4])_pack(?P<pack>[1-2])_"
    r"cellvoltage_mv_(?P<cell>\d+)$"
)
GLOBAL_CELL_COLUMNS = {
    "bspi2_cellvoltagemin_v": (2.0, 5.0),
    "bspi2_cellvoltagemax_v": (2.0, 5.0),
    "bspi2_compensatedcellvoltagemin_v": (2.0, 5.0),
    "bspi2_compensatedcellvoltagemax_v": (2.0, 5.0),
}
CELL_TEMPERATURE_COLUMNS = {
    "bspi2_celltemperaturemin_c": (-40.0, 100.0),
    "bspi2_celltemperaturemax_c": (-40.0, 100.0),
}


@dataclass(frozen=True)
class SignalAvailability:
    signal: str
    signal_group: str
    module: int | None
    pack: int | None
    cell: int | None
    present_in_schema: bool
    non_null_rows: int
    valid_rows: int
    non_null_coverage_pct: float
    valid_coverage_pct: float
    ready: bool


def _qualified_identifier(name: str) -> sql.Composed:
    parts = [part.strip() for part in name.split(".")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return sql.SQL(".").join(sql.Identifier(part) for part in parts)


def _table_parts(name: str) -> tuple[str, str]:
    parts = [part.strip() for part in name.split(".")]
    if len(parts) == 1:
        return "public", parts[0]
    if len(parts) == 2 and all(parts):
        return parts[0], parts[1]
    raise ValueError(f"Expected table or schema.table, received: {name!r}")


def _available_columns(connection: Any, table: str) -> set[str]:
    schema_name, table_name = _table_parts(table)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema_name, table_name),
        )
        return {str(row[0]) for row in cursor.fetchall()}


def expected_cell_columns(module_count: int) -> list[str]:
    if not 1 <= module_count <= 4:
        raise ValueError("module_count must be between 1 and 4")
    return [
        f"bspi2_batterymodules_mod{module}_pack{pack}_cellvoltage_mv_{cell}"
        for module in range(1, module_count + 1)
        for pack in range(1, 3)
        for cell in range(14)
    ]


def _latest_timestamp(
    connection: Any,
    *,
    serial: str,
    table: str,
    timestamp_col: str,
    serial_col: str,
) -> datetime:
    query = sql.SQL("SELECT MAX({timestamp}) FROM {table} WHERE {serial_col} = %s").format(
        timestamp=sql.Identifier(timestamp_col),
        table=_qualified_identifier(table),
        serial_col=sql.Identifier(serial_col),
    )
    with connection.cursor() as cursor:
        cursor.execute(query, (serial,))
        value = cursor.fetchone()[0]
    if value is None:
        raise ValueError(f"No database rows found for serial {serial}")
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _representative_recent_day(
    connection: Any,
    *,
    serial: str,
    table: str,
    timestamp_col: str,
    serial_col: str,
) -> tuple[datetime, datetime]:
    latest = _latest_timestamp(
        connection,
        serial=serial,
        table=table,
        timestamp_col=timestamp_col,
        serial_col=serial_col,
    )
    search_end = latest.astimezone(UTC).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) + timedelta(days=1)
    search_start = search_end - timedelta(days=14)
    query = sql.SQL(
        """
        SELECT date_trunc('day', {timestamp}) AS sample_day, COUNT(*) AS row_count
        FROM {table}
        WHERE {serial_col} = %s
          AND {timestamp} >= %s
          AND {timestamp} < %s
        GROUP BY sample_day
        ORDER BY row_count DESC, sample_day DESC
        LIMIT 1
        """
    ).format(
        timestamp=sql.Identifier(timestamp_col),
        table=_qualified_identifier(table),
        serial_col=sql.Identifier(serial_col),
    )
    with connection.cursor() as cursor:
        cursor.execute(query, (serial, search_start, search_end))
        row = cursor.fetchone()
    if row is None:
        raise ValueError(f"No recent sample day found for serial {serial}")
    start = row[0] if row[0].tzinfo else row[0].replace(tzinfo=UTC)
    return start, start + timedelta(days=1)


def _observed_module_count(
    connection: Any,
    *,
    serial: str,
    start: datetime,
    end: datetime,
    table: str,
    timestamp_col: str,
    serial_col: str,
    module_count_col: str,
) -> int:
    query = sql.SQL(
        """
        SELECT {module_count}, COUNT(*) AS observations
        FROM {table}
        WHERE {serial_col} = %s
          AND {timestamp} >= %s
          AND {timestamp} < %s
          AND {module_count} IS NOT NULL
        GROUP BY {module_count}
        ORDER BY observations DESC, {module_count} DESC
        LIMIT 1
        """
    ).format(
        module_count=sql.Identifier(module_count_col),
        table=_qualified_identifier(table),
        serial_col=sql.Identifier(serial_col),
        timestamp=sql.Identifier(timestamp_col),
    )
    with connection.cursor() as cursor:
        cursor.execute(query, (serial, start, end))
        row = cursor.fetchone()
    if row is None:
        raise ValueError(f"No module count found for serial {serial} in the sample window")
    value = float(row[0])
    if not value.is_integer() or not 1 <= int(value) <= 4:
        raise ValueError(f"Invalid module count for serial {serial}: {value}")
    return int(value)


def _signal_metadata(signal: str) -> tuple[str, int | None, int | None, int | None]:
    match = CELL_COLUMN_PATTERN.fullmatch(signal)
    if match:
        return (
            "individual_cell_voltage",
            int(match.group("module")),
            int(match.group("pack")),
            int(match.group("cell")),
        )
    if signal in GLOBAL_CELL_COLUMNS:
        return "global_cell_voltage", None, None, None
    return "cell_temperature", None, None, None


def _range_for_signal(signal: str) -> tuple[float, float]:
    if CELL_COLUMN_PATTERN.fullmatch(signal):
        return 2000.0, 5000.0
    if signal in GLOBAL_CELL_COLUMNS:
        return GLOBAL_CELL_COLUMNS[signal]
    return CELL_TEMPERATURE_COLUMNS[signal]


def _availability_query(
    signals: Sequence[str],
    *,
    table: str,
    timestamp_col: str,
    serial_col: str,
) -> sql.Composed:
    aggregates: list[sql.Composed] = [
        sql.SQL("COUNT(*) AS total_rows"),
        sql.SQL("MIN({timestamp}) AS first_timestamp").format(
            timestamp=sql.Identifier(timestamp_col)
        ),
        sql.SQL("MAX({timestamp}) AS last_timestamp").format(
            timestamp=sql.Identifier(timestamp_col)
        ),
    ]
    for index, signal in enumerate(signals):
        lower, upper = _range_for_signal(signal)
        aggregates.extend(
            (
                sql.SQL("COUNT({signal}) AS {alias}").format(
                    signal=sql.Identifier(signal),
                    alias=sql.Identifier(f"non_null_{index}"),
                ),
                sql.SQL(
                    "COUNT(*) FILTER (WHERE {signal} BETWEEN {lower} AND {upper}) "
                    "AS {alias}"
                ).format(
                    signal=sql.Identifier(signal),
                    lower=sql.Literal(lower),
                    upper=sql.Literal(upper),
                    alias=sql.Identifier(f"valid_{index}"),
                ),
            )
        )
    return sql.SQL(
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


def audit_signals(
    connection: Any,
    *,
    serial: str,
    start: datetime | None,
    end: datetime | None,
    module_count: int | None,
    readiness_threshold: float,
    table: str = DEFAULT_TABLE,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
    serial_col: str = DEFAULT_SERIAL_COL,
    module_count_col: str = "bspi2_modulecount",
) -> tuple[dict[str, Any], pd.DataFrame]:
    if (start is None) != (end is None):
        raise ValueError("start and end must be provided together")
    if start is not None and end is not None and start >= end:
        raise ValueError("start must be earlier than end")
    if not 0 < readiness_threshold <= 1:
        raise ValueError("readiness_threshold must be between 0 and 1")

    selected_automatically = start is None
    if start is None or end is None:
        start, end = _representative_recent_day(
            connection,
            serial=serial,
            table=table,
            timestamp_col=timestamp_col,
            serial_col=serial_col,
        )
    available = _available_columns(connection, table)
    if module_count_col not in available:
        raise ValueError(f"Database table is missing {module_count_col}")
    observed_modules = module_count or _observed_module_count(
        connection,
        serial=serial,
        start=start,
        end=end,
        table=table,
        timestamp_col=timestamp_col,
        serial_col=serial_col,
        module_count_col=module_count_col,
    )
    expected_cells = expected_cell_columns(observed_modules)
    auxiliary = [*GLOBAL_CELL_COLUMNS, *CELL_TEMPERATURE_COLUMNS]
    expected_signals = [*expected_cells, *auxiliary]
    query_signals = [signal for signal in expected_signals if signal in available]
    query = _availability_query(
        query_signals,
        table=table,
        timestamp_col=timestamp_col,
        serial_col=serial_col,
    )
    with connection.cursor() as cursor:
        cursor.execute(query, (serial, start, end))
        row = cursor.fetchone()
        names = [column[0] for column in cursor.description]
    values = dict(zip(names, row, strict=True))
    total_rows = int(values["total_rows"] or 0)
    if total_rows == 0:
        raise ValueError(f"No rows found for serial {serial} in {start} to {end}")

    records: list[SignalAvailability] = []
    query_index = {signal: index for index, signal in enumerate(query_signals)}
    for signal in expected_signals:
        signal_group, module, pack, cell = _signal_metadata(signal)
        index = query_index.get(signal)
        non_null_rows = int(values[f"non_null_{index}"] or 0) if index is not None else 0
        valid_rows = int(values[f"valid_{index}"] or 0) if index is not None else 0
        non_null_coverage = 100.0 * non_null_rows / total_rows
        valid_coverage = 100.0 * valid_rows / total_rows
        records.append(
            SignalAvailability(
                signal=signal,
                signal_group=signal_group,
                module=module,
                pack=pack,
                cell=cell,
                present_in_schema=index is not None,
                non_null_rows=non_null_rows,
                valid_rows=valid_rows,
                non_null_coverage_pct=non_null_coverage,
                valid_coverage_pct=valid_coverage,
                ready=(index is not None and valid_rows / total_rows >= readiness_threshold),
            )
        )

    cell_records = [item for item in records if item.signal_group == "individual_cell_voltage"]
    ready_cell_channels = sum(item.ready for item in cell_records)
    duration_seconds = max(1.0, (end - start).total_seconds())
    summary = {
        "model_family": "cell_signal_availability",
        "serial": serial,
        "sample_start": start.isoformat(),
        "sample_end": end.isoformat(),
        "sample_selected_automatically": selected_automatically,
        "module_count": observed_modules,
        "expected_cell_channels": len(expected_cells),
        "present_cell_channels": sum(item.present_in_schema for item in cell_records),
        "ready_cell_channels": ready_cell_channels,
        "readiness_threshold_pct": 100.0 * readiness_threshold,
        "total_rows": total_rows,
        "first_timestamp": str(values["first_timestamp"]),
        "last_timestamp": str(values["last_timestamp"]),
        "mean_samples_per_second": total_rows / duration_seconds,
        "nominal_1hz_coverage_pct": min(100.0, 100.0 * total_rows / duration_seconds),
        "minimum_cell_channel_valid_coverage_pct": min(
            item.valid_coverage_pct for item in cell_records
        ),
        "global_cell_voltage_signals_ready": all(
            item.ready for item in records if item.signal_group == "global_cell_voltage"
        ),
        "cell_temperature_signals_ready": all(
            item.ready for item in records if item.signal_group == "cell_temperature"
        ),
        "cell_voltage_ready": ready_cell_channels == len(expected_cells),
    }
    return summary, pd.DataFrame(asdict(item) for item in records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit active battery cell-voltage channels in PostgreSQL."
    )
    parser.add_argument("--serial", required=True)
    parser.add_argument("--start", type=parse_timestamp)
    parser.add_argument("--end", type=parse_timestamp)
    parser.add_argument("--module-count", type=int)
    parser.add_argument("--readiness-threshold", type=float, default=0.95)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--timestamp-col", default=DEFAULT_TIMESTAMP_COL)
    parser.add_argument("--serial-col", default=DEFAULT_SERIAL_COL)
    parser.add_argument("--module-count-col", default="bspi2_modulecount")
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
        db_config = DBConfig.from_env(args.env_file)
        storage = StorageConfig.from_env(args.env_file)
        output_root = args.output_root or storage.data_dir / "processed" / "cell_signal_audit"
        with open_db_connection(db_config) as connection:
            summary, availability = audit_signals(
                connection,
                serial=str(args.serial),
                start=args.start,
                end=args.end,
                module_count=args.module_count,
                readiness_threshold=args.readiness_threshold,
                table=args.table,
                timestamp_col=args.timestamp_col,
                serial_col=args.serial_col,
                module_count_col=args.module_count_col,
            )

        run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = (
            output_root
            / f"serial={args.serial}"
            / f"cell_audit_run={run_stamp}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        compression = None if args.compression == "none" else args.compression
        write_parquet_chunk(
            availability,
            run_dir / "cell_signal_availability.parquet",
            compression=compression,
        )
        (run_dir / "model_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Cell-voltage audit complete for serial=%s: %s/%s active channels ready; %s",
            args.serial,
            summary["ready_cell_channels"],
            summary["expected_cell_channels"],
            run_dir,
        )
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Cell-voltage signal audit failed")
        else:
            logger.error("Cell-voltage signal audit failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
