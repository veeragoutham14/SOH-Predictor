from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import pandas as pd

from src.config import (
    DBConfig,
    SignalColumnConfig,
    StorageConfig,
    get_extraction_column_profile,
    get_extraction_columns,
)
from src.db import (
    DEFAULT_SERIAL_COL,
    DEFAULT_TABLE,
    DEFAULT_TIMESTAMP_COL,
    build_extract_query,
    cursor_column_names,
    open_db_connection,
)
from src.io_utils import (
    build_partition_dir,
    iter_month_ranges,
    normalize_timestamp_columns,
    parse_timestamp,
    write_parquet_chunk,
)
from src.logging_utils import configure_logging


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractRequest:
    serial: str | int
    start: datetime
    end: datetime
    limit: int | None = None
    columns: Sequence[str] | None = None
    chunk_size: int = 100_000
    table: str = DEFAULT_TABLE
    timestamp_col: str = DEFAULT_TIMESTAMP_COL
    serial_col: str = DEFAULT_SERIAL_COL
    compression: str | None = "snappy"

    def validate(self) -> None:
        if self.start >= self.end:
            raise ValueError("start must be earlier than end.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when provided.")


@dataclass(frozen=True)
class ExtractionResult:
    serial: str | int
    start: datetime
    end: datetime
    rows: int
    chunks: int
    files: tuple[Path, ...]
    output_dir: Path


def _dataframe_from_rows(rows: list[tuple], columns: Sequence[str]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(rows, columns=columns)
    return normalize_timestamp_columns(df)


def extract_range_to_parquet(
    request: ExtractRequest,
    *,
    db_config: DBConfig,
    output_base_dir: Path,
) -> ExtractionResult:
    """Extract one time range for one battery serial into chunked Parquet files."""
    request.validate()

    output_dir = build_partition_dir(
        output_base_dir,
        serial=request.serial,
        interval_start=request.start,
    )

    query, params = build_extract_query(
        serial=request.serial,
        start=request.start,
        end=request.end,
        limit=request.limit,
        columns=request.columns,
        table=request.table,
        timestamp_col=request.timestamp_col,
        serial_col=request.serial_col,
    )

    logger.info(
        "Extracting serial=%s from %s to %s into %s",
        request.serial,
        request.start.isoformat(),
        request.end.isoformat(),
        output_dir,
    )

    rows_written = 0
    chunk_count = 0
    files: list[Path] = []
    cursor_name = f"battery_extract_{uuid4().hex}"

    with open_db_connection(db_config) as conn:
        with conn.cursor(name=cursor_name) as cur:
            cur.itersize = request.chunk_size
            cur.execute(query, params)
            columns: list[str] | None = None

            while True:
                rows = cur.fetchmany(request.chunk_size)
                if not rows:
                    break

                if columns is None:
                    columns = cursor_column_names(cur)

                df = _dataframe_from_rows(rows, columns)
                part_path = output_dir / f"part-{chunk_count:05d}.parquet"
                write_parquet_chunk(
                    df,
                    part_path,
                    compression=request.compression,
                )

                chunk_rows = len(df)
                rows_written += chunk_rows
                chunk_count += 1
                files.append(part_path)

                logger.info(
                    "Wrote chunk %s with %s rows to %s",
                    chunk_count,
                    f"{chunk_rows:,}",
                    part_path,
                )

    if rows_written == 0:
        logger.warning("No rows returned for serial=%s in requested range.", request.serial)
    else:
        logger.info(
            "Finished range. Wrote %s rows across %s files.",
            f"{rows_written:,}",
            chunk_count,
        )

    return ExtractionResult(
        serial=request.serial,
        start=request.start,
        end=request.end,
        rows=rows_written,
        chunks=chunk_count,
        files=tuple(files),
        output_dir=output_dir,
    )


def _parse_cli_timestamp(value: str) -> datetime:
    try:
        return parse_timestamp(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract battery log rows from TimescaleDB/PostgreSQL to Parquet.",
    )
    parser.add_argument("--serial", required=True, help="Battery serial number to extract.")
    parser.add_argument("--start", required=True, type=_parse_cli_timestamp, help="Start timestamp, inclusive.")
    parser.add_argument("--end", required=True, type=_parse_cli_timestamp, help="End timestamp, exclusive.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum rows to extract.")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Rows fetched and written per chunk.")
    parser.add_argument("--column-profile", default=None, help="Configured extraction column profile to use.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override raw Parquet output directory.")
    parser.add_argument("--env-file", type=Path, default=None, help="Path to a .env file.")
    parser.add_argument("--table", default=DEFAULT_TABLE, help="Source table, optionally schema qualified.")
    parser.add_argument("--timestamp-col", default=DEFAULT_TIMESTAMP_COL, help="Timestamp column used for filtering.")
    parser.add_argument("--serial-col", default=DEFAULT_SERIAL_COL, help="Serial number column used for filtering.")
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=("snappy", "zstd", "gzip", "none"),
        help="Parquet compression codec.",
    )
    parser.add_argument(
        "--no-month-split",
        action="store_true",
        help="Run one query over the full range instead of splitting by month.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console log level.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    try:
        db_config = DBConfig.from_env(args.env_file)
        storage_config = StorageConfig.from_env(
            args.env_file,
            raw_parquet_dir=args.output_dir,
        )
        storage_config.ensure_dirs()
        signal_columns = SignalColumnConfig.from_env(args.env_file)
        column_profile = get_extraction_column_profile(args.column_profile)
        columns = get_extraction_columns(signal_columns, profile=column_profile)

        intervals = (
            [(args.start, args.end)]
            if args.no_month_split
            else iter_month_ranges(args.start, args.end)
        )

        compression = None if args.compression == "none" else args.compression
        remaining_limit = args.limit

        total_rows = 0
        total_files = 0

        logger.info(
            "Using extraction column profile '%s': %s",
            column_profile,
            ", ".join(columns),
        )

        for interval_start, interval_end in intervals:
            if remaining_limit is not None and remaining_limit <= 0:
                break

            request = ExtractRequest(
                serial=args.serial,
                start=interval_start,
                end=interval_end,
                limit=remaining_limit,
                columns=columns,
                chunk_size=args.chunk_size,
                table=args.table,
                timestamp_col=args.timestamp_col,
                serial_col=args.serial_col,
                compression=compression,
            )
            result = extract_range_to_parquet(
                request,
                db_config=db_config,
                output_base_dir=storage_config.raw_parquet_dir,
            )

            total_rows += result.rows
            total_files += len(result.files)
            if remaining_limit is not None:
                remaining_limit -= result.rows

        logger.info(
            "Extraction complete. Total rows=%s, files=%s, output base=%s",
            f"{total_rows:,}",
            total_files,
            storage_config.raw_parquet_dir,
        )
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Extraction failed.")
        else:
            logger.error("Extraction failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
