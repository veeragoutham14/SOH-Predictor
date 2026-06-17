from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.config import SignalColumnConfig, StorageConfig
from src.io_utils import collect_parquet_files
from src.logging_utils import configure_logging


logger = logging.getLogger(__name__)

PART_RE = re.compile(r"part-(\d+)\.parquet$")


@dataclass(frozen=True)
class FileSummary:
    path: Path
    serial: str
    year: int | None
    month: int | None
    part_number: int | None
    rows: int
    size_mb: float
    start_timestamp: Any
    end_timestamp: Any


def _parse_partition_value(path: Path, key: str) -> str | None:
    prefix = f"{key}="
    for part in path.parts:
        if part.startswith(prefix):
            return part.removeprefix(prefix)
    return None


def _parse_part_number(path: Path) -> int | None:
    match = PART_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _timestamp_stats_from_metadata(path: Path, timestamp_col: str) -> tuple[Any, Any]:
    """Read min/max timestamp statistics from Parquet metadata when available."""
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    column_index = None

    for index in range(metadata.num_columns):
        if metadata.schema.column(index).name == timestamp_col:
            column_index = index
            break

    if column_index is None:
        return None, None

    mins: list[Any] = []
    maxs: list[Any] = []
    for row_group_index in range(metadata.num_row_groups):
        column = metadata.row_group(row_group_index).column(column_index)
        stats = column.statistics
        if stats is None or not stats.has_min_max:
            continue
        mins.append(stats.min)
        maxs.append(stats.max)

    if not mins or not maxs:
        return None, None
    return min(mins), max(maxs)


def summarize_file(path: Path, columns: SignalColumnConfig) -> FileSummary:
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata

    serial = _parse_partition_value(path, "serial") or "unknown"
    year_raw = _parse_partition_value(path, "year")
    month_raw = _parse_partition_value(path, "month")
    start_timestamp, end_timestamp = _timestamp_stats_from_metadata(
        path,
        columns.timestamp_col,
    )

    return FileSummary(
        path=path,
        serial=serial,
        year=int(year_raw) if year_raw is not None else None,
        month=int(month_raw) if month_raw is not None else None,
        part_number=_parse_part_number(path),
        rows=metadata.num_rows,
        size_mb=path.stat().st_size / (1024 * 1024),
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )


def _missing_parts(part_numbers: list[int]) -> str:
    if not part_numbers:
        return ""

    expected = set(range(min(part_numbers), max(part_numbers) + 1))
    missing = sorted(expected.difference(part_numbers))
    if not missing:
        return ""
    return ",".join(str(value) for value in missing)


def build_file_summary_table(
    files: Sequence[Path],
    columns: SignalColumnConfig,
) -> pd.DataFrame:
    summaries = [summarize_file(path, columns) for path in files]
    rows = [
        {
            "serial": item.serial,
            "year": item.year,
            "month": item.month,
            "part": item.part_number,
            "rows": item.rows,
            "size_mb": round(item.size_mb, 2),
            "start_timestamp": item.start_timestamp,
            "end_timestamp": item.end_timestamp,
            "path": str(item.path),
        }
        for item in summaries
    ]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["serial", "year", "month", "part", "path"],
        na_position="last",
    )


def build_month_summary_table(file_table: pd.DataFrame) -> pd.DataFrame:
    if file_table.empty:
        return pd.DataFrame()

    grouped = file_table.groupby(["serial", "year", "month"], dropna=False, sort=True)
    rows = []
    for (serial, year, month), group in grouped:
        parts = sorted(
            int(value)
            for value in group["part"].dropna().tolist()
        )
        rows.append(
            {
                "serial": serial,
                "year": year,
                "month": month,
                "file_count": len(group),
                "total_rows": int(group["rows"].sum()),
                "total_size_mb": round(float(group["size_mb"].sum()), 2),
                "first_timestamp": group["start_timestamp"].min(),
                "last_timestamp": group["end_timestamp"].max(),
                "first_part": min(parts) if parts else None,
                "last_part": max(parts) if parts else None,
                "missing_parts": _missing_parts(parts),
            }
        )

    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame, *, max_rows: int | None = None) -> None:
    if df.empty:
        print("No rows to display.")
        return

    display = df.copy()
    if max_rows is not None:
        display = display.head(max_rows)
    print(display.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print raw Parquet validation summaries by month and file.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Raw Parquet file or directory to inspect.")
    parser.add_argument("--serial", default=None, help="Serial folder to inspect when --input is omitted.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Override raw Parquet base directory.")
    parser.add_argument("--env-file", type=Path, default=None, help="Path to a .env file.")
    parser.add_argument("--details", action="store_true", help="Also print file-level details.")
    parser.add_argument("--max-detail-rows", type=int, default=None, help="Limit printed file-level detail rows.")
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
        storage_config = StorageConfig.from_env(args.env_file, raw_parquet_dir=args.raw_dir)
        columns = SignalColumnConfig.from_env(args.env_file)

        if args.input is not None:
            input_path = args.input
        elif args.serial is not None:
            input_path = storage_config.raw_parquet_dir / f"serial={args.serial}"
        else:
            input_path = storage_config.raw_parquet_dir

        files = collect_parquet_files(input_path)
        logger.info("Found %s Parquet files under %s", len(files), input_path)

        file_table = build_file_summary_table(files, columns)
        month_table = build_month_summary_table(file_table)

        print("\nMonthly raw Parquet summary:")
        print_table(month_table)

        if args.details:
            print("\nFile-level raw Parquet details:")
            print_table(file_table, max_rows=args.max_detail_rows)

        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Raw Parquet validation failed.")
        else:
            logger.error("Raw Parquet validation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
