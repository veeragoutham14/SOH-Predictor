from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_timestamp(value: str) -> datetime:
    """Parse CLI timestamps, accepting date, datetime, and trailing Z forms."""
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Invalid timestamp {value!r}. Use YYYY-MM-DD or ISO datetime format."
        ) from exc


def iter_month_ranges(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Split a time range into month-bounded intervals."""
    if start >= end:
        raise ValueError("start must be earlier than end.")

    intervals: list[tuple[datetime, datetime]] = []
    current = start

    while current < end:
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
        next_month = next_month.replace(hour=0, minute=0, second=0, microsecond=0)

        interval_end = min(next_month, end)
        intervals.append((current, interval_end))
        current = interval_end

    return intervals


def build_partition_dir(
    base_dir: Path,
    *,
    serial: str | int,
    interval_start: datetime,
) -> Path:
    """Build a stable partition path for raw extracted Parquet chunks."""
    return (
        base_dir
        / f"serial={serial}"
        / f"year={interval_start.year:04d}"
        / f"month={interval_start.month:02d}"
    )


def build_processing_partition_dir(
    base_dir: Path,
    *,
    serial: str | int,
    interval_start: datetime,
    run_label: str,
    run_id: str,
) -> Path:
    """Build a partition path for derived local processing layers."""
    return (
        base_dir
        / f"serial={serial}"
        / f"year={interval_start.year:04d}"
        / f"month={interval_start.month:02d}"
        / f"{run_label}={run_id}"
    )


def collect_parquet_files(path: Path) -> list[Path]:
    """Return sorted Parquet files from a file or directory path."""
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Input file is not a Parquet file: {path}")
        return [path]

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Input path is neither a file nor a directory: {path}")

    files = sorted(file for file in path.rglob("*.parquet") if file.is_file())
    if not files:
        raise FileNotFoundError(f"No Parquet files found under: {path}")
    return files


def normalize_timestamp_columns(
    df: pd.DataFrame,
    timestamp_columns: tuple[str, ...] = ("time",),
) -> pd.DataFrame:
    """Convert known timestamp columns to pandas datetime when present."""
    for column in timestamp_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def write_parquet_chunk(
    df: pd.DataFrame,
    path: Path,
    *,
    compression: str | None = "snappy",
) -> None:
    """Write one DataFrame chunk to Parquet using pyarrow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression=compression)
