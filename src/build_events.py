from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import SignalColumnConfig, StorageConfig
from src.event_segmentation import EventMerger, summarize_classified_events
from src.io_utils import (
    build_processing_partition_dir,
    collect_parquet_files,
    parse_timestamp,
    write_parquet_chunk,
)
from src.logging_utils import configure_logging
from src.mode_classification import classify_operating_modes, mode_counts


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildEventsResult:
    raw_files_processed: int
    classified_files_written: int
    events_created: int
    event_files_written: tuple[Path, ...]


def _timestamp_bound(value: datetime | None) -> pd.Timestamp | None:
    if value is None:
        return None
    bound = pd.Timestamp(value)
    if bound.tzinfo is None:
        return bound.tz_localize("UTC")
    return bound.tz_convert("UTC")


def _filter_frame(
    df: pd.DataFrame,
    columns: SignalColumnConfig,
    *,
    serial: str | int | None,
    start: datetime | None,
    end: datetime | None,
) -> pd.DataFrame:
    work = df.copy()
    if columns.timestamp_col not in work.columns:
        raise ValueError(f"Missing timestamp column: {columns.timestamp_col}")

    work[columns.timestamp_col] = pd.to_datetime(
        work[columns.timestamp_col],
        errors="coerce",
        utc=True,
    )
    before_drop = len(work)
    work = work.dropna(subset=[columns.timestamp_col])
    dropped = before_drop - len(work)
    if dropped:
        logger.warning("Dropped %s rows with missing/invalid timestamps.", f"{dropped:,}")

    if serial is not None:
        if columns.serial_col not in work.columns:
            raise ValueError(f"Missing serial column: {columns.serial_col}")
        work = work[work[columns.serial_col].astype(str).eq(str(serial))]

    start_bound = _timestamp_bound(start)
    end_bound = _timestamp_bound(end)
    if start_bound is not None:
        work = work[work[columns.timestamp_col] >= start_bound]
    if end_bound is not None:
        work = work[work[columns.timestamp_col] < end_bound]

    return work


def _classified_partition_key(
    df: pd.DataFrame,
    columns: SignalColumnConfig,
) -> pd.DataFrame:
    work = df.copy()
    work["_partition_year"] = work[columns.timestamp_col].dt.year
    work["_partition_month"] = work[columns.timestamp_col].dt.month
    return work


def write_classified_rows(
    df: pd.DataFrame,
    *,
    output_base_dir: Path,
    columns: SignalColumnConfig,
    run_id: str,
    file_index: int,
    compression: str | None,
) -> list[Path]:
    """Write classified row-level telemetry grouped by serial/year/month."""
    if df.empty:
        return []

    paths: list[Path] = []
    partitioned = _classified_partition_key(df, columns)
    group_cols = [columns.serial_col, "_partition_year", "_partition_month"]

    for group_index, ((serial, year, month), group) in enumerate(partitioned.groupby(group_cols, sort=True)):
        interval_start = datetime(int(year), int(month), 1)
        out_dir = build_processing_partition_dir(
            output_base_dir,
            serial=serial,
            interval_start=interval_start,
            run_label="classified_run",
            run_id=run_id,
        )
        out_path = out_dir / f"part-{file_index:05d}-{group_index:03d}.parquet"
        output = group.drop(columns=["_partition_year", "_partition_month"])
        write_parquet_chunk(output, out_path, compression=compression)
        paths.append(out_path)

    return paths


def _event_partition_key(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    work["_partition_year"] = work["start_timestamp"].dt.year
    work["_partition_month"] = work["start_timestamp"].dt.month
    return work


def write_event_tables(
    events: pd.DataFrame,
    *,
    output_base_dir: Path,
    run_id: str,
    compression: str | None,
) -> list[Path]:
    """Write event summaries grouped by serial and event-start month."""
    if events.empty:
        return []

    paths: list[Path] = []
    partitioned = _event_partition_key(events)

    for (serial, year, month), group in partitioned.groupby(["serial", "_partition_year", "_partition_month"], sort=True):
        interval_start = datetime(int(year), int(month), 1)
        out_dir = build_processing_partition_dir(
            output_base_dir,
            serial=serial,
            interval_start=interval_start,
            run_label="event_run",
            run_id=run_id,
        )
        out_path = out_dir / "events.parquet"
        output = group.drop(columns=["_partition_year", "_partition_month"])
        write_parquet_chunk(output, out_path, compression=compression)
        paths.append(out_path)

    return paths


def build_events_from_raw_parquet(
    input_files: Sequence[Path],
    *,
    columns: SignalColumnConfig,
    classified_output_dir: Path,
    event_output_dir: Path,
    serial: str | int | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    rest_current_threshold_a: float | None = None,
    compression: str | None = "snappy",
    run_id: str | None = None,
) -> BuildEventsResult:
    """Classify raw telemetry rows and build event summaries from local Parquet."""
    threshold = (
        columns.rest_current_threshold_a
        if rest_current_threshold_a is None
        else rest_current_threshold_a
    )
    if threshold < 0:
        raise ValueError("rest_current_threshold_a must be nonnegative.")

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    merger = EventMerger(
        rest_current_threshold_a=threshold,
        missing_gap_threshold_seconds=columns.missing_gap_threshold_seconds,
        expected_sample_interval_seconds=columns.expected_sample_interval_seconds,
    )
    classified_file_count = 0
    raw_files_processed = 0

    for file_index, input_file in enumerate(input_files):
        logger.info("Loading raw Parquet file: %s", input_file)
        raw = pd.read_parquet(input_file)
        logger.info("Loaded %s raw rows.", f"{len(raw):,}")

        filtered = _filter_frame(
            raw,
            columns,
            serial=serial,
            start=start,
            end=end,
        )
        if filtered.empty:
            logger.info("No rows remain after filters for file: %s", input_file)
            continue

        classified = classify_operating_modes(
            filtered,
            columns,
            rest_current_threshold_a=threshold,
        )
        counts = mode_counts(classified, columns.mode_col)
        logger.info("Classified %s rows. Mode counts: %s", f"{len(classified):,}", counts)

        classified_paths = write_classified_rows(
            classified,
            output_base_dir=classified_output_dir,
            columns=columns,
            run_id=run_id,
            file_index=file_index,
            compression=compression,
        )
        classified_file_count += len(classified_paths)
        for path in classified_paths:
            logger.info("Wrote classified rows to: %s", path)

        events = summarize_classified_events(
            classified,
            columns,
            rest_current_threshold_a=threshold,
        )
        logger.info("Created %s local events from file.", f"{len(events):,}")
        merger.add_events(events)
        raw_files_processed += 1

    event_table = merger.to_frame(event_id_col=columns.event_id_col)
    event_paths = write_event_tables(
        event_table,
        output_base_dir=event_output_dir,
        run_id=run_id,
        compression=compression,
    )

    if event_table.empty:
        logger.warning("No events were created.")
    else:
        logger.info("Created %s merged events.", f"{len(event_table):,}")
        logger.info("Event mode distribution: %s", event_table["mode"].value_counts().to_dict())
        for path in event_paths:
            logger.info("Wrote event table to: %s", path)

    return BuildEventsResult(
        raw_files_processed=raw_files_processed,
        classified_files_written=classified_file_count,
        events_created=len(event_table),
        event_files_written=tuple(event_paths),
    )


def _parse_cli_timestamp(value: str) -> datetime:
    try:
        return parse_timestamp(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify raw battery telemetry rows and build event-level Parquet tables.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Raw Parquet file or directory to process.")
    parser.add_argument("--serial", default=None, help="Optional serial filter. Used to locate raw files when --input is omitted.")
    parser.add_argument("--start", type=_parse_cli_timestamp, default=None, help="Optional start timestamp, inclusive.")
    parser.add_argument("--end", type=_parse_cli_timestamp, default=None, help="Optional end timestamp, exclusive.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Override raw Parquet base directory.")
    parser.add_argument("--classified-output-dir", type=Path, default=None, help="Override classified row output directory.")
    parser.add_argument("--event-output-dir", type=Path, default=None, help="Override event output directory.")
    parser.add_argument("--env-file", type=Path, default=None, help="Path to a .env file.")
    parser.add_argument("--rest-current-threshold-a", type=float, default=None, help="Override rest current threshold in amps.")
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=("snappy", "zstd", "gzip", "none"),
        help="Parquet compression codec.",
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
        storage_config = StorageConfig.from_env(
            args.env_file,
            raw_parquet_dir=args.raw_dir,
            classified_parquet_dir=args.classified_output_dir,
            event_parquet_dir=args.event_output_dir,
        )
        storage_config.ensure_dirs()
        columns = SignalColumnConfig.from_env(args.env_file)

        if args.input is not None:
            input_path = args.input
        elif args.serial is not None:
            input_path = storage_config.raw_parquet_dir / f"serial={args.serial}"
        else:
            raise ValueError("Provide either --input or --serial.")

        input_files = collect_parquet_files(input_path)
        logger.info("Found %s raw Parquet files to process.", len(input_files))

        compression = None if args.compression == "none" else args.compression
        result = build_events_from_raw_parquet(
            input_files,
            columns=columns,
            classified_output_dir=storage_config.classified_parquet_dir,
            event_output_dir=storage_config.event_parquet_dir,
            serial=args.serial,
            start=args.start,
            end=args.end,
            rest_current_threshold_a=args.rest_current_threshold_a,
            compression=compression,
        )
        logger.info(
            "Build complete. Raw files=%s, classified files=%s, events=%s.",
            result.raw_files_processed,
            result.classified_files_written,
            result.events_created,
        )
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Event build failed.")
        else:
            logger.error("Event build failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
