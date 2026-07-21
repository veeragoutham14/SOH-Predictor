from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.build_events import _filter_frame, write_classified_rows
from src.config import SignalColumnConfig, StorageConfig
from src.io_utils import collect_parquet_files, parse_timestamp
from src.logging_utils import configure_logging
from src.mode_classification import classify_operating_modes, mode_counts

logger = logging.getLogger(__name__)


def _parse_cli_timestamp(value: str) -> datetime:
    try:
        return parse_timestamp(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def classify_raw_rows(
    input_path: Path,
    *,
    output_base_dir: Path,
    columns: SignalColumnConfig,
    serial: str,
    start: datetime | None = None,
    end: datetime | None = None,
    rest_current_threshold_a: float | None = None,
    compression: str | None = "snappy",
    run_id: str | None = None,
) -> tuple[str, int, int]:
    threshold = (
        columns.rest_current_threshold_a
        if rest_current_threshold_a is None
        else rest_current_threshold_a
    )
    if threshold < 0:
        raise ValueError("rest_current_threshold_a must be nonnegative")

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    files = collect_parquet_files(input_path)
    if not files:
        raise FileNotFoundError(f"No raw Parquet files found under {input_path}")

    rows_written = 0
    files_written = 0
    for file_index, input_file in enumerate(files):
        logger.info("Classifying raw Parquet file: %s", input_file)
        raw = pd.read_parquet(input_file)
        filtered = _filter_frame(raw, columns, serial=serial, start=start, end=end)
        if filtered.empty:
            continue
        classified = classify_operating_modes(
            filtered,
            columns,
            rest_current_threshold_a=threshold,
        )
        logger.info("Mode counts: %s", mode_counts(classified, columns.mode_col))
        paths = write_classified_rows(
            classified,
            output_base_dir=output_base_dir,
            columns=columns,
            run_id=run_id,
            file_index=file_index,
            compression=compression,
        )
        rows_written += len(classified)
        files_written += len(paths)

    if rows_written == 0:
        raise ValueError("No rows remained after applying the classification filters")
    logger.info(
        "Classification complete. run=%s rows=%s files=%s",
        run_id,
        f"{rows_written:,}",
        files_written,
    )
    return run_id, rows_written, files_written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify raw battery telemetry rows.")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start", type=_parse_cli_timestamp, default=None)
    parser.add_argument("--end", type=_parse_cli_timestamp, default=None)
    parser.add_argument("--rest-current-threshold-a", type=float, default=None)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=("snappy", "zstd", "gzip", "none"),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    try:
        storage = StorageConfig.from_env(
            args.env_file,
            raw_parquet_dir=args.raw_dir,
            classified_parquet_dir=args.output_dir,
        )
        storage.ensure_dirs()
        columns = SignalColumnConfig.from_env(args.env_file)
        input_path = args.input or storage.raw_parquet_dir / f"serial={args.serial}"
        compression = None if args.compression == "none" else args.compression
        classify_raw_rows(
            input_path,
            output_base_dir=storage.classified_parquet_dir,
            columns=columns,
            serial=str(args.serial),
            start=args.start,
            end=args.end,
            rest_current_threshold_a=args.rest_current_threshold_a,
            compression=compression,
        )
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Classification failed")
        else:
            logger.error("Classification failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
