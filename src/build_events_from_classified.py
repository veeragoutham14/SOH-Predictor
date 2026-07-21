from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.build_events import write_event_tables
from src.config import SignalColumnConfig, StorageConfig
from src.discharge_cycle_builder import _path_partition_values
from src.event_segmentation import EventMerger, summarize_classified_events
from src.io_utils import collect_parquet_files
from src.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def _select_classified_files(
    files: Sequence[Path],
    *,
    classified_run_id: str | None,
) -> list[Path]:
    if classified_run_id:
        selected = [
            path
            for path in files
            if _path_partition_values(path).get("classified_run") == classified_run_id
        ]
        if not selected:
            raise FileNotFoundError(
                f"No classified files found for classified_run={classified_run_id}"
            )
        return sorted(selected)

    grouped: dict[tuple[str, str, str], list[tuple[str, Path]]] = {}
    unpartitioned: list[Path] = []
    for path in files:
        values = _path_partition_values(path)
        run_id = values.get("classified_run")
        partition = (values.get("serial"), values.get("year"), values.get("month"))
        if run_id is None or any(value is None for value in partition):
            unpartitioned.append(path)
            continue
        grouped.setdefault(tuple(str(value) for value in partition), []).append(
            (run_id, path)
        )

    selected = list(unpartitioned)
    for candidates in grouped.values():
        latest_run = max(run_id for run_id, _ in candidates)
        selected.extend(path for run_id, path in candidates if run_id == latest_run)
    return sorted(selected)


def build_events_from_classified(
    input_path: Path,
    *,
    output_base_dir: Path,
    columns: SignalColumnConfig,
    serial: str,
    classified_run_id: str | None = None,
    rest_current_threshold_a: float | None = None,
    compression: str | None = "snappy",
    run_id: str | None = None,
) -> tuple[str, int, tuple[Path, ...]]:
    threshold = (
        columns.rest_current_threshold_a
        if rest_current_threshold_a is None
        else rest_current_threshold_a
    )
    if threshold < 0:
        raise ValueError("rest_current_threshold_a must be nonnegative")

    files = _select_classified_files(
        collect_parquet_files(input_path),
        classified_run_id=classified_run_id,
    )
    if not files:
        raise FileNotFoundError(f"No classified Parquet files found under {input_path}")

    merger = EventMerger(
        rest_current_threshold_a=threshold,
        missing_gap_threshold_seconds=columns.missing_gap_threshold_seconds,
        expected_sample_interval_seconds=columns.expected_sample_interval_seconds,
    )
    for input_file in files:
        logger.info("Building events from: %s", input_file)
        classified = pd.read_parquet(input_file)
        if columns.serial_col in classified.columns:
            classified = classified[
                classified[columns.serial_col].astype(str).eq(str(serial))
            ]
        if classified.empty:
            continue
        merger.add_events(
            summarize_classified_events(
                classified,
                columns,
                rest_current_threshold_a=threshold,
            )
        )

    events = merger.to_frame(event_id_col=columns.event_id_col)
    if events.empty:
        raise ValueError("No events were created from the classified telemetry")
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths = tuple(
        write_event_tables(
            events,
            output_base_dir=output_base_dir,
            run_id=run_id,
            compression=compression,
        )
    )
    logger.info(
        "Event build complete. run=%s events=%s files=%s",
        run_id,
        f"{len(events):,}",
        len(paths),
    )
    return run_id, len(events), paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build battery events from classified telemetry Parquet."
    )
    parser.add_argument("--serial", required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--classified-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--classified-run-id", default=None)
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
            classified_parquet_dir=args.classified_dir,
            event_parquet_dir=args.output_dir,
        )
        storage.ensure_dirs()
        columns = SignalColumnConfig.from_env(args.env_file)
        input_path = args.input or storage.classified_parquet_dir / f"serial={args.serial}"
        compression = None if args.compression == "none" else args.compression
        build_events_from_classified(
            input_path,
            output_base_dir=storage.event_parquet_dir,
            columns=columns,
            serial=str(args.serial),
            classified_run_id=args.classified_run_id,
            rest_current_threshold_a=args.rest_current_threshold_a,
            compression=compression,
        )
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Event build failed")
        else:
            logger.error("Event build failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
