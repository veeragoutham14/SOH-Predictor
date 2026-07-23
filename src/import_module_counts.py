from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from psycopg2 import sql

from src.config import DBConfig
from src.db import DEFAULT_SERIAL_COL, DEFAULT_TABLE, open_db_connection
from src.logging_utils import configure_logging


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "device_metadata" / "module_counts.parquet"


def _qualified_identifier(name: str) -> sql.Composed:
    parts = [part.strip() for part in name.split(".")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return sql.SQL(".").join(sql.Identifier(part) for part in parts)


def _configured_serials(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    variants = payload.get("variants", {})
    serials: set[str] = set()
    for variant in variants.values():
        for device in variant.get("devices", []):
            serial = str(device.get("serial", "")).strip()
            if not serial.isdigit():
                raise ValueError(f"Invalid configured serial: {serial!r}")
            serials.add(serial)
    if not serials:
        raise ValueError(f"No device serials found in {path}")
    return sorted(serials)


def fetch_module_counts(
    *,
    db_config: DBConfig,
    serials: Sequence[str],
    table: str,
    serial_col: str,
    module_count_col: str,
    timestamp_col: str,
    recent_samples: int,
) -> dict[str, list[Any]]:
    query = sql.SQL(
        """
        WITH requested(serial) AS (
            SELECT unnest(%s::bigint[])
        )
        SELECT requested.serial, recent.module_count, recent.sample_count
        FROM requested
        LEFT JOIN LATERAL (
            SELECT module_count, count(*) AS sample_count
            FROM (
                SELECT {module_count_col} AS module_count
                FROM {table}
                WHERE {serial_col} = requested.serial
                  AND {module_count_col} IS NOT NULL
                ORDER BY {timestamp_col} DESC
                LIMIT %s
            ) latest
            GROUP BY module_count
        ) recent ON TRUE
        ORDER BY requested.serial, recent.module_count
        """
    ).format(
        serial_col=sql.Identifier(serial_col),
        module_count_col=sql.Identifier(module_count_col),
        timestamp_col=sql.Identifier(timestamp_col),
        table=_qualified_identifier(table),
    )
    values: dict[str, list[Any]] = {serial: [] for serial in serials}
    with open_db_connection(db_config) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, [[int(serial) for serial in serials], recent_samples])
            for serial, module_count, sample_count in cursor.fetchall():
                if module_count is not None:
                    values.setdefault(str(serial), []).extend([module_count] * sample_count)
    return values


def _normalized_count(value: Any) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def build_snapshot(
    serials: Sequence[str],
    observed: dict[str, list[Any]],
    minimum_agreement: float = 0.95,
) -> pd.DataFrame:
    imported_at = datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    for serial in serials:
        counts = Counter(_normalized_count(value) for value in observed.get(serial, []))
        values = sorted(counts)
        sample_rows = sum(counts.values())
        module_count = counts.most_common(1)[0][0] if counts else None
        agreement = counts[module_count] / sample_rows if module_count is not None else 0.0
        valid_count = (
            module_count is not None
            and float(module_count).is_integer()
            and module_count > 0
        )
        status = (
            "missing"
            if not values
            else "invalid"
            if not valid_count
            else "verified"
            if agreement >= minimum_agreement
            else "unstable"
        )
        rows.append(
            {
                "serial": serial,
                "module_count": int(module_count) if valid_count else module_count,
                "observed_module_counts": ",".join(str(value) for value in values),
                "module_count_sample_rows": sample_rows,
                "module_count_agreement": agreement,
                "module_count_status": status,
                "imported_at": imported_at,
            }
        )
    return pd.DataFrame(rows)


def write_snapshot(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import module-count metadata for every configured battery serial.",
    )
    parser.add_argument("--profiles-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--module-count-col", default="bspi2_modulecount")
    parser.add_argument("--serial-col", default=DEFAULT_SERIAL_COL)
    parser.add_argument("--timestamp-col", default="time")
    parser.add_argument("--recent-samples", type=int, default=1_000)
    parser.add_argument("--minimum-agreement", type=float, default=0.95)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    if args.recent_samples <= 0:
        raise ValueError("recent-samples must be positive")
    if not 0 < args.minimum_agreement <= 1:
        raise ValueError("minimum-agreement must be greater than 0 and at most 1")

    serials = _configured_serials(args.profiles_file)
    observed = fetch_module_counts(
        db_config=DBConfig.from_env(args.env_file),
        serials=serials,
        table=args.table,
        serial_col=args.serial_col,
        module_count_col=args.module_count_col,
        timestamp_col=args.timestamp_col,
        recent_samples=args.recent_samples,
    )
    snapshot = build_snapshot(
        serials,
        observed,
        args.minimum_agreement,
    )
    write_snapshot(snapshot, args.output)

    verified = int((snapshot["module_count_status"] == "verified").sum())
    rejected = snapshot.loc[
        snapshot["module_count_status"] != "verified", "serial"
    ].tolist()
    logger.info("Module count verified for %s/%s serials. Snapshot: %s", verified, len(serials), args.output)
    if rejected:
        logger.error("Missing, invalid, or unstable module count for: %s", ", ".join(rejected))
    return 0 if not rejected else 2


if __name__ == "__main__":
    raise SystemExit(main())
