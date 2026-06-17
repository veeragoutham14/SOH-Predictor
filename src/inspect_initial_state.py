from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from psycopg2 import sql

from src.config import DBConfig
from src.db import (
    DEFAULT_SERIAL_COL,
    DEFAULT_TABLE,
    DEFAULT_TIMESTAMP_COL,
    cursor_column_names,
    open_db_connection,
)
from src.logging_utils import configure_logging


logger = logging.getLogger(__name__)

INITIAL_STATE_COLUMNS = (
    "time",
    "bspi2_soh_pct",
    "bspi2_soc_pct",
    "bspi2_current_a",
    "bspi2_voltage_v",
)

DISPLAY_NAMES = {
    "time": "timestamp",
    "bspi2_soh_pct": "SOH_pct",
    "bspi2_soc_pct": "SOC_pct",
    "bspi2_current_a": "current_A",
    "bspi2_voltage_v": "voltage_V",
}


def _qualified_identifier(name: str) -> sql.Composed:
    parts = [part.strip() for part in name.split(".")]
    if not parts or any(part == "" for part in parts):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return sql.SQL(".").join(sql.Identifier(part) for part in parts)


def build_initial_state_query(
    *,
    table: str,
    serial_col: str,
    timestamp_col: str,
) -> sql.Composed:
    columns_sql = sql.SQL(", ").join(sql.Identifier(column) for column in INITIAL_STATE_COLUMNS)
    return sql.SQL(
        """
        SELECT {columns}
        FROM {table}
        WHERE {serial_col} = %s
        ORDER BY {timestamp_col} ASC
        LIMIT %s
        """
    ).format(
        columns=columns_sql,
        table=_qualified_identifier(table),
        serial_col=sql.Identifier(serial_col),
        timestamp_col=sql.Identifier(timestamp_col),
    )


def fetch_initial_state_rows(
    *,
    serial: int,
    limit: int,
    db_config: DBConfig,
    table: str = DEFAULT_TABLE,
    serial_col: str = DEFAULT_SERIAL_COL,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
) -> pd.DataFrame:
    if limit <= 0:
        raise ValueError("limit must be positive.")

    query = build_initial_state_query(
        table=table,
        serial_col=serial_col,
        timestamp_col=timestamp_col,
    )

    logger.info("Fetching first %s rows for serial=%s", limit, serial)
    with open_db_connection(db_config) as conn:
        with conn.cursor() as cur:
            cur.execute(query, [serial, limit])
            rows = cur.fetchall()
            columns = cursor_column_names(cur)

    df = pd.DataFrame.from_records(rows, columns=columns)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df


def classify_soc(
    soc_pct: Any,
    *,
    empty_threshold: float = 5.0,
    full_threshold: float = 95.0,
) -> str:
    if pd.isna(soc_pct):
        return "unknown because SOC is missing"

    soc = float(soc_pct)
    if soc >= full_threshold:
        return "fully charged or near full"
    if soc <= empty_threshold:
        return "empty or near empty"
    return "partially charged"


def _first_available(df: pd.DataFrame, column: str) -> tuple[Any, Any]:
    available = df.loc[df[column].notna(), ["time", column]]
    if available.empty:
        return None, None
    first = available.iloc[0]
    return first[column], first["time"]


def print_initial_state(df: pd.DataFrame, *, empty_threshold: float, full_threshold: float) -> None:
    if df.empty:
        print("No rows found for this serial number.")
        return

    display = df.rename(columns=DISPLAY_NAMES).copy()
    for column in ("SOH_pct", "SOC_pct", "current_A", "voltage_V"):
        if column in display.columns:
            display[column] = pd.to_numeric(display[column], errors="coerce").round(3)

    print("\nEarliest returned rows:")
    print(display.to_string(index=False))

    soh, soh_time = _first_available(df, "bspi2_soh_pct")
    soc, soc_time = _first_available(df, "bspi2_soc_pct")
    state = classify_soc(
        soc,
        empty_threshold=empty_threshold,
        full_threshold=full_threshold,
    )

    print("\nInitial state summary:")
    print(f"- Earliest timestamp returned: {df.iloc[0]['time']}")
    print(f"- First available SOH: {soh if soh is not None else 'missing'} at {soh_time}")
    print(f"- First available SOC: {soc if soc is not None else 'missing'} at {soc_time}")
    print(f"- SOC-based state: {state}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print the earliest SOH/SOC/current/voltage rows for one battery serial.",
    )
    parser.add_argument("--serial", type=int, default=300000172, help="Battery serial number.")
    parser.add_argument("--limit", type=int, default=10, help="Number of earliest rows to print.")
    parser.add_argument("--env-file", type=Path, default=None, help="Path to a .env file.")
    parser.add_argument("--table", default=DEFAULT_TABLE, help="Source table, optionally schema qualified.")
    parser.add_argument("--serial-col", default=DEFAULT_SERIAL_COL, help="Serial number column.")
    parser.add_argument("--timestamp-col", default=DEFAULT_TIMESTAMP_COL, help="Timestamp column.")
    parser.add_argument("--empty-threshold", type=float, default=5.0, help="SOC percent treated as empty.")
    parser.add_argument("--full-threshold", type=float, default=95.0, help="SOC percent treated as full.")
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
        df = fetch_initial_state_rows(
            serial=args.serial,
            limit=args.limit,
            db_config=db_config,
            table=args.table,
            serial_col=args.serial_col,
            timestamp_col=args.timestamp_col,
        )
        print_initial_state(
            df,
            empty_threshold=args.empty_threshold,
            full_threshold=args.full_threshold,
        )
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Initial-state inspection failed.")
        else:
            logger.error("Initial-state inspection failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
