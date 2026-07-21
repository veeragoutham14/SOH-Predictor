from __future__ import annotations

import argparse
import ast
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from psycopg2 import sql

from src.config import DBConfig
from src.db import (
    DEFAULT_SERIAL_COL,
    DEFAULT_TABLE,
    DEFAULT_TIMESTAMP_COL,
    _qualified_identifier,
    open_db_connection,
)


DEFAULT_MODULE_COLUMN = "batterystackprocimage_batterymodules"


def _default_raw_dir() -> Path:
    return Path(os.environ.get("RAW_PARQUET_DIR", "data/raw_parquet"))


def _iter_parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".parquet" else []
    return sorted(path.rglob("*.parquet"))


def _schema_names(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema.names)


def _count_from_parsed(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, dict):
        for key in ("batteryModules", "modules", "items", "values"):
            nested = value.get(key)
            if isinstance(nested, (list, tuple)):
                return len(nested)
        return len(value) if value else None
    if isinstance(value, (int, float)) and pd.notna(value):
        as_float = float(value)
        if as_float.is_integer() and 0 < as_float < 1000:
            return int(as_float)
    return None


def infer_module_count(value: Any) -> int | None:
    """Infer module count from text/JSON-like batteryModules values."""
    direct_count = _count_from_parsed(value)
    if direct_count is not None:
        return direct_count

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]", "{}"}:
        return None

    if re.fullmatch(r"\d+", text):
        count = int(text)
        return count if 0 < count < 1000 else None

    module_count_match = re.search(
        r"(?:battery\s*)?modules?\D{0,40}(\d{1,3})|(\d{1,3})\D{0,40}(?:battery\s*)?modules?",
        text,
        flags=re.IGNORECASE,
    )
    if module_count_match:
        count = int(next(group for group in module_count_match.groups() if group))
        if 0 < count < 1000:
            return count

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
        parsed_count = _count_from_parsed(parsed)
        if parsed_count is not None:
            return parsed_count

    object_markers = len(re.findall(r"\{[^{}]*\}", text))
    if object_markers:
        return object_markers

    for separator in ("|", ";"):
        if separator in text:
            parts = [part.strip() for part in text.split(separator) if part.strip()]
            return len(parts) if parts else None

    if "," in text:
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if len(parts) > 1:
            return len(parts)

    return None


def inspect_modules(
    input_path: Path,
    *,
    serial: str | None,
    module_column: str,
    max_examples: int,
) -> int:
    files = _iter_parquet_files(input_path)
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {input_path}")

    counts: Counter[int] = Counter()
    examples: list[str] = []
    scanned_files = 0
    files_with_column = 0
    non_null_rows = 0

    for parquet_file in files:
        names = _schema_names(parquet_file)
        if module_column not in names:
            continue

        columns = [module_column]
        if serial and "serial" in names:
            columns.append("serial")

        frame = pd.read_parquet(parquet_file, columns=columns)
        scanned_files += 1
        files_with_column += 1

        if serial and "serial" in frame.columns:
            frame = frame[frame["serial"].astype(str) == str(serial)]

        values = frame[module_column].dropna()
        non_null_rows += int(len(values))

        for value in values:
            count = infer_module_count(value)
            if count is not None:
                counts[count] += 1
                if len(examples) < max_examples:
                    examples.append(str(value)[:500])

    print(f"Input path          : {input_path}")
    if serial:
        print(f"Serial              : {serial}")
    print(f"Module column       : {module_column}")
    print(f"Parquet files found : {len(files)}")
    print(f"Files with column   : {files_with_column}")
    print(f"Non-null rows       : {non_null_rows}")

    if not counts:
        print("Module count        : not found")
        return 1

    print("Observed counts     :")
    for count, rows in counts.most_common():
        print(f"  {count} modules -> {rows} rows")

    best_count, best_rows = counts.most_common(1)[0]
    print(f"Most likely count   : {best_count} modules ({best_rows} rows)")

    if examples:
        print("\nExample value:")
        print(examples[0])

    return 0


def inspect_modules_from_db(
    *,
    serial: str | int,
    since: str,
    limit: int,
    module_column: str,
    table: str,
    timestamp_col: str,
    serial_col: str,
    env_file: str | Path | None,
    max_examples: int,
) -> int:
    if limit <= 0:
        raise ValueError("limit must be positive.")

    db_config = DBConfig.from_env(env_file)
    query = sql.SQL(
        """
        SELECT {module_column}
        FROM {table}
        WHERE {serial_col} = %s
          AND {timestamp_col} > %s
          AND {module_column} IS NOT NULL
        ORDER BY {timestamp_col} ASC
        LIMIT %s
        """
    ).format(
        module_column=sql.Identifier(module_column),
        table=_qualified_identifier(table),
        serial_col=sql.Identifier(serial_col),
        timestamp_col=sql.Identifier(timestamp_col),
    )

    counts: Counter[int] = Counter()
    examples: list[str] = []
    rows_seen = 0

    with open_db_connection(db_config) as conn:
        with conn.cursor() as cur:
            cur.execute(query, [serial, since, limit])
            for (value,) in cur.fetchall():
                rows_seen += 1
                count = infer_module_count(value)
                if count is None:
                    continue
                counts[count] += 1
                if len(examples) < max_examples:
                    examples.append(str(value)[:500])

    print(f"Source              : TimescaleDB")
    print(f"Table               : {table}")
    print(f"Serial              : {serial}")
    print(f"Since               : {since}")
    print(f"Module column       : {module_column}")
    print(f"Rows fetched        : {rows_seen}")

    if not counts:
        print("Module count        : not found")
        return 1

    print("Observed counts     :")
    for count, rows in counts.most_common():
        print(f"  {count} modules -> {rows} rows")

    best_count, best_rows = counts.most_common(1)[0]
    print(f"Most likely count   : {best_count} modules ({best_rows} rows)")

    if examples:
        print("\nExample value:")
        print(examples[0])

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect raw parquet batteryModules values and infer module count."
    )
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Query TimescaleDB/PostgreSQL instead of reading local parquet.",
    )
    parser.add_argument("--env-file", default=None, help="Optional .env file path.")
    parser.add_argument("--since", default="1970-01-01", help="Timestamp lower bound for DB mode.")
    parser.add_argument("--limit", type=int, default=10_000, help="Maximum DB rows to inspect.")
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--timestamp-col", default=DEFAULT_TIMESTAMP_COL)
    parser.add_argument("--serial-col", default=DEFAULT_SERIAL_COL)
    parser.add_argument("--serial", default=None, help="Battery serial, e.g. 300000083.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Raw parquet file/folder. Defaults to data/raw_parquet/serial=<serial>.",
    )
    parser.add_argument("--column", default=DEFAULT_MODULE_COLUMN)
    parser.add_argument("--max-examples", type=int, default=1)
    args = parser.parse_args()

    if args.from_db:
        if not args.serial:
            parser.error("--serial is required with --from-db.")
        return inspect_modules_from_db(
            serial=args.serial,
            since=args.since,
            limit=args.limit,
            module_column=args.column,
            table=args.table,
            timestamp_col=args.timestamp_col,
            serial_col=args.serial_col,
            env_file=args.env_file,
            max_examples=max(0, args.max_examples),
        )

    input_path = args.input
    if input_path is None:
        if not args.serial:
            parser.error("--serial is required when --input is not provided.")
        input_path = _default_raw_dir() / f"serial={args.serial}"

    return inspect_modules(
        input_path,
        serial=str(args.serial) if args.serial else None,
        module_column=args.column,
        max_examples=max(0, args.max_examples),
    )


if __name__ == "__main__":
    raise SystemExit(main())
