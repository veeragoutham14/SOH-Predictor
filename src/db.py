from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import connection as PsycopgConnection
from psycopg2.extensions import cursor as PsycopgCursor

from src.config import DBConfig


DEFAULT_TABLE = "data_harvest.datalogger_hvb"
DEFAULT_TIMESTAMP_COL = "time"
DEFAULT_SERIAL_COL = "serial"


@contextmanager
def open_db_connection(config: DBConfig) -> Iterator[PsycopgConnection]:
    """Open and close a PostgreSQL connection with useful error context."""
    conn: PsycopgConnection | None = None
    try:
        conn = psycopg2.connect(
            host=config.host,
            port=config.port,
            dbname=config.database,
            user=config.user,
            password=config.password,
            connect_timeout=config.connect_timeout_s,
            application_name=config.application_name,
        )
        conn.set_session(readonly=True, autocommit=False)
        yield conn
    except psycopg2.Error as exc:
        raise RuntimeError(f"Database operation failed: {exc}") from exc
    finally:
        if conn is not None:
            conn.close()


def _qualified_identifier(name: str) -> sql.Composed:
    parts = [part.strip() for part in name.split(".")]
    if not parts or any(part == "" for part in parts):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return sql.SQL(".").join(sql.Identifier(part) for part in parts)


def _columns_sql(columns: Sequence[str] | None) -> sql.SQL | sql.Composed:
    if columns is None:
        return sql.SQL("*")
    if len(columns) == 0:
        raise ValueError("columns cannot be an empty sequence.")
    return sql.SQL(", ").join(sql.Identifier(column) for column in columns)


def build_extract_query(
    *,
    serial: str | int,
    start: Any,
    end: Any,
    limit: int | None = None,
    columns: Sequence[str] | None = None,
    table: str = DEFAULT_TABLE,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
    serial_col: str = DEFAULT_SERIAL_COL,
) -> tuple[sql.Composed, list[Any]]:
    """Build a parameterized extraction query for one serial and time range."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer when provided.")

    query = sql.SQL(
        """
        SELECT {columns}
        FROM {table}
        WHERE {serial_col} = %s
          AND {timestamp_col} >= %s
          AND {timestamp_col} < %s
        ORDER BY {timestamp_col} ASC
        """
    ).format(
        columns=_columns_sql(columns),
        table=_qualified_identifier(table),
        serial_col=sql.Identifier(serial_col),
        timestamp_col=sql.Identifier(timestamp_col),
    )

    params: list[Any] = [serial, start, end]
    if limit is not None:
        query += sql.SQL(" LIMIT %s")
        params.append(limit)

    return query, params


def cursor_column_names(cursor: PsycopgCursor) -> list[str]:
    if cursor.description is None:
        raise RuntimeError("Cursor has no result description. Was a SELECT query executed?")
    return [column[0] for column in cursor.description]
