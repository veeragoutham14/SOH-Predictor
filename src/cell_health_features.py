from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from psycopg2 import sql

from src.cell_signal_audit import (
    CELL_TEMPERATURE_COLUMNS,
    GLOBAL_CELL_COLUMNS,
    _available_columns,
    _qualified_identifier,
)
from src.config import DBConfig, StorageConfig
from src.db import (
    DEFAULT_SERIAL_COL,
    DEFAULT_TABLE,
    DEFAULT_TIMESTAMP_COL,
    open_db_connection,
)
from src.io_utils import (
    iter_month_ranges,
    normalize_timestamp_columns,
    parse_timestamp,
    write_parquet_chunk,
)
from src.logging_utils import configure_logging

logger = logging.getLogger(__name__)

MIN_MODULE_COUNT = 1
MAX_MODULE_COUNT = 4
MODULE_COUNT_INTEGER_TOLERANCE = 0.001

FEATURE_COLUMNS = {
    "bspi2_soc_pct",
    "bspi2_soh_pct",
    "bspi2_current_a",
    "bspi2_voltage_v",
    "bspi2_power_w",
    "bspi2_modulecount",
    *GLOBAL_CELL_COLUMNS,
    *CELL_TEMPERATURE_COLUMNS,
}


def _valid_spread(
    minimum: str,
    maximum: str,
    *,
    scale: float,
    lower: float,
    upper: float,
) -> sql.Composed:
    return sql.SQL(
        """
        CASE
          WHEN {minimum} BETWEEN {lower} AND {upper}
           AND {maximum} BETWEEN {lower} AND {upper}
           AND {maximum} >= {minimum}
          THEN ({maximum} - {minimum}) * {scale}
        END
        """
    ).format(
        minimum=sql.Identifier(minimum),
        maximum=sql.Identifier(maximum),
        lower=sql.Literal(lower),
        upper=sql.Literal(upper),
        scale=sql.Literal(scale),
    )


def build_feature_query(
    *,
    table: str = DEFAULT_TABLE,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
    serial_col: str = DEFAULT_SERIAL_COL,
) -> sql.Composed:
    cell_spread = _valid_spread(
        "bspi2_cellvoltagemin_v",
        "bspi2_cellvoltagemax_v",
        scale=1000.0,
        lower=2.0,
        upper=5.0,
    )
    compensated_spread = _valid_spread(
        "bspi2_compensatedcellvoltagemin_v",
        "bspi2_compensatedcellvoltagemax_v",
        scale=1000.0,
        lower=2.0,
        upper=5.0,
    )
    temperature_spread = _valid_spread(
        "bspi2_celltemperaturemin_c",
        "bspi2_celltemperaturemax_c",
        scale=1.0,
        lower=-40.0,
        upper=100.0,
    )
    return sql.SQL(
        """
        WITH samples AS (
          SELECT
            date_bin(%s::interval, {timestamp}, TIMESTAMPTZ '2000-01-01 00:00:00+00') AS bucket,
            {serial_col} AS serial,
            {timestamp} AS source_timestamp,
            bspi2_soc_pct,
            bspi2_soh_pct,
            bspi2_current_a,
            bspi2_voltage_v,
            bspi2_power_w,
            CASE
              WHEN bspi2_modulecount BETWEEN {min_module_count} AND {max_module_count}
               AND ABS(bspi2_modulecount - ROUND(bspi2_modulecount))
                   <= {module_count_tolerance}
              THEN ROUND(bspi2_modulecount)::INTEGER
            END AS valid_module_count,
            CASE
              WHEN bspi2_modulecount IS NOT NULL
               AND NOT (
                 bspi2_modulecount BETWEEN {min_module_count} AND {max_module_count}
                 AND ABS(bspi2_modulecount - ROUND(bspi2_modulecount))
                     <= {module_count_tolerance}
               )
              THEN 1
              ELSE 0
            END AS invalid_module_count_sample,
            bspi2_cellvoltagemin_v,
            bspi2_cellvoltagemax_v,
            bspi2_compensatedcellvoltagemin_v,
            bspi2_compensatedcellvoltagemax_v,
            bspi2_celltemperaturemin_c,
            bspi2_celltemperaturemax_c,
            {cell_spread} AS cell_spread_mv,
            {compensated_spread} AS compensated_cell_spread_mv,
            {temperature_spread} AS cell_temperature_spread_c
          FROM {table}
          WHERE {serial_col} = %s
            AND {timestamp} >= %s
            AND {timestamp} < %s
        )
        SELECT
          bucket AS time,
          serial,
          MIN(source_timestamp) AS first_sample_time,
          MAX(source_timestamp) AS last_sample_time,
          COUNT(*) AS sample_count,
          COUNT(cell_spread_mv) AS valid_cell_voltage_sample_count,
          AVG(bspi2_soc_pct) AS soc_mean_pct,
          MIN(bspi2_soc_pct) AS soc_min_pct,
          MAX(bspi2_soc_pct) AS soc_max_pct,
          MIN(bspi2_soh_pct) AS soh_min_pct,
          MAX(bspi2_soh_pct) AS soh_max_pct,
          AVG(bspi2_current_a) AS current_mean_a,
          MIN(bspi2_current_a) AS current_min_a,
          MAX(bspi2_current_a) AS current_max_a,
          MAX(ABS(bspi2_current_a)) AS current_max_abs_a,
          AVG(bspi2_voltage_v) AS pack_voltage_mean_v,
          MIN(bspi2_voltage_v) AS pack_voltage_min_v,
          MAX(bspi2_voltage_v) AS pack_voltage_max_v,
          AVG(bspi2_power_w) AS power_mean_w,
          MIN(bspi2_power_w) AS power_min_w,
          MAX(bspi2_power_w) AS power_max_w,
          MODE() WITHIN GROUP (ORDER BY valid_module_count)
            FILTER (WHERE valid_module_count IS NOT NULL) AS module_count_bucket_mode,
          COUNT(valid_module_count) AS valid_module_count_sample_count,
          SUM(invalid_module_count_sample) AS invalid_module_count_sample_count,
          MIN(bspi2_cellvoltagemin_v) AS cell_voltage_min_v,
          MAX(bspi2_cellvoltagemax_v) AS cell_voltage_max_v,
          AVG(cell_spread_mv) AS cell_spread_mean_mv,
          PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cell_spread_mv)
            FILTER (WHERE cell_spread_mv IS NOT NULL) AS cell_spread_p95_mv,
          MAX(cell_spread_mv) AS cell_spread_max_mv,
          AVG(compensated_cell_spread_mv) AS compensated_cell_spread_mean_mv,
          PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY compensated_cell_spread_mv)
            FILTER (WHERE compensated_cell_spread_mv IS NOT NULL)
            AS compensated_cell_spread_p95_mv,
          MAX(compensated_cell_spread_mv) AS compensated_cell_spread_max_mv,
          MIN(bspi2_celltemperaturemin_c) AS cell_temperature_min_c,
          MAX(bspi2_celltemperaturemax_c) AS cell_temperature_max_c,
          MAX(cell_temperature_spread_c) AS cell_temperature_spread_max_c
        FROM samples
        GROUP BY bucket, serial
        ORDER BY bucket
        """
    ).format(
        timestamp=sql.Identifier(timestamp_col),
        serial_col=sql.Identifier(serial_col),
        table=_qualified_identifier(table),
        cell_spread=cell_spread,
        compensated_spread=compensated_spread,
        temperature_spread=temperature_spread,
        min_module_count=sql.Literal(MIN_MODULE_COUNT),
        max_module_count=sql.Literal(MAX_MODULE_COUNT),
        module_count_tolerance=sql.Literal(MODULE_COUNT_INTEGER_TOLERANCE),
    )


def add_module_configuration_epochs(features: pd.DataFrame) -> pd.DataFrame:
    """Clean module counts and label stable, day-level configuration epochs."""
    result = features.copy()
    timestamps = pd.to_datetime(result["time"], utc=True, errors="coerce")
    bucket_modes = pd.to_numeric(
        result.get("module_count_bucket_mode", result.get("module_count")),
        errors="coerce",
    )
    valid_bucket = (
        bucket_modes.between(MIN_MODULE_COUNT, MAX_MODULE_COUNT)
        & (bucket_modes - bucket_modes.round()).abs().le(MODULE_COUNT_INTEGER_TOLERANCE)
    )
    result["module_count_bucket_mode"] = bucket_modes.where(valid_bucket)
    result["_module_day"] = timestamps.dt.floor("D")

    weights = pd.to_numeric(
        result.get("valid_module_count_sample_count", result.get("sample_count")),
        errors="coerce",
    ).fillna(0.0)
    daily_source = pd.DataFrame(
        {
            "day": result["_module_day"],
            "module_count": result["module_count_bucket_mode"],
            "weight": weights,
        }
    ).dropna(subset=["day", "module_count"])
    if daily_source.empty:
        result["module_count"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        result["module_configuration_epoch"] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Int64",
        )
        return result.drop(columns=["_module_day"])

    daily_weights = (
        daily_source.groupby(["day", "module_count"], as_index=False)["weight"]
        .sum()
        .sort_values(["day", "weight", "module_count"], ascending=[True, False, True])
    )
    daily_modes = daily_weights.drop_duplicates("day").sort_values("day").copy()
    daily_modes["module_count"] = daily_modes["module_count"].astype(int)

    run_id = daily_modes["module_count"].ne(daily_modes["module_count"].shift()).cumsum()
    run_days = daily_modes.groupby(run_id)["day"].transform("size")
    short_runs = run_days.lt(2)
    previous = daily_modes["module_count"].shift()
    following = daily_modes["module_count"].shift(-1)
    merge_between_same = short_runs & previous.eq(following) & previous.notna()
    daily_modes.loc[merge_between_same, "module_count"] = previous[merge_between_same]

    daily_modes["module_configuration_epoch"] = (
        daily_modes["module_count"].ne(daily_modes["module_count"].shift()).cumsum()
    )
    day_to_count = daily_modes.set_index("day")["module_count"]
    day_to_epoch = daily_modes.set_index("day")["module_configuration_epoch"]
    result["module_count"] = result["_module_day"].map(day_to_count).astype("Int64")
    result["module_configuration_epoch"] = (
        result["_module_day"].map(day_to_epoch).astype("Int64")
    )
    return result.drop(columns=["_module_day"])


def module_configuration_epochs(features: pd.DataFrame) -> list[dict[str, Any]]:
    required = {"module_configuration_epoch", "module_count", "time", "sample_count"}
    if not required.issubset(features.columns):
        return []
    valid = features.dropna(subset=["module_configuration_epoch", "module_count", "time"])
    epochs: list[dict[str, Any]] = []
    for epoch_id, frame in valid.groupby("module_configuration_epoch", sort=True):
        ordered = frame.sort_values("time")
        epochs.append(
            {
                "epoch": int(epoch_id),
                "module_count": int(ordered["module_count"].mode().iloc[0]),
                "start": pd.Timestamp(ordered["time"].iloc[0]).isoformat(),
                "end": pd.Timestamp(ordered["last_sample_time"].iloc[-1]).isoformat(),
                "feature_rows": int(len(ordered)),
                "source_sample_rows": int(ordered["sample_count"].sum()),
            }
        )
    return epochs


def extract_feature_range(
    connection: Any,
    *,
    serial: str,
    start: datetime,
    end: datetime,
    aggregation_minutes: int,
    table: str = DEFAULT_TABLE,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
    serial_col: str = DEFAULT_SERIAL_COL,
) -> pd.DataFrame:
    if start >= end:
        raise ValueError("start must be earlier than end")
    if aggregation_minutes <= 0:
        raise ValueError("aggregation_minutes must be positive")
    available = _available_columns(connection, table)
    missing = sorted(FEATURE_COLUMNS - available)
    if missing:
        raise ValueError(
            "Database table is missing required cell-health signals: "
            f"{', '.join(missing)}"
        )
    query = build_feature_query(
        table=table,
        timestamp_col=timestamp_col,
        serial_col=serial_col,
    )
    interval = f"{aggregation_minutes} minutes"
    frames: list[pd.DataFrame] = []
    for interval_start, interval_end in iter_month_ranges(start, end):
        logger.info(
            "Building %s-minute cell-health features for serial=%s, %s to %s",
            aggregation_minutes,
            serial,
            interval_start.isoformat(),
            interval_end.isoformat(),
        )
        with connection.cursor() as cursor:
            cursor.execute(query, (interval, serial, interval_start, interval_end))
            rows = cursor.fetchall()
            if not rows:
                continue
            names = [column[0] for column in cursor.description]
        frames.append(pd.DataFrame.from_records(rows, columns=names))
    if not frames:
        raise ValueError(
            f"No cell-health data found for serial {serial} in the requested range"
        )
    normalized = normalize_timestamp_columns(
        pd.concat(frames, ignore_index=True),
        timestamp_columns=("time", "first_sample_time", "last_sample_time"),
    )
    return add_module_configuration_epochs(normalized)


def summarize_features(
    features: pd.DataFrame,
    *,
    serial: str,
    start: datetime,
    end: datetime,
    aggregation_minutes: int,
) -> dict[str, Any]:
    def finite_max(column: str) -> float | None:
        values = pd.to_numeric(features[column], errors="coerce").dropna()
        return round(float(values.max()), 6) if not values.empty else None

    total_samples = int(features["sample_count"].sum())
    valid_samples = int(features["valid_cell_voltage_sample_count"].sum())
    epochs = module_configuration_epochs(features)
    module_values = pd.to_numeric(features["module_count"], errors="coerce").dropna()
    dominant_module_count = (
        int(module_values.mode().iloc[0]) if not module_values.empty else None
    )
    latest_module_count = int(module_values.iloc[-1]) if not module_values.empty else None
    valid_module_samples = int(
        pd.to_numeric(
            features.get("valid_module_count_sample_count", pd.Series(dtype=float)),
            errors="coerce",
        ).sum()
    )
    invalid_module_samples = int(
        pd.to_numeric(
            features.get("invalid_module_count_sample_count", pd.Series(dtype=float)),
            errors="coerce",
        ).sum()
    )
    module_sample_total = valid_module_samples + invalid_module_samples
    return {
        "model_family": "cell_health_features",
        "serial": serial,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "aggregation_minutes": aggregation_minutes,
        "feature_rows": int(len(features)),
        "source_sample_rows": total_samples,
        "valid_cell_voltage_sample_rows": valid_samples,
        "valid_cell_voltage_sample_pct": (
            100.0 * valid_samples / total_samples if total_samples else 0.0
        ),
        "observed_module_count": latest_module_count,
        "dominant_module_count": dominant_module_count,
        "latest_module_count": latest_module_count,
        "module_configuration_changed": len(epochs) > 1,
        "module_configuration_epochs": epochs,
        "valid_module_count_sample_rows": valid_module_samples,
        "invalid_module_count_sample_rows": invalid_module_samples,
        "invalid_module_count_sample_pct": (
            100.0 * invalid_module_samples / module_sample_total
            if module_sample_total
            else 0.0
        ),
        "maximum_cell_spread_mv": finite_max("cell_spread_max_mv"),
        "maximum_compensated_cell_spread_mv": finite_max(
            "compensated_cell_spread_max_mv"
        ),
        "maximum_cell_temperature_spread_c": finite_max(
            "cell_temperature_spread_max_c"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build compact, server-aggregated cell-health features."
    )
    parser.add_argument("--serial", required=True)
    parser.add_argument("--start", required=True, type=parse_timestamp)
    parser.add_argument("--end", required=True, type=parse_timestamp)
    parser.add_argument("--aggregation-minutes", type=int, default=5)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--timestamp-col", default=DEFAULT_TIMESTAMP_COL)
    parser.add_argument("--serial-col", default=DEFAULT_SERIAL_COL)
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=("snappy", "zstd", "gzip", "none"),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        db_config = DBConfig.from_env(args.env_file)
        storage = StorageConfig.from_env(args.env_file)
        output_root = args.output_root or storage.data_dir / "processed" / "cell_health"
        with open_db_connection(db_config) as connection:
            features = extract_feature_range(
                connection,
                serial=str(args.serial),
                start=args.start,
                end=args.end,
                aggregation_minutes=args.aggregation_minutes,
                table=args.table,
                timestamp_col=args.timestamp_col,
                serial_col=args.serial_col,
            )
        summary = summarize_features(
            features,
            serial=str(args.serial),
            start=args.start,
            end=args.end,
            aggregation_minutes=args.aggregation_minutes,
        )
        run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = output_root / f"serial={args.serial}" / f"feature_run={run_stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        compression = None if args.compression == "none" else args.compression
        write_parquet_chunk(
            features,
            run_dir / "cell_health_features.parquet",
            compression=compression,
        )
        (run_dir / "model_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Cell-health features complete for serial=%s: %s rows from %s source samples; %s",
            args.serial,
            f"{len(features):,}",
            f"{summary['source_sample_rows']:,}",
            run_dir,
        )
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Cell-health feature extraction failed")
        else:
            logger.error("Cell-health feature extraction failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
