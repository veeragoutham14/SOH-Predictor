# Veera SOH Battery ML Pipeline

This project extracts high-volume battery log data from PostgreSQL/TimescaleDB and stores it locally as Parquet for later battery analytics work.

The first phase focuses on database extraction from `data_harvest.datalogger_hvb` into local Parquet files. The structure is ready to extend with preprocessing, feature engineering, mode-aware anomaly detection, capacity integration, SOH analysis, and degradation modeling.

## Why Parquet

Battery logs can contain millions of rows per month. CSV is slow to write, slow to read, large on disk, and loses useful typing information. Parquet is columnar, compressed, preserves schema better, and works well with pandas, pyarrow, Spark, DuckDB, and future ML pipelines.

## Install

Create and activate your Python environment, then install the required packages:

```powershell
pip install -r requirements.txt
```

Required packages:

- `psycopg2-binary` for PostgreSQL/TimescaleDB access
- `pandas` for chunk DataFrames
- `numpy` for capacity trend regression and forecast calculations
- `pyarrow` for Parquet writing
- `python-dotenv` for `.env` loading

## Configure

Copy `.env.example` to `.env` and fill in your database credentials:

```powershell
Copy-Item .env.example .env
```

Required variables:

```text
DB_HOST
DB_PORT
DB_NAME
DB_USER
DB_PASSWORD
```

Optional storage variables:

```text
DATA_DIR=data
RAW_PARQUET_DIR=data/raw_parquet
CLASSIFIED_PARQUET_DIR=data/processed/classified_rows
EVENT_PARQUET_DIR=data/processed/events
CAPACITY_TREND_DIR=data/processed/capacity_trend
CAPACITY_ML_DIR=data/processed/capacity_ml
```

Battery telemetry column names can also be configured:

```text
TIMESTAMP_COL=time
SERIAL_COL=serial
CURRENT_COL=bspi2_current_a
VOLTAGE_COL=bspi2_voltage_v
SOC_COL=bspi2_soc_pct
SOH_COL=bspi2_soh_pct
MODE_COL=operating_mode
EVENT_ID_COL=event_id
EXPECTED_SAMPLE_INTERVAL_SECONDS=1.0
MISSING_GAP_THRESHOLD_SECONDS=60.0
EXTRACTION_COLUMN_PROFILE=soh_core
REST_CURRENT_THRESHOLD_A=0.5
NOMINAL_CAPACITY_AH=50.0
BMS_SOH_RELIABLE_AFTER=2025-03-21T18:45:00+00:00
MIN_REASONABLE_CAPACITY_AH=30.0
MAX_REASONABLE_CAPACITY_AH=55.0
```

Never commit `.env` or extracted raw data.

## Run Extraction

Extract one battery serial over a time range:

```powershell
python -m src.extract_db --serial 309000009 --start 2026-04-01 --end 2026-05-01
```

Use a smaller chunk size while testing:

```powershell
python -m src.extract_db --serial 309000009 --start 2026-04-01 --end 2026-05-01 --chunk-size 25000 --limit 100000
```

Use a specific predefined extraction profile:

```powershell
python -m src.extract_db --serial 309000009 --start 2026-04-01 --end 2026-05-01 --column-profile soh_core
```

By default, extraction uses `EXTRACTION_COLUMN_PROFILE=soh_core` from `.env`.
That profile selects only the configured timestamp, serial, SOH, SOC, current,
and voltage columns. The extractor does not use `SELECT *` for this pipeline
unless the code is intentionally changed later.

## Inspect Initial Battery State

To print the earliest 10 rows for serial `300000172` with timestamp, SOH, SOC, current, and voltage:

```powershell
python -m src.inspect_initial_state
```

Equivalent explicit command:

```powershell
python -m src.inspect_initial_state --serial 300000172 --limit 10
```

The script prints a small table and an SOC-based summary:

- `SOC >= 95%`: fully charged or near full
- `SOC <= 5%`: empty or near empty
- otherwise: partially charged

Override the output directory:

```powershell
python -m src.extract_db --serial 309000009 --start 2026-04-01 --end 2026-05-01 --output-dir D:\battery_data\raw_parquet
```

## Output Layout

Parquet files are written in partition-style folders:

```text
data/raw_parquet/
  serial=309000009/
    year=2026/
      month=04/
        part-00000.parquet
        part-00001.parquet
```

Each `part-*.parquet` file corresponds to one fetched database chunk. The default extraction splits long time ranges month by month, which keeps queries and output organization manageable.

## Validate Raw Parquet Files

To inspect extracted raw Parquet files without loading every telemetry column:

```powershell
python -m src.validate_raw_parquet --serial 300000172
```

This prints a monthly table with file count, total rows, total size, first/last
timestamp, and missing `part-xxxxx.parquet` sequence gaps.

To also print file-level details:

```powershell
python -m src.validate_raw_parquet --serial 300000172 --details
```

If one month has far fewer files than neighboring months, compare `total_rows`,
`first_timestamp`, and `last_timestamp`. Fewer files are fine when the month is
only partially extracted; suspicious gaps show up in `missing_parts` or in a
timestamp range that does not cover the expected month.

To create an interactive SOC-over-time graph for a suspicious month:

```powershell
python -m src.plot_raw_signal --input data\raw_parquet\serial=300000172\year=2024\month=08 --output data\validation_plots\august_2024_soc.html --title "Serial 300000172 SOC - August 2024" --ylabel "SOC (%)"
```

The output is an interactive HTML graph with zoom, pan, hover, a range slider,
and red vertical lines for large timestamp gaps. By default the graph is
downsampled to keep the browser responsive; pass `--max-points 0` to plot every
row.

## Build Classified Rows And Events

After raw data has been extracted locally, build the local processing layers from Parquet. This stage does not query the database again.

For one serial:

```powershell
python -m src.build_events --serial 300000172
```

For a specific raw Parquet file or folder:

```powershell
python -m src.build_events --input data\raw_parquet\serial=300000172
```

With an optional time filter and current threshold:

```powershell
python -m src.build_events --serial 300000172 --start 2024-07-01 --end 2024-08-01 --rest-current-threshold-a 0.5
```

The mode classification is current-based:

```text
current > threshold   -> charging
current < -threshold  -> discharging
otherwise             -> rest
```

The classified row-level layer is written to:

```text
data/processed/classified_rows/
  serial=300000172/
    year=2024/
      month=07/
        classified_run=20260430T081500Z/
          part-00000-000.parquet
```

The event-level layer is written to:

```text
data/processed/events/
  serial=300000172/
    year=2024/
      month=07/
        event_run=20260430T081500Z/
          events.parquet
```

Each event is a contiguous block of rows with the same mode. The event table includes start/end timestamps, duration, row count, SOC/SOH summaries, mean current/voltage, positive Ah throughput, and positive Wh energy.

Event IDs include the event type:

```text
event_r_0        rest event
event_c_1        charging event
event_d_2        discharging event
event_missing_3  missing telemetry gap
```

Timestamp gaps larger than `MISSING_GAP_THRESHOLD_SECONDS` become explicit
`missing` events. These events have `num_rows = 0`, carry boundary SOC/SOH
values, and leave current/voltage/throughput/energy as unknown because the
battery behavior during the gap was not observed.

## Extraction Flow

1. Load database and storage configuration from `.env`.
2. Open a PostgreSQL connection using `psycopg2`.
3. Build a parameterized query for serial number, start time, end time, optional limit, and optional selected columns.
4. Use a server-side cursor and `fetchmany()` to stream rows in chunks.
5. Convert each chunk to a pandas DataFrame.
6. Normalize timestamp columns where present.
7. Write each chunk as Parquet using pyarrow.
8. Log progress and close database resources cleanly.

## Local Processing Flow

1. Load raw extracted Parquet from `data/raw_parquet`.
2. Classify each row as `charging`, `discharging`, or `rest`.
3. Store classified row-level Parquet in `data/processed/classified_rows`.
4. Segment consecutive rows with the same mode into events.
5. Store event-level Parquet in `data/processed/events`.
6. Build discharge capacity trend tables from the event layer.
7. Use those capacity trend tables for SOH/capacity analysis.

## Build Discharge Capacity Trend

After events are built, create the capacity trend layer:

```powershell
python -m src.discharge_cycle_builder --serial 300000172
```

This stage creates:

```text
data/processed/capacity_trend/
  serial=300000172/
    capacity_run=20260507T100000Z/
      discharge_usage_ledger.parquet
      discharge_episodes.parquet
      capacity_measurements.parquet
      capacity_trend.parquet
```

The usage ledger contains every observed discharging event, including partial
discharges. Its cumulative Ah/Wh columns act like a battery usage odometer.

Capacity measurements are built from trusted SOC anchors:

```text
trusted 100% SOC anchor
rest is allowed
sum only observed discharging Ah/Wh
rest is allowed
trusted 0/1% SOC anchor
```

This is intentional because intermediate SOC values can be unreliable, while
the 100% and 0% endpoints are treated as trusted voltage-defined anchors.

The trend table contains both:

```text
event_discharge_id                 boss-friendly Event 1, 2, ...
cumulative_all_discharge_ah/wh     physical usage axis for prediction
```

Future rows are scenario estimates for the next full discharge events. They
assume each future full discharge contributes roughly the latest measured
capacity to cumulative throughput.

## Build Capacity ML Forecast

After the capacity trend layer exists, build the machine-learning forecast layer:

```powershell
python -m src.capacity_ml --serial 300000172
```

This reads the latest `capacity_run` for the serial and writes:

```text
data/processed/capacity_ml/
  serial=300000172/
    ml_run=20260515T100000Z/
      ml_training_table.parquet
      model_summary.parquet
      model_summary.json
      capacity_forecast.parquet
      model_backtest.parquet
```

The ML target is measured capacity from the full anchor-to-anchor discharges:

```text
capacity_ah
capacity_wh
```

BMS SOH is not used as training truth. It is only marked as reliable for
comparison after:

```text
BMS_SOH_RELIABLE_AFTER=2025-03-21T18:45:00+00:00
```

The ML stage fits two kinds of models using NumPy.

First, it fits flexible historical-description models:

```text
capacity vs event_discharge_id
capacity vs cumulative_all_discharge_ah/wh
capacity vs sqrt(cumulative usage)
capacity vs log(cumulative usage)
capacity vs quadratic cumulative usage
capacity vs cumulative usage + calendar age
```

Second, it fits a degradation forecast model from the calculated capacity loss
after the reliable/stable period:

```text
capacity_loss_ah = learned_degradation_rate * cumulative_discharge_usage
predicted_capacity_ah = learned_baseline_capacity_ah - capacity_loss_ah
```

The flexible model is kept for comparison as
`statistical_predicted_capacity_ah`. The forecast-facing columns use the
degradation model:

```text
predicted_capacity_ah
predicted_future_soh_pct
```

The forecast table creates low, normal, and high future-usage scenarios. These
scenarios project future cumulative usage and then predict future capacity and
measured SOH:

```text
predicted_measured_soh_pct = predicted_capacity_ah / NOMINAL_CAPACITY_AH * 100
predicted_future_soh_pct   = same value, clearer forecast-facing name
```

Custom scenarios can be passed as multipliers of historical usage:

```powershell
python -m src.capacity_ml --serial 300000172 --scenarios low=0.5,normal=1.0,high=1.5
```

For a 10-year forecast, use `3650` forecast days:

```powershell
python -m src.capacity_ml --serial 300000172 --future-days 3650 --step-days 30
```

To generate an interactive usage and SOH forecast graph from the latest ML run:

```powershell
python -m src.plot_capacity_forecast --serial 300000172
```

This writes:

```text
data/validation_plots/capacity_usage_forecast_serial_300000172.html
```

The graph shows:

```text
historical cumulative discharged Ah
future cumulative discharged Ah for low/normal/high usage
measured historical capacity and SOH
predicted future capacity and SOH
backtest actual capacity vs predicted capacity
holdout prediction error in Ah and SOH %
```

The backtest hides the newest historical capacity measurements, trains on the
older measurements, and predicts the hidden ones. The important validation
columns are:

```text
actual_capacity_ah
predicted_capacity_ah
capacity_error_ah
actual_soh_pct
predicted_soh_pct
soh_error_pct
```

This tells you how wrong the model was on data it was not allowed to learn
from, which is the practical uncertainty check for the future SOH forecast.

## Extension Points

The next modules can be added beside the extractor:

- `src/preprocessing.py` for raw signal cleanup and schema normalization
- `src/feature_engineering.py` for electrical, thermal, rolling, and mode-aware features
- `src/mode_classification.py` for rest, charging, and discharging labels
- `src/event_segmentation.py` for charge/rest/discharge event summaries
- `src/anomaly_detection/` for Isolation Forest, PCA, Mahalanobis, and residual models
- `src/capacity.py` for current integration and usable capacity estimates
- `src/soh.py` for SOH metrics and degradation trends
