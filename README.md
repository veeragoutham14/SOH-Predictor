# Veera SOH Battery ML Pipeline

This project extracts high-volume battery telemetry from PostgreSQL/TimescaleDB, stores it locally as Parquet, and turns the raw second-level logs into operating-mode events, full-discharge capacity measurements, validated SOH trend models, and 10-year capacity forecasts.

The pipeline is designed as a layered local-processing workflow. The database is used only for raw extraction; downstream stages read local Parquet layers. This makes the workflow reusable across serial numbers, scalable to large monthly extracts, and traceable from raw telemetry to forecast outputs.

Core pipeline:

```text
TimescaleDB telemetry
-> raw Parquet extraction
-> row-level operating mode classification
-> event segmentation including missing telemetry gaps
-> discharge usage ledger and full 100%-to-0% capacity measurements
-> usage-based capacity/SOH degradation model
-> cutoff validation against hidden future data
-> 10-year SOH forecast with 80% threshold and uncertainty estimate
```

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
- `plotly` for interactive HTML validation and forecast graphs
- `matplotlib` for Confluence-ready static PNG reports

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
# Optional. Leave empty to use capacity_ah from event_discharge_id 1.
NOMINAL_CAPACITY_AH=
BMS_SOH_RELIABLE_AFTER=2025-03-21T18:45:00+00:00
MIN_REASONABLE_CAPACITY_AH=30.0
MAX_REASONABLE_CAPACITY_AH=55.0
USAGE_RATE_MODES=historical_mean,historical_median,recent_median
USAGE_RATE_MODE=historical_mean
RECENT_USAGE_DAYS=90
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
trusted 0% SOC anchor
```

This is intentional because intermediate SOC values can be unreliable, while
the 100% and 0% endpoints are treated as trusted voltage-defined anchors.

The trend table contains both:

```text
event_discharge_id                 boss-friendly Event 1, 2, ...
cumulative_all_discharge_ah/wh     physical usage axis for prediction
```

The current implementation accepts only exact `100%` start anchors and exact
`0%` end anchors for measured capacity rows. Near-full or near-empty events are
still useful in the usage ledger, but they are not treated as complete capacity
measurements.

## Build Capacity ML Forecast

After the capacity trend layer exists, build the capacity ML forecast layer:

```powershell
python -m src.capacity_ml --serial 300000172
```

For the current validation and 10-year forecast workflow:

```powershell
python -m src.capacity_ml --serial 300000172 --degradation-model usage_linear --training-cutoff "2025-12-31T23:59:59+00:00" --validation-end "2026-05-31T23:59:59+00:00" --future-days 3650 --step-days 30
```

This reads the latest `capacity_run` for the serial and writes one timestamped
ML run:

```text
data/processed/capacity_ml/
  serial=300000172/
    ml_run=20260611T134511Z/
      ml_training_table.parquet
      ml_all_capacity_rows.parquet
      model_summary.parquet
      degradation_model_summary.parquet
      model_summary.json
      capacity_forecast.parquet
      model_backtest.parquet
      cutoff_validation.parquet
      cutoff_forecast_validation.parquet
```

The target is measured capacity from exact full anchor-to-anchor discharges:

```text
capacity_ah
capacity_wh
```

BMS SOH is not used as training truth. The model learns from measured
`capacity_ah`. BMS SOH is kept only for comparison after the configured
reliable point:

```text
BMS_SOH_RELIABLE_AFTER=2025-03-21T18:45:00+00:00
```

If `NOMINAL_CAPACITY_AH` is left empty, nominal capacity is inferred from the
measured `capacity_ah` of `event_discharge_id == 1`. This keeps SOH relative to
the first complete measured full-discharge capacity rather than a hardcoded
nameplate value.

The ML stage fits two families of models using NumPy.

First, it fits flexible statistical capacity models:

```text
capacity vs event_discharge_id
capacity vs cumulative_all_discharge_ah/wh
capacity vs sqrt(cumulative usage)
capacity vs log(cumulative usage)
capacity vs quadratic cumulative usage
capacity vs cumulative usage + calendar age
```

These are useful for comparison and validation. The selected model is stored in
`best_capacity_ah_model` and its prediction is stored as
`statistical_predicted_capacity_ah`.

Second, it fits the forecast-facing degradation model:

```text
capacity_loss_ah = intercept_loss_ah
                 + loss_slope_per_1000ah * cumulative_usage_since_reference / 1000

predicted_capacity_ah = baseline_capacity_ah - capacity_loss_ah
predicted_soh_pct = predicted_capacity_ah / nominal_capacity_ah * 100
```

The forecast-facing columns use this degradation model:

```text
predicted_capacity_ah
predicted_measured_soh_pct
predicted_future_soh_pct
```

### Usage-Rate Models

The current default forecast compares three ways of estimating future daily
discharge usage from the training-period usage ledger:

```text
historical_mean     average Ah/day over the training history
historical_median   median daily Ah over the full training history
recent_median       median daily Ah over the recent lookback window
```

The usage ledger includes every observed discharge, including partial
discharges, and daily usage is allocated by interval overlap. If a discharge
crosses midnight, its Ah/Wh is split across the affected calendar days by
duration.

The selected usage-rate models are controlled by:

```text
USAGE_RATE_MODES=historical_mean,historical_median,recent_median
RECENT_USAGE_DAYS=90
```

You can still pass custom scenario multipliers when needed:

```powershell
python -m src.capacity_ml --serial 300000172 --scenarios low=0.7,normal=1.0,high=1.3
```

### Validation Logic

There are two validation outputs:

```text
cutoff_validation.parquet
cutoff_forecast_validation.parquet
```

`cutoff_validation.parquet` validates the degradation model against hidden
post-cutoff capacity rows using their actual cumulative Ah. It answers:

```text
If actual future usage were known, how well does the degradation model predict capacity?
```

`cutoff_forecast_validation.parquet` validates the full future-forecast logic.
It uses only the training-period usage estimate to predict future cumulative Ah,
then compares the predicted Jan-May 2026 capacity/SOH against the actual hidden
Jan-May 2026 measurements. It answers:

```text
If we stopped learning at the cutoff, which usage-rate model best predicts the hidden future period?
```

When a training cutoff is provided, forecast validation is anchored at the
cutoff using known training-period cumulative Ah from the discharge usage
ledger:

```text
forecast_anchor_time = training_cutoff
forecast_anchor_source = usage_ledger_training_cutoff
```

This avoids leaking validation data while also avoiding an unfair forecast from
the last capacity event if the last event is earlier than the cutoff.

The forecast validation also estimates usage-rate uncertainty. It fits:

```text
cumulative_Ah_error = intercept + daily_Ah_error_slope * days_after_cutoff
```

This separates an initial offset from ongoing daily drift. The drift slope is
then projected to the forecasted 80% SOH point and converted into:

```text
projected_soh_error_pct_slope
estimated 80% date uncertainty
```

### Forecast Outputs

`capacity_forecast.parquet` contains one forecast curve per usage-rate model.
Important columns include:

```text
usage_rate_model
forecast_ah_per_day
forecast_timestamp
cumulative_all_discharge_ah
equivalent_full_cycles
predicted_capacity_ah
predicted_future_soh_pct
projected_soh_error_pct_slope
projected_soh_error_pct_final
```

The `equivalent_full_cycles` column is calculated from cumulative discharged Ah
divided by nominal capacity. It is used as a warranty-friendly cycle-equivalent
axis and includes the effect of partial discharges through cumulative Ah.

### Plot Forecast Results

Generate an interactive HTML forecast graph from the latest ML run:

```powershell
python -m src.plot_capacity_forecast --serial 300000172
```

This writes:

```text
data/validation_plots/capacity_usage_forecast_serial_300000172.html
```

The HTML graph shows historical capacity measurements, predicted capacity/SOH,
future cumulative Ah, hover details, and uncertainty values for each usage-rate
model.

Generate a Confluence-ready static PNG report:

```powershell
python -m src.plot_capacity_forecast_report_image --serial 300000172
```

This writes:

```text
data/validation_plots/capacity_forecast_report_serial_300000172_ml_run_<run_id>.png
```

The PNG report summarizes:

```text
training cutoff
validation period
degradation model and loss slope
forecast usage-rate models
validation MAE and max SOH error
80% SOH crossing month
80% SOH uncertainty
80% date uncertainty
80% equivalent full cycles
10-year predicted SOH
```

The final report is intended to connect raw telemetry, measured capacity,
validated degradation behavior, future usage assumptions, and the predicted
80% SOH threshold in one auditable output.

## Extension Points

The next modules can be added beside the extractor:

- `src/preprocessing.py` for raw signal cleanup and schema normalization
- `src/feature_engineering.py` for electrical, thermal, rolling, and mode-aware features
- `src/mode_classification.py` for rest, charging, and discharging labels
- `src/event_segmentation.py` for charge/rest/discharge event summaries
- `src/anomaly_detection/` for Isolation Forest, PCA, Mahalanobis, and residual models
- `src/capacity.py` for current integration and usable capacity estimates
- `src/soh.py` for SOH metrics and degradation trends
