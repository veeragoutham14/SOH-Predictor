import psycopg2

# --- connection config ---
conn = psycopg2.connect(
    host="10.0.215.136", port=5432, database="storage_data", user="vgouthama", password="LS7{R85b8/og"
)

# query = """
# SELECT
#     MIN(time) AS first_timestamp,
#     MAX(time) AS last_timestamp,
#     COUNT(*) AS total_rows
# FROM data_harvest.datalogger_hvb
# WHERE serial = %s
#   AND time >= %s
#   AND time < %s;
# """
#
# params = (
#     300000172,
#     "2025-03-20T11:37:25+00:00",
#     "2025-03-20T11:37:59+00:00",
# )
#
# with conn.cursor() as cur:
#     cur.execute(query, params)
#     row = cur.fetchone()
#     print("First timestamp:", row[0])
#     print("Last timestamp:", row[1])
#     print("Total rows:", row[2])

# query = """
# SELECT
#     time,
#     bspi2_soc_pct,
#     bspi2_soh_pct
# FROM data_harvest.datalogger_hvb
# WHERE serial = %s
#   AND time = %s
# """
#
# params = (
#     300000172,
#     "2024-07-17T22:30:00+00:00",
# )
#
# with conn.cursor() as cur:
#     cur.execute(query, params)
#     rows = cur.fetchall()
#
#     if not rows:
#         print("No row found for timestamp.")
#     else:
#         for row in rows:
#             print("Timestamp:", row[0])
#             print("SOC:", row[1])
#             print("SOH:", row[2])

SERIAL = 300000172
START_TIME = "2025-01-13T14:39:31.000Z"
END_TIME = "2025-01-13T19:11:22.000Z"
REST_CURRENT_THRESHOLD_A = 0.5

query = """
SELECT
    time,
    bspi2_current_a,
    bspi2_voltage_v,
    bspi2_soc_pct,
    bspi2_soh_pct
FROM data_harvest.datalogger_hvb
WHERE serial = %s
  AND time >= %s
  AND time <= %s
ORDER BY time ASC;
"""

params = (SERIAL, START_TIME, END_TIME)

with conn.cursor() as cur:
    cur.execute(query, params)
    rows = cur.fetchall()

if len(rows) < 2:
    print("Not enough rows found to calculate Ah.")
    print("Rows returned:", len(rows))
else:
    total_abs_throughput_ah = 0.0
    discharge_only_ah = 0.0
    total_abs_energy_wh = 0.0
    ignored_intervals = 0

    for current_row, next_row in zip(rows, rows[1:]):
        timestamp = current_row[0]
        next_timestamp = next_row[0]
        current_a = current_row[1]
        voltage_v = current_row[2]

        delta_seconds = (next_timestamp - timestamp).total_seconds()
        if delta_seconds <= 0:
            ignored_intervals += 1
            continue

        current_a = float(current_a or 0.0)
        voltage_v = float(voltage_v or 0.0)
        delta_hours = delta_seconds / 3600.0

        abs_current_a = abs(current_a)
        total_abs_throughput_ah += abs_current_a * delta_hours
        total_abs_energy_wh += voltage_v * abs_current_a * delta_hours

        if current_a < -REST_CURRENT_THRESHOLD_A:
            discharge_only_ah += abs_current_a * delta_hours

    first = rows[0]
    last = rows[-1]
    print("Serial:", SERIAL)
    print("Start timestamp:", first[0])
    print("End timestamp:", last[0])
    print("Rows returned:", len(rows))
    print("Ignored intervals:", ignored_intervals)
    print("Start SOC:", first[3])
    print("End SOC:", last[3])
    print("Start SOH:", first[4])
    print("End SOH:", last[4])
    print("Absolute throughput Ah:", round(total_abs_throughput_ah, 4))
    print("Discharge-only Ah:", round(discharge_only_ah, 4))
    print("Absolute energy Wh:", round(total_abs_energy_wh, 4))

conn.close()
