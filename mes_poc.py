"""
MetroEnergy Solutions (MES) — Proof of Concept
================================================
COM7020 Big Data and Cloud Computing — Cloud Data Engineer PoC

WHAT THIS SCRIPT DOES (maps directly to the report's Architecture, Section 7):
  1. DATA GENERATION  -> simulates the "Data Sources" layer (smart meters, EV
     chargers, solar panels, weather feed) because MES has no public dataset;
     a synthetic generator with realistic daily/seasonal structure is built
     instead of pure random noise, so the forecasting step has real signal.
  2. BATCH INGESTION   -> simulates the "Batch Ingestion" pipeline: raw
     readings are written to disk as CSV/Parquet, exactly as billing/
     maintenance batch files would land in a cloud data lake landing zone.
  3. SPARK PROCESSING  -> simulates the "Processing Layer" (Apache Spark on
     a managed cluster in production): cleaning, aggregation, and feature
     engineering using PySpark DataFrame + SQL transformations.
  4. FORECASTING       -> simulates the "ML / Forecasting Layer": a Spark
     MLlib regression model predicts peak demand from calendar + weather
     features, demonstrating the analytics capability described in Task 1
     and Task 3 of the assignment brief.
  5. OUTPUTS           -> simulates the "Dashboard / Reporting Layer": CSV
     exports and PNG charts that would in production feed a BI dashboard
     (e.g. Power BI / QuickSight / Looker sitting on top of the warehouse).

Run with:  python3 mes_poc.py
Requires:  pyspark, pandas, matplotlib  (see requirements printed at top)

Output artefacts (written to ./outputs/):
  - raw_smart_meter_readings.csv       (raw synthetic sensor data)
  - daily_region_summary.csv           (Spark aggregation output)
  - hourly_load_profile.csv            (Spark aggregation output)
  - peak_demand_predictions.csv        (model predictions vs actuals)
  - model_metrics.txt                  (RMSE / R2 / feature coefficients)
  - chart_daily_consumption_trend.png
  - chart_hourly_load_profile.png
  - chart_prediction_vs_actual.png
"""

import os
import random
import math
import subprocess
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering, no display server required
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
# GCS bucket configuration.
# WHY THIS CHANGED: the original script wrote to a bare relative path
# ("outputs/..."), which Spark resolved against the cluster's DEFAULT
# filesystem (HDFS) rather than local disk -- that mismatch is exactly
# what produced the PATH_NOT_FOUND error. Every path below now has an
# explicit scheme (gs:// or file://) so there is no ambiguity about
# which filesystem Spark or pandas is talking to.
GCS_BUCKET = "mes-bd-cc"
GCS_RAW_DATASET = f"gs://{GCS_BUCKET}/raw_smart_meter_readings.csv"   # already uploaded
GCS_OUTPUT_PREFIX = f"gs://{GCS_BUCKET}/outputs"                       # where results land

# pandas / matplotlib cannot write directly to gs:// -- only Spark's
# DataFrameWriter understands the GCS connector. So small artefacts
# (aggregated CSVs, charts, metrics) are written to local disk on the
# driver first, then uploaded to GCS explicitly at the end of the script.
LOCAL_OUTPUT_DIR = "/tmp/mes_poc_outputs"
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
OUTPUT_DIR = LOCAL_OUTPUT_DIR  # kept as OUTPUT_DIR so the rest of the script is unchanged

# Since raw_smart_meter_readings.csv already exists at GCS_RAW_DATASET,
# the generation step is SKIPPED by default. Set to True only if you
# want to regenerate synthetic data and overwrite what's in the bucket.
REGENERATE_DATASET = False

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

REGIONS = ["Berlin", "Hamburg", "Munich", "Cologne", "Leipzig"]
METERS_PER_REGION = 40          # smart meters simulated per region
DAYS_OF_HISTORY = 365           # one full year of hourly data
START_DATE = datetime(2025, 1, 1, 0, 0, 0)

# --------------------------------------------------------------------------
# STEP 1 — SYNTHETIC DATA GENERATION
# --------------------------------------------------------------------------
# Design rationale (see Report Section 8.1):
#   Real MES data is confidential and no public dataset matches this exact
#   scenario, so a synthetic generator is used, as explicitly permitted by
#   the assignment brief ("You may use synthetic or hypothetical datasets").
#   Consumption follows:
#     - a daily cycle (low overnight, morning peak ~08:00, evening peak ~19:00)
#     - a weekly cycle (lower on weekends)
#     - a seasonal cycle (higher in winter for heating-adjacent regions)
#     - weather coupling (temperature drives heating/cooling load)
#     - Gaussian sensor noise + occasional fault spikes/dropouts
# This produces a plausible signal for the forecasting model to learn from,
# while remaining clearly and transparently synthetic.

def daily_load_shape(hour: float) -> float:
    """Return a 0-1 multiplier representing typical household load by hour."""
    morning_peak = math.exp(-((hour - 8) ** 2) / (2 * 2.0 ** 2))
    evening_peak = math.exp(-((hour - 19) ** 2) / (2 * 2.5 ** 2))
    baseline = 0.25
    return min(1.0, baseline + 0.55 * evening_peak + 0.35 * morning_peak)


def seasonal_temperature(day_of_year: int) -> float:
    """Approximate a Central European seasonal temperature curve (deg C)."""
    return 9.0 - 9.0 * math.cos((2 * math.pi / 365.0) * (day_of_year - 15))


def generate_readings():
    """Yield one dict per (meter, hour) combination for the full history."""
    for region_idx, region in enumerate(REGIONS):
        region_base_load = random.uniform(1.4, 2.2)  # kWh baseline household load
        for meter_num in range(METERS_PER_REGION):
            meter_id = f"{region[:3].upper()}-{meter_num:04d}"
            household_scale = random.uniform(0.7, 1.5)
            has_solar = random.random() < 0.35
            has_ev = random.random() < 0.25
            timestamp = START_DATE

            for day in range(DAYS_OF_HISTORY):
                temp_today = seasonal_temperature(day) + random.gauss(0, 2.0)
                is_weekend = timestamp.weekday() >= 5

                for hour in range(24):
                    ts = timestamp + timedelta(hours=hour)
                    load_shape = daily_load_shape(hour)
                    weekend_factor = 0.85 if is_weekend else 1.0

                    # Heating/cooling coupling: consumption rises when cold or hot
                    thermal_factor = 1.0 + max(0, (10 - temp_today)) * 0.02
                    thermal_factor += max(0, (temp_today - 24)) * 0.015

                    consumption = (
                        region_base_load
                        * household_scale
                        * load_shape
                        * weekend_factor
                        * thermal_factor
                    )
                    consumption += random.gauss(0, 0.08)  # sensor noise
                    consumption = max(0.0, consumption)

                    # Fault injection: rare dropout or spike (~0.3% of readings)
                    fault_flag = 0
                    r = random.random()
                    if r < 0.002:
                        consumption = 0.0
                        fault_flag = 1
                    elif r < 0.003:
                        consumption *= random.uniform(3, 5)
                        fault_flag = 1

                    solar_kwh = 0.0
                    if has_solar and 6 <= hour <= 18:
                        solar_curve = math.sin(math.pi * (hour - 6) / 12.0)
                        seasonal_sun = 0.5 + 0.5 * math.cos(
                            (2 * math.pi / 365.0) * (day - 172)
                        )  # more sun in summer
                        solar_kwh = max(
                            0.0, 1.8 * solar_curve * seasonal_sun + random.gauss(0, 0.05)
                        )

                    ev_kwh = 0.0
                    if has_ev and (hour in (18, 19, 20, 21, 22, 23) or hour in (0, 1)):
                        if random.random() < 0.4:
                            ev_kwh = random.uniform(2.0, 7.0)

                    voltage = round(230 + random.gauss(0, 2.5), 2)

                    yield {
                        "timestamp": ts,
                        "meter_id": meter_id,
                        "region": region,
                        "consumption_kwh": round(consumption, 4),
                        "solar_generation_kwh": round(solar_kwh, 4),
                        "ev_charging_kwh": round(ev_kwh, 4),
                        "voltage_v": voltage,
                        "temperature_c": round(temp_today, 2),
                        "fault_flag": fault_flag,
                    }
                timestamp += timedelta(days=1)


if REGENERATE_DATASET:
    print("Generating synthetic MES smart-meter dataset (this simulates the raw")
    print("ingestion layer — in production these rows would arrive continuously")
    print("from smart meters / IoT gateways rather than being generated locally)...")

    # NOTE: full resolution (5 regions x 40 meters x 365 days x 24h) = ~1.75M rows.
    records = list(generate_readings())
    raw_pdf = pd.DataFrame.from_records(records)
    local_raw_path = os.path.join(OUTPUT_DIR, "raw_smart_meter_readings.csv")
    raw_pdf.to_csv(local_raw_path, index=False)
    print(f"Generated {len(raw_pdf):,} raw readings -> {local_raw_path}")
    # Upload to GCS so this run's data matches what's in the bucket.
    subprocess.run(["gcloud", "storage", "cp", local_raw_path, GCS_RAW_DATASET], check=True)
    print(f"Uploaded regenerated dataset -> {GCS_RAW_DATASET}")

raw_csv_path = GCS_RAW_DATASET  # Spark reads directly from GCS, not local disk
print(f"Using existing dataset in bucket: {raw_csv_path}")

# --------------------------------------------------------------------------
# STEP 2 — SPARK SESSION + BATCH INGESTION
# --------------------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("MES-BigData-PoC")
    .master("local[*]")          # local mode for PoC; production = cluster/EMR/Databricks
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print("\n[Spark] Session started. Loading raw CSV into a Spark DataFrame ...")

schema = T.StructType([
    T.StructField("timestamp", T.TimestampType(), False),
    T.StructField("meter_id", T.StringType(), False),
    T.StructField("region", T.StringType(), False),
    T.StructField("consumption_kwh", T.DoubleType(), False),
    T.StructField("solar_generation_kwh", T.DoubleType(), False),
    T.StructField("ev_charging_kwh", T.DoubleType(), False),
    T.StructField("voltage_v", T.DoubleType(), False),
    T.StructField("temperature_c", T.DoubleType(), False),
    T.StructField("fault_flag", T.IntegerType(), False),
])

df = spark.read.csv(raw_csv_path, header=True, schema=schema)
df = df.withColumn("hour", F.hour("timestamp")) \
       .withColumn("day_of_week", F.dayofweek("timestamp")) \
       .withColumn("month", F.month("timestamp")) \
       .withColumn("date", F.to_date("timestamp")) \
       .withColumn("is_weekend", F.when(F.dayofweek("timestamp").isin(1, 7), 1).otherwise(0))

print(f"[Spark] Loaded {df.count():,} rows across {df.rdd.getNumPartitions()} partitions.")

# --------------------------------------------------------------------------
# STEP 3 — DATA QUALITY / CLEANING
# --------------------------------------------------------------------------
# Demonstrates the "data quality" big-data challenge named in Task 1:
# faulty readings (fault_flag = 1) are excluded from aggregate analytics
# but retained in a separate quarantine table for the maintenance team.
clean_df = df.filter(F.col("fault_flag") == 0)
faulty_df = df.filter(F.col("fault_flag") == 1)
print(f"[Spark] Quarantined {faulty_df.count():,} faulty readings "
      f"({faulty_df.count() / df.count():.3%} of total) for maintenance review.")

# --------------------------------------------------------------------------
# STEP 4 — BATCH AGGREGATION (simulates the daily billing / reporting batch)
# --------------------------------------------------------------------------
daily_region_summary = (
    clean_df.groupBy("date", "region")
    .agg(
        F.round(F.sum("consumption_kwh"), 2).alias("total_consumption_kwh"),
        F.round(F.avg("consumption_kwh"), 4).alias("avg_consumption_kwh"),
        F.round(F.max("consumption_kwh"), 4).alias("peak_meter_consumption_kwh"),
        F.round(F.sum("solar_generation_kwh"), 2).alias("total_solar_kwh"),
        F.round(F.sum("ev_charging_kwh"), 2).alias("total_ev_kwh"),
        F.round(F.avg("temperature_c"), 2).alias("avg_temperature_c"),
    )
    .orderBy("date", "region")
)

hourly_load_profile = (
    clean_df.groupBy("region", "hour")
    .agg(F.round(F.avg("consumption_kwh"), 4).alias("avg_consumption_kwh"))
    .orderBy("region", "hour")
)

# Identify system-wide peak demand hour per day (grid-stability use case)
peak_demand_per_day = (
    clean_df.groupBy("date", "hour")
    .agg(F.round(F.sum("consumption_kwh"), 2).alias("total_grid_load_kwh"))
    .withColumn(
        "rank",
        F.row_number().over(
            __import__("pyspark").sql.Window.partitionBy("date").orderBy(F.desc("total_grid_load_kwh"))
        ),
    )
    .filter(F.col("rank") == 1)
    .drop("rank")
    .orderBy("date")
)

daily_region_summary_pdf = daily_region_summary.toPandas()
hourly_load_profile_pdf = hourly_load_profile.toPandas()
peak_demand_per_day_pdf = peak_demand_per_day.toPandas()

daily_region_summary_pdf.to_csv(os.path.join(OUTPUT_DIR, "daily_region_summary.csv"), index=False)
hourly_load_profile_pdf.to_csv(os.path.join(OUTPUT_DIR, "hourly_load_profile.csv"), index=False)
peak_demand_per_day_pdf.to_csv(os.path.join(OUTPUT_DIR, "peak_demand_per_day.csv"), index=False)
print("[Spark] Batch aggregation complete -> daily_region_summary.csv, "
      "hourly_load_profile.csv, peak_demand_per_day.csv")

# --------------------------------------------------------------------------
# STEP 5 — PEAK DEMAND FORECASTING (Spark MLlib Linear Regression)
# --------------------------------------------------------------------------
# Business question (Task 1 / Task 4): can MES predict the system-wide peak
# demand for a given day from calendar + weather features, ahead of time,
# to support grid-stability planning and proactive capacity management?

daily_features = (
    clean_df.groupBy("date")
    .agg(
        F.round(F.sum("consumption_kwh"), 2).alias("total_daily_kwh"),
        F.round(F.max("consumption_kwh"), 4).alias("peak_reading_kwh"),
        F.first("temperature_c").alias("temperature_c"),
        F.first("day_of_week").alias("day_of_week"),
        F.first("month").alias("month"),
        F.first("is_weekend").alias("is_weekend"),
    )
    .join(
        peak_demand_per_day.select("date", "total_grid_load_kwh"),
        on="date",
        how="inner",
    )
    .withColumnRenamed("total_grid_load_kwh", "peak_grid_demand_kwh")
    .orderBy("date")
)

assembler = VectorAssembler(
    inputCols=["temperature_c", "day_of_week", "month", "is_weekend", "total_daily_kwh"],
    outputCol="features",
)
ml_ready = assembler.transform(daily_features).select("date", "features", "peak_grid_demand_kwh")

train_df, test_df = ml_ready.randomSplit([0.8, 0.2], seed=RANDOM_SEED)

lr = LinearRegression(featuresCol="features", labelCol="peak_grid_demand_kwh")
lr_model = lr.fit(train_df)

predictions = lr_model.transform(test_df)

evaluator_rmse = RegressionEvaluator(
    labelCol="peak_grid_demand_kwh", predictionCol="prediction", metricName="rmse"
)
evaluator_r2 = RegressionEvaluator(
    labelCol="peak_grid_demand_kwh", predictionCol="prediction", metricName="r2"
)
rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

predictions_pdf = predictions.select(
    "date", "peak_grid_demand_kwh", "prediction"
).orderBy("date").toPandas()
predictions_pdf.to_csv(os.path.join(OUTPUT_DIR, "peak_demand_predictions.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "model_metrics.txt"), "w") as f:
    f.write("MES Peak Demand Forecasting Model — Spark MLlib Linear Regression\n")
    f.write("=" * 65 + "\n")
    f.write(f"Training rows: {train_df.count()}\n")
    f.write(f"Test rows:     {test_df.count()}\n")
    f.write(f"RMSE (kWh):    {rmse:.4f}\n")
    f.write(f"R^2:           {r2:.4f}\n\n")
    f.write("Feature coefficients (temperature_c, day_of_week, month, "
            "is_weekend, total_daily_kwh):\n")
    f.write(f"{lr_model.coefficients}\n")
    f.write(f"Intercept: {lr_model.intercept:.4f}\n")

print(f"\n[MLlib] Peak demand model trained. RMSE={rmse:.4f} kWh, R2={r2:.4f}")
print("[MLlib] Metrics written -> model_metrics.txt")

# --------------------------------------------------------------------------
# STEP 6 — VISUALISATIONS (for the "Dashboard / Reporting" layer)
# --------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Chart 1: daily total consumption trend (system-wide, 30-day rolling mean)
daily_totals = (
    daily_region_summary_pdf.groupby("date")["total_consumption_kwh"].sum().reset_index()
)
daily_totals["date"] = pd.to_datetime(daily_totals["date"])
daily_totals = daily_totals.sort_values("date")
daily_totals["rolling_30d"] = daily_totals["total_consumption_kwh"].rolling(30, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(daily_totals["date"], daily_totals["total_consumption_kwh"], alpha=0.35, label="Daily total (kWh)")
ax.plot(daily_totals["date"], daily_totals["rolling_30d"], linewidth=2.2, label="30-day rolling mean")
ax.set_title("MES System-Wide Daily Energy Consumption — Full Year (Synthetic PoC Data)")
ax.set_xlabel("Date")
ax.set_ylabel("Total Consumption (kWh)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "chart_daily_consumption_trend.png"), dpi=150)
plt.close(fig)

# Chart 2: average hourly load profile per region
fig, ax = plt.subplots(figsize=(10, 5))
for region in REGIONS:
    subset = hourly_load_profile_pdf[hourly_load_profile_pdf["region"] == region]
    ax.plot(subset["hour"], subset["avg_consumption_kwh"], marker="o", markersize=3, label=region)
ax.set_title("Average Hourly Load Profile by Region (Synthetic PoC Data)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Average Consumption per Meter (kWh)")
ax.set_xticks(range(0, 24, 2))
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "chart_hourly_load_profile.png"), dpi=150)
plt.close(fig)

# Chart 3: predicted vs actual peak demand (model evaluation)
pred_sorted = predictions_pdf.sort_values("date")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(pd.to_datetime(pred_sorted["date"]), pred_sorted["peak_grid_demand_kwh"], marker="o", markersize=4, label="Actual peak demand")
ax.plot(pd.to_datetime(pred_sorted["date"]), pred_sorted["prediction"], marker="x", markersize=4, label="Predicted peak demand")
ax.set_title(f"Peak Grid Demand — Predicted vs Actual (Test Set, RMSE={rmse:.2f} kWh)")
ax.set_xlabel("Date")
ax.set_ylabel("Peak Grid Demand (kWh)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "chart_prediction_vs_actual.png"), dpi=150)
plt.close(fig)

print("\n[Charts] Saved 3 PNG charts locally.")

# --------------------------------------------------------------------------
# STEP 7 — UPLOAD LOCAL ARTEFACTS TO GCS
# --------------------------------------------------------------------------
# Everything in LOCAL_OUTPUT_DIR (CSVs, PNGs, model_metrics.txt) was written
# to the driver's local disk because pandas/matplotlib cannot target gs://
# directly. Files are uploaded one at a time with explicit paths --
# deliberately NOT relying on shell "*" glob expansion, since subprocess
# with shell=False will not expand it and shell=True is an unnecessary
# injection risk for a path built from a hardcoded bucket name.
print(f"\nUploading local artefacts from {LOCAL_OUTPUT_DIR} -> {GCS_OUTPUT_PREFIX} ...")
for fname in sorted(os.listdir(LOCAL_OUTPUT_DIR)):
    local_path = os.path.join(LOCAL_OUTPUT_DIR, fname)
    if os.path.isfile(local_path):
        dest = f"{GCS_OUTPUT_PREFIX}/{fname}"
        subprocess.run(["gcloud", "storage", "cp", local_path, dest], check=True)
        print(f"  uploaded {fname} -> {dest}")
print(f"[GCS] Upload complete. Results available at: {GCS_OUTPUT_PREFIX}/")

print("\n=== PoC COMPLETE ===")
print(f"Local artefacts:  {os.path.abspath(LOCAL_OUTPUT_DIR)}")
print(f"GCS artefacts:    {GCS_OUTPUT_PREFIX}/")

spark.stop()
