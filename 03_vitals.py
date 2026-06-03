from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ── START SPARK ───────────────────────────────────────────
spark = SparkSession.builder \
    .appName("NUTM_Group4_Vitals") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("STEP 3: PROCESSING VITAL SIGNS")
print("=" * 60)

# ── LOAD DATA FROM HDFS ───────────────────────────────────
# Read raw CHARTEVENTS from HDFS
chartevents_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("hdfs://localhost:9000/healthcare/raw/CHARTEVENTS.csv")

# Read our master patient table (Parquet this time)
master_df = spark.read \
    .parquet("hdfs://localhost:9000/healthcare/processed/master_patients/")

print("Data loaded from HDFS.")
print("Total chart event rows:", chartevents_df.count())
print()

# ── STEP 1: FILTER TO VITAL SIGNS WE NEED ────────────────
# We only keep rows where itemid matches one of our 6 vital signs
# This reduces 758,355 rows down to a much smaller number

vital_itemids = [
    220045,   # Heart Rate
    220179,   # Systolic Blood Pressure
    220180,   # Diastolic Blood Pressure
    223762,   # Temperature (Celsius)
    220210,   # Respiratory Rate
    220277    # Oxygen Saturation (SpO2)
]

vitals_df = chartevents_df.filter(
    F.col("itemid").isin(vital_itemids)
)

print("After filtering to 6 vital signs:")
print("Vital sign rows:", vitals_df.count())
print()

# ── STEP 2: REMOVE BAD DATA ───────────────────────────────
# Remember we saw a value of 2,222,221.7 during exploration?
# That is clearly a data entry error — no vital sign is that large.
# We also remove any rows where valuenum is NULL.
# valuenum is the column containing the actual numeric measurement.

vitals_df = vitals_df.filter(
    F.col("valuenum").isNotNull() &   # remove NULLs
    (F.col("valuenum") > 0) &         # remove zeros and negatives
    (F.col("valuenum") < 10000)       # remove absurd outliers
)

print("After removing bad values:")
print("Clean vital sign rows:", vitals_df.count())
print()

# ── STEP 3: CALCULATE AVERAGE PER PATIENT PER VITAL ──────
# For each hospital admission (hadm_id) and each vital sign (itemid),
# calculate the average measurement across the entire ICU stay.
#
# Example: patient 10006 had heart rate measured 50 times.
# We calculate one average: 78.3 bpm for their whole stay.

vitals_avg = vitals_df.groupBy("hadm_id", "itemid").agg(
    F.avg("valuenum").alias("avg_value")
)

print("Average vitals per patient per vital sign:")
vitals_avg.show(10)
print()

# ── STEP 4: PIVOT ─────────────────────────────────────────
# Right now the data looks like this (long format):
#   hadm_id | itemid | avg_value
#   142345  | 220045 | 88.3      <- heart rate
#   142345  | 220179 | 125.0     <- systolic BP
#   142345  | 220180 | 72.0      <- diastolic BP
#
# We need it to look like this (wide format):
#   hadm_id | heart_rate | systolic_bp | diastolic_bp ...
#   142345  | 88.3       | 125.0       | 72.0
#
# pivot() does exactly this transformation.
# It takes the unique values in 'itemid' and makes each one a column.

vitals_wide = vitals_avg.groupBy("hadm_id").pivot(
    "itemid",
    vital_itemids   # specify the order of columns
).agg(F.avg("avg_value"))

# Rename columns from numeric codes to meaningful names
vitals_wide = vitals_wide \
    .withColumnRenamed("220045", "heart_rate") \
    .withColumnRenamed("220179", "systolic_bp") \
    .withColumnRenamed("220180", "diastolic_bp") \
    .withColumnRenamed("223762", "temperature") \
    .withColumnRenamed("220210", "respiratory_rate") \
    .withColumnRenamed("220277", "spo2")

print("After pivoting — one row per patient:")
vitals_wide.show(10)
print("Row count:", vitals_wide.count())
print()

# ── STEP 5: CHECK FOR MISSING VALUES ─────────────────────
# Not every patient will have every vital sign recorded.
# Let's see how many NULLs we have per column.

print("Missing values per vital sign column:")
vitals_wide.select([
    F.count(
        F.when(F.col(c).isNull(), c)
    ).alias(c)
    for c in ["heart_rate", "systolic_bp", "diastolic_bp",
              "temperature", "respiratory_rate", "spo2"]
]).show()
print()

# ── STEP 6: FILL MISSING VALUES WITH MEDIAN ───────────────
# For missing values, we use the MEDIAN of each vital sign.
# Why median and not mean (average)?
# Because extreme outliers pull the mean away from the centre.
# The median is the middle value — more robust for clinical data.
#
# percentile_approx(column, 0.5) calculates the 50th percentile
# which is the median.

medians = vitals_wide.agg(
    F.percentile_approx("heart_rate", 0.5).alias("hr_median"),
    F.percentile_approx("systolic_bp", 0.5).alias("sbp_median"),
    F.percentile_approx("diastolic_bp", 0.5).alias("dbp_median"),
    F.percentile_approx("temperature", 0.5).alias("temp_median"),
    F.percentile_approx("respiratory_rate", 0.5).alias("rr_median"),
    F.percentile_approx("spo2", 0.5).alias("spo2_median")
).collect()[0]

print("Median values used for filling missing data:")
print(f"  Heart Rate:       {medians['hr_median']} bpm")
print(f"  Systolic BP:      {medians['sbp_median']} mmHg")
print(f"  Diastolic BP:     {medians['dbp_median']} mmHg")
print(f"  Temperature:      {medians['temp_median']} C")
print(f"  Respiratory Rate: {medians['rr_median']} breaths/min")
print(f"  SpO2:             {medians['spo2_median']} %")
print()

vitals_clean = vitals_wide \
    .fillna({"heart_rate": medians["hr_median"]}) \
    .fillna({"systolic_bp": medians["sbp_median"]}) \
    .fillna({"diastolic_bp": medians["dbp_median"]}) \
    .fillna({"temperature": medians["temp_median"]}) \
    .fillna({"respiratory_rate": medians["rr_median"]}) \
    .fillna({"spo2": medians["spo2_median"]})

# ── STEP 7: JOIN VITALS INTO MASTER TABLE ────────────────
# Now we bring the clean vitals into our master patient table
# We join on hadm_id — the hospital admission ID

master_with_vitals = master_df.join(
    vitals_clean,
    on="hadm_id",
    how="left"
)

# Fill any remaining NULLs for patients with no vitals at all
master_with_vitals = master_with_vitals \
    .fillna({"heart_rate": medians["hr_median"]}) \
    .fillna({"systolic_bp": medians["sbp_median"]}) \
    .fillna({"diastolic_bp": medians["dbp_median"]}) \
    .fillna({"temperature": medians["temp_median"]}) \
    .fillna({"respiratory_rate": medians["rr_median"]}) \
    .fillna({"spo2": medians["spo2_median"]})

print("Master table with vitals:")
master_with_vitals.select(
    "subject_id",
    "age",
    "los",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "temperature",
    "respiratory_rate",
    "spo2",
    "hospital_expire_flag"
).show(10)

print("Final row count:", master_with_vitals.count())
print()

# ── SAVE TO HDFS ──────────────────────────────────────────
# Save with partitioning by first_careunit
# This satisfies the partitioning requirement in the rubric.
#
# Partitioning by first_careunit means Spark creates separate
# subfolders for each ICU unit: MICU, SICU, CCU, TSICU, CSRU
# If we later want only MICU patients, Spark reads just that
# subfolder instead of scanning the entire dataset.

master_with_vitals.write \
    .mode("overwrite") \
    .partitionBy("first_careunit") \
    .parquet("hdfs://localhost:9000/healthcare/processed/master_with_vitals/")

print("=" * 60)
print("Master table with vitals saved to HDFS.")
print("Partitioned by first_careunit.")
print("=" * 60)

spark.stop()
