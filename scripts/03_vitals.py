from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("NUTM_Group4_Vitals") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("STEP 3: PROCESSING VITAL SIGNS")
print("=" * 60)

chartevents_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("hdfs://localhost:9000/healthcare/raw/CHARTEVENTS.csv")

master_df = spark.read \
    .parquet("hdfs://localhost:9000/healthcare/processed/master_patients/")

print("Data loaded from HDFS.")
print("Total chart event rows:", chartevents_df.count())
print()

# Filter to six target vital sign ITEMIDs: HR, SBP, DBP, Temperature, RR, SpO2
# Reduces CHARTEVENTS from 758,355 to ~33,996 rows
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

# Remove NULLs, zeros, and absurd outliers (e.g. valuenum=2,222,221.7 seen in exploration)
vitals_df = vitals_df.filter(
    F.col("valuenum").isNotNull() &
    (F.col("valuenum") > 0) &
    (F.col("valuenum") < 10000)
)

print("After removing bad values:")
print("Clean vital sign rows:", vitals_df.count())
print()

# Average each vital sign across the full ICU stay per admission
vitals_avg = vitals_df.groupBy("hadm_id", "itemid").agg(
    F.avg("valuenum").alias("avg_value")
)

print("Average vitals per patient per vital sign:")
vitals_avg.show(10)
print()

# Pivot from long format (one row per vital per patient) to wide format (one row per patient)
vitals_wide = vitals_avg.groupBy("hadm_id").pivot(
    "itemid",
    vital_itemids
).agg(F.avg("avg_value"))

vitals_wide = vitals_wide \
    .withColumnRenamed("220045", "heart_rate") \
    .withColumnRenamed("220179", "systolic_bp") \
    .withColumnRenamed("220180", "diastolic_bp") \
    .withColumnRenamed("223762", "temperature") \
    .withColumnRenamed("220210", "respiratory_rate") \
    .withColumnRenamed("220277", "spo2")

print("After pivoting - one row per patient:")
vitals_wide.show(10)
print("Row count:", vitals_wide.count())
print()

print("Missing values per vital sign column:")
vitals_wide.select([
    F.count(
        F.when(F.col(c).isNull(), c)
    ).alias(c)
    for c in ["heart_rate", "systolic_bp", "diastolic_bp",
              "temperature", "respiratory_rate", "spo2"]
]).show()
print()

# Use median (not mean) for imputation - more robust to the outliers present in ICU data
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

# Partition by first_careunit to enable care-unit-specific queries with minimal I/O
master_with_vitals.write \
    .mode("overwrite") \
    .partitionBy("first_careunit") \
    .parquet("hdfs://localhost:9000/healthcare/processed/master_with_vitals/")

print("=" * 60)
print("Master table with vitals saved to HDFS.")
print("Partitioned by first_careunit.")
print("=" * 60)

spark.stop()
