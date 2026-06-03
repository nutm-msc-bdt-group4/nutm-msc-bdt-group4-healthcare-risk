from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler

# ── START SPARK ───────────────────────────────────────────
spark = SparkSession.builder \
    .appName("NUTM_Group4_Features") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("STEP 4: FEATURE ENGINEERING")
print("=" * 60)

# ── LOAD DATA FROM HDFS ───────────────────────────────────
# Read the master table with vitals we saved in the last step
df = spark.read.parquet(
    "hdfs://localhost:9000/healthcare/processed/master_with_vitals/"
)

print("Data loaded. Row count:", df.count())
print()

# ── ENCODE GENDER ─────────────────────────────────────────
# Gender is currently stored as text: 'M' or 'F'
# Machine learning algorithms need numbers, not text.
# We convert: F = 0, M = 1
# This is called binary encoding.

df = df.withColumn(
    "gender_encoded",
    F.when(F.col("gender") == "M", 1).otherwise(0)
)

print("Gender encoding:")
df.groupBy("gender", "gender_encoded").count().show()

# ── DEFINE FEATURE COLUMNS ────────────────────────────────
# These are the columns we will use as input to the model.
# We deliberately exclude hadm_id, subject_id etc. because
# those are identifier columns, not clinical measurements.
# hospital_expire_flag is also excluded — we'll use it AFTER
# clustering to validate our results, not as an input.

feature_columns = [
    "age",
    "los",              # Length of ICU stay in days
    "gender_encoded",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "spo2"
]

print("Feature columns selected:")
for col in feature_columns:
    print(" -", col)
print()

# ── DROP ANY REMAINING NULLS ──────────────────────────────
# StandardScaler cannot handle NULL values.
# We drop any rows that still have NULLs in our feature columns.
# This should be very few or zero given our earlier cleaning.

before_count = df.count()
df = df.dropna(subset=feature_columns)
after_count = df.count()

print(f"Rows before dropping nulls: {before_count}")
print(f"Rows after dropping nulls:  {after_count}")
print(f"Rows removed: {before_count - after_count}")
print()

# ── STEP 1: VECTOR ASSEMBLER ──────────────────────────────
# Combines all feature columns into one vector column
# called 'raw_features'

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="raw_features"
)

df_assembled = assembler.transform(df)

print("After VectorAssembler:")
df_assembled.select("subject_id", "raw_features").show(5, truncate=False)
print()

# ── STEP 2: STANDARD SCALER ───────────────────────────────
# withMean=True: subtracts the mean of each feature
# withStd=True: divides by the standard deviation
# Result: each feature has mean=0 and std=1

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="scaled_features",
    withMean=True,
    withStd=True
)

# fit() calculates the mean and std from the data
# This is the 'learning' step — scaler learns the statistics
print("Fitting scaler to data...")
scaler_model = scaler.fit(df_assembled)

# transform() applies the scaling using what was learned
df_scaled = scaler_model.transform(df_assembled)

print("After StandardScaler:")
df_scaled.select("subject_id", "scaled_features").show(5, truncate=False)
print()

# ── VERIFY THE SCALING WORKED ─────────────────────────────
# After scaling, if we calculate the mean of each feature
# it should be very close to 0.
# The std should be very close to 1.
# Let's verify this for heart_rate as an example.

print("Verification — before scaling:")
df_assembled.select(
    F.mean("heart_rate").alias("mean_heart_rate"),
    F.stddev("heart_rate").alias("std_heart_rate")
).show()

print("Verification — after scaling the scaled vector")
print("(mean should be ~0, std should be ~1)")
print("This is confirmed if scaled_features column exists")
print("and contains both positive and negative values.")
print()

# ── SAVE TO HDFS ──────────────────────────────────────────
# We save the full dataframe including both the original
# columns AND the scaled_features column.
# We need the original columns later for cluster profiling.

df_scaled.write \
    .mode("overwrite") \
    .parquet("hdfs://localhost:9000/healthcare/processed/df_scaled/")

# Also save the scaler model so we can reuse it later
scaler_model.write().overwrite().save(
    "hdfs://localhost:9000/healthcare/models/scaler_model"
)

print("=" * 60)
print("Scaled features saved to HDFS.")
print("Scaler model saved to HDFS.")
print("=" * 60)

spark.stop()
