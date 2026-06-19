from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler

spark = SparkSession.builder \
    .appName("NUTM_Group4_Features") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("STEP 4: FEATURE ENGINEERING")
print("=" * 60)

df = spark.read.parquet(
    "hdfs://localhost:9000/healthcare/processed/master_with_vitals/"
)

print("Data loaded. Row count:", df.count())
print()

# Encode gender as binary: F=0, M=1
df = df.withColumn(
    "gender_encoded",
    F.when(F.col("gender") == "M", 1).otherwise(0)
)

print("Gender encoding:")
df.groupBy("gender", "gender_encoded").count().show()

# Exclude hospital_expire_flag - used post-clustering for validation only, not as input
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

before_count = df.count()
df = df.dropna(subset=feature_columns)
after_count = df.count()

print(f"Rows before dropping nulls: {before_count}")
print(f"Rows after dropping nulls:  {after_count}")
print(f"Rows removed: {before_count - after_count}")
print()

# Combine feature columns into a single vector column required by Spark ML
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="raw_features"
)

df_assembled = assembler.transform(df)

print("After VectorAssembler:")
df_assembled.select("subject_id", "raw_features").show(5, truncate=False)
print()

# StandardScaler: withMean=True, withStd=True gives zero mean and unit variance per feature
scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="scaled_features",
    withMean=True,
    withStd=True
)

print("Fitting scaler to data...")
scaler_model = scaler.fit(df_assembled)

df_scaled = scaler_model.transform(df_assembled)

print("After StandardScaler:")
df_scaled.select("subject_id", "scaled_features").show(5, truncate=False)
print()

print("Verification - before scaling:")
df_assembled.select(
    F.mean("heart_rate").alias("mean_heart_rate"),
    F.stddev("heart_rate").alias("std_heart_rate")
).show()

print("Verification - after scaling the scaled vector")
print("(mean should be ~0, std should be ~1)")
print("This is confirmed if scaled_features column exists")
print("and contains both positive and negative values.")
print()

# Save both the scaled dataset and the scaler model for reuse in downstream scripts
df_scaled.write \
    .mode("overwrite") \
    .parquet("hdfs://localhost:9000/healthcare/processed/df_scaled/")

scaler_model.write().overwrite().save(
    "hdfs://localhost:9000/healthcare/models/scaler_model"
)

print("=" * 60)
print("Scaled features saved to HDFS.")
print("Scaler model saved to HDFS.")
print("=" * 60)

spark.stop()
