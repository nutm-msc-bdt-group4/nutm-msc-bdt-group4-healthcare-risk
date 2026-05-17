from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ── START SPARK ───────────────────────────────────────────
spark = SparkSession.builder \
    .appName("NUTM_Group4_Joining") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("STEP 2: JOINING TABLES")
print("=" * 60)

# ── LOAD DATA FROM HDFS ───────────────────────────────────
# We read from HDFS again — always read from HDFS in every script
# This is the correct big data pattern

patients_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("hdfs://localhost:9000/healthcare/raw/PATIENTS.csv")

admissions_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("hdfs://localhost:9000/healthcare/raw/ADMISSIONS.csv")

icustays_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("hdfs://localhost:9000/healthcare/raw/ICUSTAYS.csv")

print("Tables loaded from HDFS.")
print()

# ── JOIN 1: ICU STAYS + ADMISSIONS ───────────────────────
# We join on 'hadm_id' because that column exists in BOTH tables
# It is the shared key that links them together.
#
# join type 'inner' means:
# Only keep rows where hadm_id exists in BOTH tables.
# If an admission has no ICU stay, drop it.
# If an ICU stay has no matching admission, drop it.
# We only want records that are complete on both sides.

print("Joining ICU stays with admissions...")

icu_adm = icustays_df.join(
    admissions_df.select(
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "admission_type",
        "diagnosis",
        "hospital_expire_flag"   # 1 = died, 0 = survived
    ),
    on="hadm_id",
    how="inner"
)

print("ICU + Admissions joined. Row count:", icu_adm.count())
icu_adm.show(5)

# Drop the duplicate subject_id that came from the admissions table
# After the join, subject_id exists twice — we keep only one copy
icu_adm = icu_adm.drop(admissions_df["subject_id"])

print()

# ── JOIN 2: ADD PATIENT DEMOGRAPHICS ─────────────────────
# Now we bring in patient-level info from PATIENTS table
# We join on 'subject_id' this time — the unique patient identifier
# 'left' join means:
# Keep ALL rows from icu_adm even if no match found in patients_df
# In practice all patients will match, but left join is safer

print("Adding patient demographics...")

master_df = icu_adm.join(
    patients_df.select(
        "subject_id",
        "gender",
        "dob",           # date of birth — for calculating age
        "expire_flag"    # 1 = patient eventually died, 0 = alive
    ),
    on="subject_id",
    how="left"
)

# ── CALCULATE PATIENT AGE ─────────────────────────────────
# Remember: MIMIC-III shifts dates forward for privacy
# But the GAP between dates is accurate
# So: age = (admission date - date of birth) / 365
#
# datediff() calculates the number of days between two dates
# We divide by 365 to convert days into years
# .cast("double") ensures the result is a decimal number

master_df = master_df.withColumn(
    "age",
    (F.datediff(
        F.col("admittime"),
        F.col("dob")
    ) / 365).cast("double")
)

# ── FILTER UNREALISTIC AGES ───────────────────────────────
# MIMIC-III anonymises patients older than 89 by setting
# their age to 300+. We filter these out to avoid skewing
# our clustering. We keep ages between 18 and 120.

master_df = master_df.filter(
    (F.col("age") >= 18) & (F.col("age") <= 120)
)

print("Master table created. Row count:", master_df.count())
print()

# ── INSPECT THE RESULT ────────────────────────────────────
print("Master table schema:")
master_df.printSchema()

print("Sample rows:")
master_df.select(
    "subject_id",
    "hadm_id",
    "icustay_id",
    "gender",
    "age",
    "los",
    "first_careunit",
    "hospital_expire_flag",
    "diagnosis"
).show(10)

# ── QUICK SUMMARY STATISTICS ──────────────────────────────
print("Summary statistics:")
master_df.select("age", "los", "hospital_expire_flag").describe().show()

print("ICU care units represented:")
master_df.groupBy("first_careunit").count().orderBy("count", ascending=False).show()

print("Mortality breakdown:")
master_df.groupBy("hospital_expire_flag").count().show()

# ── SAVE TO HDFS ──────────────────────────────────────────
# We save as Parquet format instead of CSV.
# Parquet is a compressed, column-based format that Spark
# reads much faster than CSV. Think of CSV as a text file
# and Parquet as a zip file optimised specifically for Spark.
# All processed data from here on will be saved as Parquet.

master_df.write \
    .mode("overwrite") \
    .parquet("hdfs://localhost:9000/healthcare/processed/master_patients/")

print()
print("=" * 60)
print("Master patient table saved to HDFS as Parquet.")
print("=" * 60)

spark.stop()
