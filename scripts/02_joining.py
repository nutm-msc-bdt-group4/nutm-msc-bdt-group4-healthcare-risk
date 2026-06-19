from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("NUTM_Group4_Joining") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("STEP 2: JOINING TABLES")
print("=" * 60)

# Read from HDFS in every script - this is the correct big data pattern
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

# Inner join on hadm_id: keep only records present in both tables
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
icu_adm = icu_adm.drop(admissions_df["subject_id"])

print()

# Left join: keeps all ICU records even if patient demographics are missing
print("Adding patient demographics...")

master_df = icu_adm.join(
    patients_df.select(
        "subject_id",
        "gender",
        "dob",           # date of birth - for calculating age
        "expire_flag"    # 1 = patient eventually died, 0 = alive
    ),
    on="subject_id",
    how="left"
)

# MIMIC-III shifts dates forward for privacy but gaps between dates are accurate
# age = (admittime - dob) / 365 using datediff
master_df = master_df.withColumn(
    "age",
    (F.datediff(
        F.col("admittime"),
        F.col("dob")
    ) / 365).cast("double")
)

# MIMIC-III anonymises patients >89 by setting their age to 300+; filter to 18-120
master_df = master_df.filter(
    (F.col("age") >= 18) & (F.col("age") <= 120)
)

print("Master table created. Row count:", master_df.count())
print()

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

print("Summary statistics:")
master_df.select("age", "los", "hospital_expire_flag").describe().show()

print("ICU care units represented:")
master_df.groupBy("first_careunit").count().orderBy("count", ascending=False).show()

print("Mortality breakdown:")
master_df.groupBy("hospital_expire_flag").count().show()

# Save as Parquet - column-based format that Spark reads significantly faster than CSV
master_df.write \
    .mode("overwrite") \
    .parquet("hdfs://localhost:9000/healthcare/processed/master_patients/")

print()
print("=" * 60)
print("Master patient table saved to HDFS as Parquet.")
print("=" * 60)

spark.stop()
