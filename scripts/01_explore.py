from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("NUTM_Group4_Explore") \
    .getOrCreate()

# Reduce Spark's console output so we only see our print statements
spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("Spark started successfully. Version:", spark.version)
print("=" * 60)

# Read CSV files from HDFS; inferSchema detects column types automatically
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

chartevents_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("hdfs://localhost:9000/healthcare/raw/CHARTEVENTS.csv")

notes_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("hdfs://localhost:9000/healthcare/raw/NOTEEVENTS.csv")

print("All 5 files loaded from HDFS successfully!")
print()

print("=" * 60)
print("PATIENTS")
print("=" * 60)
patients_df.printSchema()
patients_df.show(5)
print("Total rows:", patients_df.count())
print()

print("=" * 60)
print("ADMISSIONS")
print("=" * 60)
admissions_df.printSchema()
admissions_df.show(5)
print("Total rows:", admissions_df.count())
print()

print("=" * 60)
print("ICU STAYS")
print("=" * 60)
icustays_df.printSchema()
icustays_df.show(5)
print("Total rows:", icustays_df.count())
print()

print("=" * 60)
print("CHART EVENTS (Vital Signs)")
print("=" * 60)
chartevents_df.printSchema()
chartevents_df.show(5)
print("Total rows:", chartevents_df.count())
print()

print("=" * 60)
print("CLINICAL NOTES")
print("=" * 60)
notes_df.printSchema()
notes_df.show(5)
print("Total rows:", notes_df.count())
print()

print("=" * 60)
print("EXPLORATION COMPLETE")
print("=" * 60)

spark.stop()
