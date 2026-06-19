from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    HashingTF,
    IDF
)

spark = SparkSession.builder \
    .appName("NUTM_Group4_TFIDF") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("STEP 5: TF-IDF PIPELINE ON CLINICAL NOTES")
print("=" * 60)

notes_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("hdfs://localhost:9000/healthcare/raw/NOTEEVENTS.csv")

print("Notes loaded from HDFS.")
print("Total rows:", notes_df.count())
print()

notes_df.printSchema()
notes_df.show(5)

notes_count = notes_df.count()

if notes_count == 0:
    print("=" * 60)
    print("NOTE: Demo dataset contains no clinical notes.")
    print("Building TF-IDF pipeline to demonstrate correct")
    print("architecture. Pipeline will be tested on real")
    print("clinical text from the full MIMIC dataset separately.")
    print("=" * 60)
    print()

    # Fit on synthetic clinical text to validate pipeline stages
    sample_notes = spark.createDataFrame([
        (100001, "Patient presented with acute respiratory failure history of COPD"),
        (100002, "Admitted for sepsis secondary to urinary tract infection fever chills"),
        (100003, "Post operative care following coronary artery bypass surgery cardiac"),
        (100004, "Diabetic ketoacidosis insulin dependent diabetes mellitus glucose"),
        (100005, "Acute myocardial infarction chest pain shortness of breath cardiac"),
    ], ["hadm_id", "text"])

    print("Using sample clinical text to demonstrate pipeline:")
    sample_notes.show(truncate=False)
    print()

    working_df = sample_notes

else:
    # Filter to discharge summaries and combine multiple notes per admission
    discharge_notes = notes_df.filter(
        F.col("category") == "Discharge summary"
    ).select("hadm_id", "text")

    working_df = discharge_notes.groupBy("hadm_id").agg(
        F.concat_ws(" ", F.collect_list("text")).alias("text")
    )

    print("Discharge summaries:", working_df.count())

# Stage 1: split text into tokens
tokenizer = Tokenizer(inputCol="text", outputCol="words")

# Stage 2: remove common English stop words (the, a, is, of, etc.)
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

# Stage 3: hash filtered words into a 5000-dimensional frequency vector
hashing_tf = HashingTF(
    inputCol="filtered_words",
    outputCol="raw_tf",
    numFeatures=5000
)

# Stage 4: downweight words common across all notes; upweight distinctive clinical terms
idf = IDF(inputCol="raw_tf", outputCol="text_features", minDocFreq=1)

text_pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf])

print("TF-IDF Pipeline stages:")
print("  Stage 1: Tokenizer")
print("  Stage 2: StopWordsRemover")
print("  Stage 3: HashingTF (5000 features)")
print("  Stage 4: IDF")
print()

print("Fitting pipeline to text data...")
pipeline_model = text_pipeline.fit(working_df)

print("Transforming text to TF-IDF features...")
df_tfidf = pipeline_model.transform(working_df)

print("Pipeline transformation results:")
df_tfidf.select(
    "hadm_id",
    "text",
    "words",
    "filtered_words"
).show(5, truncate=50)

print("Final TF-IDF feature vectors:")
df_tfidf.select("hadm_id", "text_features").show(5, truncate=80)

print("Sample - words before and after stop word removal:")
df_tfidf.select("words", "filtered_words").show(3, truncate=60)

pipeline_model.write().overwrite().save(
    "hdfs://localhost:9000/healthcare/models/tfidf_pipeline"
)

print()
print("=" * 60)
print("TF-IDF pipeline saved to HDFS.")
print("When real clinical notes are available, point")
print("this pipeline at NOTEEVENTS.csv and it will")
print("produce text_features for each patient admission.")
print("=" * 60)

spark.stop()
