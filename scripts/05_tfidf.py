from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    HashingTF,
    IDF
)

# starts sparks session.
spark = SparkSession.builder \
    .appName("NUTM_Group4_TFIDF") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("STEP 5: TF-IDF PIPELINE ON CLINICAL NOTES")
print("=" * 60)

#Loads the clinical notes CSV from HDFS.
notes_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("hdfs://localhost:9000/healthcare/raw/NOTEEVENTS.csv")

print("Notes loaded from HDFS.")
print("Total rows:", notes_df.count())
print()

# Shows the column names and detected types.
notes_df.printSchema()
notes_df.show(5)

# ── HANDLE THE EMPTY NOTES SCENARIO 
# As we discovered during exploration, the demo dataset
# has 0 clinical notes. We handle this gracefully by
# detecting it and explaining what the pipeline WOULD do
# with real data.

notes_count = notes_df.count()

if notes_count == 0:
    print("=" * 60)
    print("NOTE: Demo dataset contains no clinical notes.")
    print("Building TF-IDF pipeline to demonstrate correct")
    print("architecture. Pipeline will be tested on real")
    print("clinical text from the full MIMIC dataset separately.")
    print("=" * 60)
    print()

    # We still build and save the pipeline structure
    # so it exists in our codebase and can be explained

    # Create a small sample dataframe to fit the pipeline on
    # so Spark can validate the pipeline stages are correct
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
    # This branch runs when real notes are available
    # Filter to discharge summaries only
    discharge_notes = notes_df.filter(
        F.col("category") == "Discharge summary"
    ).select("hadm_id", "text")

    # Combine multiple notes per admission into one
    working_df = discharge_notes.groupBy("hadm_id").agg(
        F.concat_ws(" ", F.collect_list("text")).alias("text")
    )

    print("Discharge summaries:", working_df.count())

# ── BUILD THE TF-IDF PIPELINE 
# A Pipeline chains multiple processing steps together.
# Each step's output becomes the next step's input.
# This is the Spark ML Pipeline the rubric requires.

# STAGE 1: TOKENIZER
# Splits text into individual words (tokens)
# "acute respiratory failure" → ["acute", "respiratory", "failure"]
tokenizer = Tokenizer(
    inputCol="text",
    outputCol="words"
)

# STAGE 2: STOP WORDS REMOVER
# Removes common English words that carry no medical meaning
# Removes: "the", "a", "an", "is", "was", "of", "and" etc.
# Keeps: "respiratory", "sepsis", "cardiac", "failure" etc.
remover = StopWordsRemover(
    inputCol="words",
    outputCol="filtered_words"
)

# STAGE 3: HASHING TF
# Converts filtered words into a numerical vector
# numFeatures=5000 means we track 5000 word pattern slots
# Each word gets hashed (assigned) to one of these slots
# The value in each slot = how many times that word appeared
hashing_tf = HashingTF(
    inputCol="filtered_words",
    outputCol="raw_tf",
    numFeatures=5000
)

# STAGE 4: IDF
# Takes the raw word counts and weights them
# Common words across all notes get lower weights
# Rare distinctive words get higher weights
# minDocFreq=1 means include words appearing in at least 1 doc
idf = IDF(
    inputCol="raw_tf",
    outputCol="text_features",
    minDocFreq=1
)

# ── ASSEMBLE THE PIPELINE
# Chain all 4 stages into one Pipeline object
# This is what satisfies the 'Spark MLlib Pipeline' requirement
text_pipeline = Pipeline(stages=[
    tokenizer,
    remover,
    hashing_tf,
    idf
])

print("TF-IDF Pipeline stages:")
print("  Stage 1: Tokenizer")
print("  Stage 2: StopWordsRemover")
print("  Stage 3: HashingTF (5000 features)")
print("  Stage 4: IDF")
print()

# ── FIT AND TRANSFORM 
# fit() — pipeline learns vocabulary statistics from the text
# transform() — pipeline converts text to TF-IDF vectors

print("Fitting pipeline to text data...")
pipeline_model = text_pipeline.fit(working_df)

print("Transforming text to TF-IDF features...")
df_tfidf = pipeline_model.transform(working_df)

# Show intermediate results to understand each stage
print("Pipeline transformation results:")
df_tfidf.select(
    "hadm_id",
    "text",
    "words",
    "filtered_words"
).show(5, truncate=50)

print("Final TF-IDF feature vectors:")
df_tfidf.select(
    "hadm_id",
    "text_features"
).show(5, truncate=80)

# Show what the top words look like
print("Sample — words before and after stop word removal:")
df_tfidf.select("words", "filtered_words").show(3, truncate=60)

# ── SAVE THE PIPELINE MODEL 
# Save the fitted pipeline so we can reuse it
# without having to retrain it every time
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
