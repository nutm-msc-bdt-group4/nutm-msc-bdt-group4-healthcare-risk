from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# START SPARK 
spark = SparkSession.builder \
    .appName("NUTM_Group4_Clustering") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("STEP 6: BISECTING K-MEANS CLUSTERING")
print("=" * 60)

# ── LOAD SCALED FEATURES FROM HDFS 
# We load the scaled dataset we created in Step 4.
# The 'scaled_features' column is what the model will use.

df = spark.read.parquet(
    "hdfs://localhost:9000/healthcare/processed/df_scaled/"
)

print("Data loaded. Row count:", df.count())
print("Columns available:", df.columns)
print()

# Filter to only patients with REAL vital sign data
# Patients with median-filled vitals all share heart_rate = 85.76
# These patients lack sufficient clinical data for meaningful clustering
MEDIAN_HR = 85.76153846153846

df = df.filter(df.heart_rate != MEDIAN_HR)

print("Patients with real vital sign data:", df.count())
print("(Patients with median-filled vitals excluded from clustering)")
print()

# ── TRAIN BISECTING K-MEANS WITH K=4 
# We start with K=4 as our initial attempt.
# The next script (silhouette analysis) will tell us
# whether 4 is actually the best number of clusters.
#
# featuresCol: the column containing our feature vectors
# predictionCol: the name of the new column Spark will add
#                containing each patient's cluster number
# k: number of clusters we want
# seed: fixes randomness for reproducibility

print("Training Bisecting K-Means model with K=4...")
print("This may take a minute...")
print()

bkm = BisectingKMeans(
    featuresCol="scaled_features",
    predictionCol="cluster",
    k=4,
    seed=42,
    maxIter=30,
    minDivisibleClusterSize=2
)

# fit() is where the actual training happens
# Spark analyses all the feature vectors and finds
# the best way to split patients into 4 groups
bkm_model = bkm.fit(df)

# transform() assigns each patient to their cluster
# It adds a new column called 'cluster' with values 0, 1, 2, or 3
predictions_df = bkm_model.transform(df)

print("Clustering complete!")
print()

# ── HOW MANY PATIENTS IN EACH CLUSTER?
print("Patient count per cluster:")
predictions_df.groupBy("cluster") \
    .count() \
    .orderBy("cluster") \
    .show()

# ── CALCULATE SILHOUETTE SCORE 
# The Silhouette score measures how well separated
# the clusters are. Range is -1 to 1.
# Closer to 1.0 = patients within a cluster are very
#                 similar to each other AND very different
#                 from patients in other clusters
# Closer to 0.0 = clusters are overlapping
# Negative      = patients may be in the wrong cluster

evaluator = ClusteringEvaluator(
    featuresCol="scaled_features",
    predictionCol="cluster",
    metricName="silhouette"
)

score = evaluator.evaluate(predictions_df)
print(f"Silhouette Score for K=4: {score:.4f}")
print()

# ── EXAMINE THE CLUSTER CENTRES 
# The cluster centre is the 'average patient' in each group
# It tells us what the typical patient in each cluster
# looks like in the scaled feature space

print("Cluster centres (in scaled feature space):")
centers = bkm_model.clusterCenters()
feature_names = [
    "age", "los", "gender",
    "heart_rate", "systolic_bp", "diastolic_bp",
    "temperature", "respiratory_rate", "spo2"
]

for i, center in enumerate(centers):
    print(f"\nCluster {i} centre:")
    for name, value in zip(feature_names, center):
        print(f"  {name:20s}: {value:+.4f}")

print()

# ── PROFILE EACH CLUSTER WITH REAL VALUES 
# The cluster centres are in scaled space (hard to interpret)
# Let's look at the ORIGINAL unscaled values instead
# This tells us what each cluster actually looks like
# in real clinical terms

print("=" * 60)
print("CLUSTER PROFILES (original unscaled values):")
print("=" * 60)

cluster_profile = predictions_df.groupBy("cluster").agg(
    F.count("subject_id").alias("patient_count"),
    F.round(F.avg("age"), 1).alias("avg_age"),
    F.round(F.avg("los"), 2).alias("avg_los_days"),
    F.round(F.avg("heart_rate"), 1).alias("avg_heart_rate"),
    F.round(F.avg("systolic_bp"), 1).alias("avg_systolic_bp"),
    F.round(F.avg("diastolic_bp"), 1).alias("avg_diastolic_bp"),
    F.round(F.avg("respiratory_rate"), 1).alias("avg_resp_rate"),
    F.round(F.avg("spo2"), 1).alias("avg_spo2"),
    F.round(F.avg("hospital_expire_flag") * 100, 1).alias("mortality_pct")
).orderBy("cluster")

cluster_profile.show(truncate=False)

# ── IDENTIFY THE HIGH RISK CLUSTER 
# The high risk cluster is the one with the highest
# mortality percentage — these are the patients most
# likely to die during their hospital admission

print("Clusters ranked by mortality rate (highest first):")
cluster_profile.orderBy("mortality_pct", ascending=False).show()

# ── SAVE EVERYTHING TO HDFS 
# Save the predictions dataframe — patients with cluster labels
predictions_df.write \
    .mode("overwrite") \
    .parquet("hdfs://localhost:9000/healthcare/processed/predictions_k4/")

# Save the model itself
bkm_model.write().overwrite().save(
    "hdfs://localhost:9000/healthcare/models/bkm_k4"
)

# Save cluster profiles as CSV for the dashboard later
cluster_profile.coalesce(1).write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("/home/hadoop/project/outputs/cluster_profiles_k4/")

print()
print("=" * 60)
print("Model saved to HDFS.")
print("Predictions saved to HDFS.")
print("Cluster profiles saved to local outputs folder.")
print("=" * 60)

spark.stop()
