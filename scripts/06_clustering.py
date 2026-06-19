from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

spark = SparkSession.builder \
    .appName("NUTM_Group4_Clustering") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("STEP 6: BISECTING K-MEANS CLUSTERING")
print("=" * 60)

df = spark.read.parquet(
    "hdfs://localhost:9000/healthcare/processed/df_scaled/"
)

print("Data loaded. Row count:", df.count())
print("Columns available:", df.columns)
print()

# Exclude patients whose vitals were median-filled (heart_rate == median value)
# These patients lack real vital sign data and would distort cluster separation
MEDIAN_HR = 85.76153846153846

df = df.filter(df.heart_rate != MEDIAN_HR)

print("Patients with real vital sign data:", df.count())
print("(Patients with median-filled vitals excluded from clustering)")
print()

# Initial run with K=4; silhouette analysis in script 07 determines optimal K
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

bkm_model = bkm.fit(df)
predictions_df = bkm_model.transform(df)

print("Clustering complete!")
print()

print("Patient count per cluster:")
predictions_df.groupBy("cluster") \
    .count() \
    .orderBy("cluster") \
    .show()

# Silhouette score: -1 to 1; higher = better-separated clusters
evaluator = ClusteringEvaluator(
    featuresCol="scaled_features",
    predictionCol="cluster",
    metricName="silhouette"
)

score = evaluator.evaluate(predictions_df)
print(f"Silhouette Score for K=4: {score:.4f}")
print()

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

print("Clusters ranked by mortality rate (highest first):")
cluster_profile.orderBy("mortality_pct", ascending=False).show()

predictions_df.write \
    .mode("overwrite") \
    .parquet("hdfs://localhost:9000/healthcare/processed/predictions_k4/")

bkm_model.write().overwrite().save(
    "hdfs://localhost:9000/healthcare/models/bkm_k4"
)

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
