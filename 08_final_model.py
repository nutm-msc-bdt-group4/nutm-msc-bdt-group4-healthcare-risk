from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import json

# Start Spark
spark = SparkSession.builder \
    .appName("NUTM_Group4_FinalModel") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("STEP 8: FINAL MODEL — BISECTING K-MEANS K=2")
print("=" * 60)

# Load Data

df = spark.read.parquet(
    "hdfs://localhost:9000/healthcare/processed/df_scaled/"
)

# Apply same filter as before

MEDIAN_HR = 85.76153846153846
df = df.filter(df.heart_rate != MEDIAN_HR)

print(f"Patients for clustering: {df.count()}")
print()

#Train Final Model With Best K=2
print("Training final Bisecting K-Means model with K=2...")

final_bkm = BisectingKMeans(
    featuresCol="scaled_features",
    predictionCol="cluster",
    k=2,
    seed=42,
    maxIter=30,
    minDivisibleClusterSize=2
)

final_model = final_bkm.fit(df)
final_predictions = final_model.transform(df)

print("Training complete!")
print()

# Confirm Silhouette Score

evaluator = ClusteringEvaluator(
    featuresCol="scaled_features",
    predictionCol="cluster",
    metricName="silhouette"
)

final_score = evaluator.evaluate(final_predictions)
print(f"Final Silhouette Score (K=2): {final_score:.4f}")
print()

# Patient Count Per Cluster
print("Patient distribution:")
final_predictions.groupBy("cluster") \
    .count() \
    .orderBy("cluster") \
    .show()

# Deep Cluster Profiles
# Now we profile each cluster using original unscaled values
# This tells us what each group actually looks like
# in real clinical terms

print("=" * 60)
print("DEEP CLUSTER PROFILES")
print("=" * 60)

profile = final_predictions.groupBy("cluster").agg(
    F.count("subject_id").alias("patient_count"),

    # Demographics
    F.round(F.avg("age"), 1).alias("avg_age"),
    F.round(F.min("age"), 1).alias("min_age"),
    F.round(F.max("age"), 1).alias("max_age"),

    # ICU stay

    F.round(F.avg("los"), 2).alias("avg_los_days"),
    F.round(F.max("los"), 2).alias("max_los_days"),

    # Vital signs
    F.round(F.avg("heart_rate"), 1).alias("avg_heart_rate"),
    F.round(F.avg("systolic_bp"), 1).alias("avg_systolic_bp"),
    F.round(F.avg("diastolic_bp"), 1).alias("avg_diastolic_bp"),
    F.round(F.avg("respiratory_rate"), 1).alias("avg_resp_rate"),
    F.round(F.avg("spo2"), 1).alias("avg_spo2"),

    # Outcomes
    F.round(
        F.avg("hospital_expire_flag") * 100, 1
    ).alias("mortality_pct"),

    # Gender breakdown
    F.round(
        F.avg("gender_encoded") * 100, 1
    ).alias("pct_male")

).orderBy("cluster")

print("Full cluster comparison:")
profile.show(truncate=False)

# Identify and Label Clusters
# Collect results to identify which cluster is high risk

profile_data = profile.collect()

for row in profile_data:
    print(f"\nCluster {row['cluster']} Summary:")
    print(f"  Patients:          {row['patient_count']}")
    print(f"  Avg Age:           {row['avg_age']} years")
    print(f"  Avg ICU Stay:      {row['avg_los_days']} days")
    print(f"  Avg Heart Rate:    {row['avg_heart_rate']} bpm")
    print(f"  Avg Systolic BP:   {row['avg_systolic_bp']} mmHg")
    print(f"  Avg Resp Rate:     {row['avg_resp_rate']} breaths/min")
    print(f"  Avg SpO2:          {row['avg_spo2']} %")
    print(f"  Mortality Rate:    {row['mortality_pct']}%")
    print(f"  % Male:            {row['pct_male']}%")

# Identify high risk cluster
high_risk = max(profile_data, key=lambda x: x["mortality_pct"])
low_risk = min(profile_data, key=lambda x: x["mortality_pct"])

print()
print("=" * 60)
print(f"HIGH RISK CLUSTER: Cluster {high_risk['cluster']}")
print(f"  Mortality rate: {high_risk['mortality_pct']}%")
print(f"  Average ICU stay: {high_risk['avg_los_days']} days")
print(f"  Average heart rate: {high_risk['avg_heart_rate']} bpm")
print()
print(f"LOW RISK CLUSTER: Cluster {low_risk['cluster']}")
print(f"  Mortality rate: {low_risk['mortality_pct']}%")
print(f"  Average ICU stay: {low_risk['avg_los_days']} days")
print(f"  Average heart rate: {low_risk['avg_heart_rate']} bpm")
print("=" * 60)

# Care Unit Breakdown Per Cluster
print()
print("ICU care unit distribution per cluster:")
final_predictions.groupBy("cluster", "first_careunit") \
    .count() \
    .orderBy("cluster", "count", ascending=[True, False]) \
    .show()

# Diagnosis Breakdown Per Cluster
print("Top 5 diagnoses in HIGH RISK cluster:")
final_predictions.filter(
    F.col("cluster") == int(high_risk["cluster"])
).groupBy("diagnosis") \
    .count() \
    .orderBy("count", ascending=False) \
    .show(5, truncate=False)

print("Top 5 diagnoses in LOW RISK cluster:")
final_predictions.filter(
    F.col("cluster") == int(low_risk["cluster"])
).groupBy("diagnosis") \
    .count() \
    .orderBy("count", ascending=False) \
    .show(5, truncate=False)

# Save Everything
# Save final model
final_model.write().overwrite().save(
    "hdfs://localhost:9000/healthcare/models/bkm_final_k2"
)

# Save predictions
final_predictions.write \
    .mode("overwrite") \
    .parquet(
        "hdfs://localhost:9000/healthcare/processed/final_predictions/"
    )

# Save profile as CSV for dashboard

profile.coalesce(1).write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("/home/hadoop/project/outputs/final_cluster_profiles/")

# Save summary as JSON for dashboard

summary = {
    "best_k": 2,
    "silhouette_score": round(final_score, 4),
    "high_risk_cluster": int(high_risk["cluster"]),
    "low_risk_cluster": int(low_risk["cluster"]),
    "high_risk_mortality": float(high_risk["mortality_pct"]),
    "low_risk_mortality": float(low_risk["mortality_pct"])
}

with open("/home/hadoop/project/outputs/model_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print()
print("=" * 60)
print("Final model saved to HDFS.")
print("Final predictions saved to HDFS.")
print("Cluster profiles saved to outputs folder.")
print("=" * 60)

spark.stop()