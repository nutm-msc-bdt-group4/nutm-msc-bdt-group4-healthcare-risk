from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator  # type: ignore[import]
import json

spark = SparkSession.builder \
    .appName("NUTM_Group4_Silhouette") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("STEP 7: SILHOUETTE ANALYSIS")
print("Finding the optimal number of clusters (K)")
print("=" * 60)

df = spark.read.parquet(
    "hdfs://localhost:9000/healthcare/processed/df_scaled/"
)

# Same filter as script 06 - exclude median-filled patients
MEDIAN_HR = 85.76153846153846
df = df.filter(df.heart_rate != MEDIAN_HR)

print(f"Patients available for analysis: {df.count()}")
print()

evaluator = ClusteringEvaluator(
    featuresCol="scaled_features",
    predictionCol="cluster",
    metricName="silhouette"
)

# Test K=2 to K=8; upper bound chosen to avoid single-patient clusters at n=69
k_values = [2, 3, 4, 5, 6, 7, 8]
results = []

print("Testing different values of K...")
print("-" * 40)

for k in k_values:
    bkm = BisectingKMeans(
        featuresCol="scaled_features",
        predictionCol="cluster",
        k=k,
        seed=42,
        maxIter=30,
        minDivisibleClusterSize=2
    )

    model = bkm.fit(df)
    predictions = model.transform(df)

    # Algorithm may produce fewer clusters than K if data does not support that many splits
    actual_clusters = predictions.select("cluster").distinct().count()

    if actual_clusters > 1:
        score = evaluator.evaluate(predictions)
        results.append({
            "k": k,
            "actual_clusters": actual_clusters,
            "silhouette": round(score, 4)
        })
        print(f"K={k}: actual clusters={actual_clusters}, "
              f"Silhouette Score={score:.4f}")
    else:
        print(f"K={k}: algorithm only produced 1 cluster - skipping")

print("-" * 40)
print()

if results:
    best = max(results, key=lambda x: x["silhouette"])
    print(f"BEST K = {best['k']}")
    print(f"Silhouette Score = {best['silhouette']}")
    print()

    print("Full results summary:")
    print(f"{'K':>4} | {'Actual Clusters':>15} | {'Silhouette Score':>16}")
    print("-" * 42)
    for r in results:
        marker = " <-- BEST" if r["k"] == best["k"] else ""
        print(f"{r['k']:>4} | {r['actual_clusters']:>15} | "
              f"{r['silhouette']:>16.4f}{marker}")
    print()

    output = {
        "results": results,
        "best_k": best["k"],
        "best_score": best["silhouette"]
    }

    output_path = "/home/hadoop/project/outputs/silhouette_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_path}")

else:
    print("No valid results - all K values produced only 1 cluster.")

print()
print("=" * 60)
print("SILHOUETTE ANALYSIS COMPLETE")
print("=" * 60)

spark.stop()
