from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator  # type: ignore[import]
import json

# ── START SPARK ───────────────────────────────────────────
spark = SparkSession.builder \
    .appName("NUTM_Group4_Silhouette") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("STEP 7: SILHOUETTE ANALYSIS")
print("Finding the optimal number of clusters (K)")
print("=" * 60)

# ── LOAD DATA ─────────────────────────────────────────────
df = spark.read.parquet(
    "hdfs://localhost:9000/healthcare/processed/df_scaled/"
)

# Keep only patients with real vital sign data
# Same filter we applied in the clustering script
MEDIAN_HR = 85.76153846153846
df = df.filter(df.heart_rate != MEDIAN_HR)

print(f"Patients available for analysis: {df.count()}")
print()

# ── SET UP THE EVALUATOR ──────────────────────────────────
# We reuse the same evaluator for every value of K
evaluator = ClusteringEvaluator(
    featuresCol="scaled_features",
    predictionCol="cluster",
    metricName="silhouette"
)

# ── TEST K FROM 2 TO 8 ────────────────────────────────────
# We test a range of K values.
# K=2 is the minimum (you need at least 2 clusters to compare)
# K=8 is a reasonable maximum for 69 patients
# Going higher risks creating clusters with only 1-2 patients

k_values = [2, 3, 4, 5, 6, 7, 8]
results = []

print("Testing different values of K...")
print("-" * 40)

for k in k_values:
    # Train a new model for each K value
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

    # Count how many distinct clusters were actually created
    # Sometimes the algorithm creates fewer than K if data
    # doesn't support that many splits
    actual_clusters = predictions.select("cluster") \
        .distinct().count()

    # Only calculate Silhouette if we got more than 1 cluster
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
        print(f"K={k}: algorithm only produced 1 cluster "
              f"— skipping")

print("-" * 40)
print()

# ── FIND THE BEST K ───────────────────────────────────────
if results:
    best = max(results, key=lambda x: x["silhouette"])
    print(f"BEST K = {best['k']}")
    print(f"Silhouette Score = {best['silhouette']}")
    print()

    # ── DISPLAY FULL RESULTS TABLE ────────────────────────
    print("Full results summary:")
    print(f"{'K':>4} | {'Actual Clusters':>15} | {'Silhouette Score':>16}")
    print("-" * 42)
    for r in results:
        marker = " <-- BEST" if r["k"] == best["k"] else ""
        print(f"{r['k']:>4} | {r['actual_clusters']:>15} | "
              f"{r['silhouette']:>16.4f}{marker}")
    print()

    # ── SAVE RESULTS ──────────────────────────────────────
    # Save as JSON so the dashboard script can read it
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
    print("No valid results — all K values produced only 1 cluster.")

print()
print("=" * 60)
print("SILHOUETTE ANALYSIS COMPLETE")
print("=" * 60)

spark.stop()
