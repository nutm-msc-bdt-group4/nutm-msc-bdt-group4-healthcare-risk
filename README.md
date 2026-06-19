# 🏥 ICU Risk Stratification: End-to-End Big Data Pipeline

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://icuriskcluster.streamlit.app)
[![PySpark](https://img.shields.io/badge/PySpark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)](https://spark.apache.org)
[![Hadoop](https://img.shields.io/badge/Hadoop-66CCFF?style=for-the-badge&logo=apachehadoop&logoColor=black)](https://hadoop.apache.org)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

> End-to-end distributed Big Data pipeline for ICU patient risk
> stratification — built with Apache Hadoop and Apache Spark,
> deployed as a Streamlit clinical dashboard.

---

## 🔍 Project Overview

An end-to-end pipeline that ingests, processes, and analyses clinical
data from the MIMIC-III ICU database to automatically stratify patients
into risk groups — without using any prior knowledge of patient outcomes.

**Key result:** Bisecting K-Means (K=2) identified two clinically
meaningful patient groups from vital sign data alone:

| Cluster | Patients | In-Hospital Mortality | Avg Systolic BP | Avg ICU Stay |
|---------|----------|----------------------|-----------------|--------------|
| **High Risk** | 34 (49.3%) | **50.0%** | 108.1 mmHg | 5.25 days |
| **Low Risk** | 35 (50.7%) | **8.6%** | 131.4 mmHg | 3.44 days |

A 41.4 percentage point mortality gap — discovered purely from
routinely collected vital sign measurements.

---

## 🏗️ Pipeline Architecture

Nine sequential PySpark scripts, each executed via `spark-submit`,
with intermediate outputs saved to HDFS in Parquet format:

| Script | Operation | Output |
|--------|-----------|--------|
| 01 | Exploration — schema inspection, row counts, data profiling | Analytical insights |
| 02 | Joining — multi-table joins, age calculation, age filtering | master_patients/ (Parquet) |
| 03 | Vitals Processing — ITEMID filtering, outlier removal, pivot, imputation | master_with_vitals/ (Parquet) |
| 04 | Feature Engineering — VectorAssembler, StandardScaler | df_scaled/ + scaler_model/ |
| 05 | TF-IDF Pipeline — Tokenizer, StopWordsRemover, HashingTF, IDF | tfidf_pipeline/ |
| 06 | Initial Clustering — Bisecting K-Means K=4, baseline Silhouette | predictions_k4/ |
| 07 | Silhouette Analysis — K=2 to K=8 evaluation, optimal K selection | silhouette_results.json |
| 08 | Final Model — Bisecting K-Means K=2, deep cluster profiling | final_predictions/ + bkm_final_k2/ |
| 09 | Dashboard — six-panel clinical visualisation | risk_dashboard.png |

---

## 📊 Silhouette Analysis Results

| K | Silhouette Score | Assessment |
|---|-----------------|------------|
| **2** | **0.2449** | **Best — selected** |
| 3 | 0.1753 | Notable drop |
| 4 | 0.1958 | Partial recovery |
| 5 | 0.1666 | Declining |
| 6 | 0.1630 | Declining |
| 7 | 0.1398 | Lowest |
| 8 | 0.1426 | Marginal recovery |

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Distributed Storage | Apache Hadoop HDFS 3.4.2 |
| Distributed Processing | Apache Spark 3.5.0 (PySpark) |
| Machine Learning | Spark MLlib (Bisecting K-Means, StandardScaler, TF-IDF Pipeline) |
| Visualisation | matplotlib, seaborn |
| Dashboard | Streamlit, Plotly |
| Infrastructure | VirtualBox Ubuntu VM, single-node pseudo-distributed cluster |

---

## 🗄️ HDFS Storage Architecture

```
/healthcare/
├── raw/                          # Original CSV files from MIMIC-III
├── processed/
│   ├── master_patients/          # Joined patient-admission-ICU table
│   ├── master_with_vitals/       # Vital signs, partitioned by care unit
│   ├── df_scaled/                # Feature-engineered and scaled dataset
│   └── final_predictions/        # Dataset with cluster assignments
└── models/
    ├── scaler_model/             # Saved StandardScaler
    ├── tfidf_pipeline/           # Saved TF-IDF Pipeline
    └── bkm_final_k2/            # Saved Bisecting K-Means (K=2)
```

Processed data stored in Parquet with partition pruning on
`first_careunit` — reduces I/O by up to 80% on care-unit-specific queries.

---

## 📁 Repository Structure

```
icu-risk-stratification/
├── scripts/         # PySpark pipeline scripts (01-09)
├── models/          # Saved Spark ML model artefacts
├── outputs/         # Dashboard visualisations
└── README.md
```

> **Data note:** The MIMIC-III dataset is not included. Access the
> demo dataset free at physionet.org. The full dataset (46,000+ patients)
> requires credentialed access.

---

## 🚀 Running the Pipeline

### Prerequisites
- Apache Hadoop 3.4.2 + HDFS running
- Apache Spark 3.5.0 installed
- Python 3.9+ with PySpark, matplotlib, seaborn, pandas

### Execution

```bash
# Load data into HDFS
hdfs dfs -put MIMIC_III_files/ /healthcare/raw/

# Run pipeline scripts in sequence
spark-submit scripts/01_exploration.py
spark-submit scripts/02_joining.py
spark-submit scripts/03_vitals_processing.py
spark-submit scripts/04_feature_engineering.py
spark-submit scripts/05_tfidf_pipeline.py
spark-submit scripts/06_initial_clustering.py
spark-submit scripts/07_silhouette_analysis.py
spark-submit scripts/08_final_model.py
spark-submit scripts/09_dashboard.py
```

The Streamlit dashboard is deployed at:
**[https://icuriskcluster.streamlit.app](https://icuriskcluster.streamlit.app)**

---

## ⚠️ Limitations & Future Work

**Current limitations:**
- Demo dataset only (100 patients, 69 with complete vital signs) —
  moderate Silhouette score reflects limited sample size, not pipeline quality.
- Clinical notes empty in demo — TF-IDF pipeline validated on synthetic
  text; full contribution pending real NOTEEVENTS data.
- Temperature excluded from features due to 97% missing data in demo.
- Single-node VM deployment — production-grade code, not production scale.

**Future improvements:**
- Scale to full MIMIC-III (46,000+ patients, 330M+ chart events)
- Replace TF-IDF with ClinicalBERT embeddings for richer text features
- Add laboratory values (lactate, creatinine, WBC) to feature set
- Build supervised classification layer for real-time admission-time risk prediction
- Deploy on multi-node Hadoop cluster

---

## 👤 Contributors

**Precious Faseyosan**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/precious-faseyosan)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github)](https://github.com/PreciousFaseyosan)

**Philip Oricha**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/philiporicha)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github)](https://github.com/adeizaofficial)

**Gbekeloluwa Laoye**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/laoye-gbekeloluwa-216b95150)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github)](https://github.com/laoyegbekeloluwa)

**Alpha G. Gray**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/alpha-g-gray-8474a562)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github)](https://github.com/alphagbessaygray-max)

*MSc Data Science — Big Data Technologies, NUTM, 2025/2026.*
