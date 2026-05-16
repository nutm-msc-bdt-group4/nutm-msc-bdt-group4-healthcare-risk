# Nigerian University of Technology and Management (NUTM)
## Big Data Technologies - NUTDTS 805
### Healthcare Risk Stratification — Group 4

### Group  Members
- Precious Faseyosan, 252325005 — Data Ingestion & Joining
- Alpha G. Gray — Data Cleaning & Feature Engineering
- Gbekeloluwa Laoye — TF-IDF, Clustering & Silhouette Analysis
- Philip Oricha  — Final Model, Dashboard & Visualisation

### Project Overview
An end-to-end big data pipeline built with Apache Hadoop (HDFS)
and Apache Spark to stratify ICU patients by mortality risk
using the MIMIC-III Clinical Database.

### Environment
- Hadoop 3.4.2
- Spark 3.5.0
- Python 3.12.3
- Ubuntu 24 on VirtualBox

### How to Run
1. Start Hadoop: start-dfs.sh && start-yarn.sh
2. Run scripts in order:
  e.g.,  spark-submit notebooks/ingestion/01_explore.py
3. Continue sequentially through all 9 scripts

### Dataset
MIMIC-III Demo Dataset — 100 patients
Source: physionet.org/content/mimiciii-demo/
