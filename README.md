# 🏥 ICU Risk Cluster: Healthcare Risk Stratification Pipeline

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://icuriskcluster.streamlit.app)
[![PySpark](https://img.shields.io/badge/PySpark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)](https://spark.apache.org)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

> Big Data ML pipeline for patient risk stratification — built with PySpark, Hadoop/HDFS, and deployed as a Streamlit web application.

---

## 🔍 Project Overview

A group project completed as part of the Big Data Technologies course — MSc Data Science, Nigerian University of Technology and Management (NUTM), 2024/2025.

The pipeline ingests patient health data, processes it at scale using Apache Spark on HDFS, applies unsupervised clustering to stratify patients by risk profile, and surfaces results through an interactive Streamlit dashboard.

**Team:** NUTM MSc Data Science — Group 4  
**Project Lead & Primary Contributor:** Precious Faseyosan

---

## ⚙️ Pipeline Architecture

```
Raw Data (HDFS)
    ↓
PySpark Data Cleaning & Feature Engineering
    ↓
ML Clustering Model (risk stratification)
    ↓
Streamlit Dashboard (interactive results)
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Distributed Computing | Apache Spark (PySpark), Hadoop HDFS |
| Machine Learning | PySpark MLlib |
| Visualisation & App | Streamlit, Plotly |
| Language | Python 3.9+ |

---

## 🚀 Running the App

Live at: **[https://icuriskcluster.streamlit.app](https://icuriskcluster.streamlit.app)**

To run locally:
```bash
git clone https://github.com/nutm-msc-bdt-group4/nutm-msc-bdt-group4-healthcare-risk.git
cd nutm-msc-bdt-group4-healthcare-risk
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚠️ Limitations & Future Work

**Current limitations:**
- Model trained on a demo dataset; results on clinical data require validation against medical ground truth.
- Clustering is unsupervised — risk labels are data-driven, not clinically certified.

**Future improvements:**
- Incorporate supervised classification with labelled clinical outcomes
- Scale pipeline to real-time patient data ingestion
- Add explainability layer (SHAP) for clinical interpretability

---

## 👤 Primary Contributor

**Precious Faseyosan**  
Graduate Petroleum Engineer | MSc Data Science Candidate  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/precious-faseyosan)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github)](https://github.com/PreciousFaseyosan)
