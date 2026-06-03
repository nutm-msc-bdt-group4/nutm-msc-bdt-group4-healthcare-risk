
# Healthcare Risk Stratification — Group 4
## Nigeria University of Technology and Management (NUTM)
### Big Data Technologies (NUTDTS 805) | MSc Data Science | 2025/2026

End-to-end distributed Big Data pipeline for ICU patient risk 
stratification. Built with HDFS and Apache Spark (MLlib + NLP) 
using the MIMIC-III dataset to automatically identify high-risk 
patients and deliver actionable healthcare insights.

---

## Team Members

| Name | Student ID | GitHub | Responsibility |
|------|------------|--------|----------------|
| Precious Faseyosan | 252325005 | @PreciousFaseyosan | Data Ingestion & Joining |
| Alpha G. Gray | 252325003 | @alphagbessaygray-max | Data Cleaning & Feature Engineering |
| Laoye Gbekeoluwa | 252325015 | @gbekelaoye | TF-IDF, Clustering & Silhouette Analysis |
| Philip Oricha | 252325008 | @adeizaofficial | Final Model, Dashboard & Visualisation |

**Lecturer:** Dr. Isah Charles Saidu  

---

## Tech Stack

- Apache Hadoop 3.4.2 (HDFS + YARN)
- Apache Spark 3.5.0 (MLlib + PySpark)
- Python 3.12.3
- Java OpenJDK 11
- Dataset: MIMIC-III Clinical Database Demo

---

## Repository Structure

```
scripts/
├── 01_explore.py           # Data exploration and schema inspection
├── 02_joining.py           # Multi-table joins and master table creation
├── 03_vitals.py            # Vital signs extraction and cleaning
├── 04_features.py          # Feature engineering and StandardScaler
├── 05_tfidf.py             # TF-IDF NLP pipeline for clinical notes
├── 06_clustering.py        # Bisecting K-Means initial clustering
├── 07_silhouette.py        # Silhouette analysis for optimal K
├── 08_final_model.py       # Final model training and cluster profiling
└── 09_dashboard.py         # Risk stratification dashboard

data/
├── PATIENTS.csv            # Patient demographics (100 patients)
├── ADMISSIONS.csv          # Hospital admission records (129 rows)
├── ICUSTAYS.csv            # ICU stay details (136 rows)
├── CHARTEVENTS.csv         # Vital sign measurements (758,355 rows)
└── NOTEEVENTS.csv          # Clinical notes (empty in demo dataset)

outputs/
├── risk_dashboard.png      # Six-panel clinical visualisation
├── cluster_profiles.csv    # Cluster summary statistics
├── silhouette_results.json
└── model_summary.json
```

---

## How to Run

### Prerequisites
Start Hadoop services on the project VM:
```bash
start-dfs.sh && start-yarn.sh
jps  # Verify 5 services are running
```

### Run the pipeline in order
```bash
spark-submit scripts/01_explore.py
spark-submit scripts/02_joining.py
spark-submit scripts/03_vitals.py
spark-submit scripts/04_features.py
spark-submit scripts/05_tfidf.py
spark-submit scripts/06_clustering.py
spark-submit scripts/07_silhouette.py
spark-submit scripts/08_final_model.py
python3 scripts/09_dashboard.py
```

---

## Team Workflow

### Branching Strategy
- **main** — Stable, submission-ready code only. Never push here directly.
- **develop** — Primary integration branch. All work is pushed here first.

### Step-by-Step Contribution Guide
```bash
# 1. Always start by pulling the latest develop
git checkout develop
git pull origin develop

# 2. Do your work in the scripts/ folder

# 3. Check what you are about to commit
git status

# 4. Stage and commit
git add .
git commit -m "YourName - Area - What you did"
# Examples:
# "Precious - Ingestion - Add HDFS data loading script"
# "Philip - Modelling - Add Bisecting K-Means clustering"

# 5. Push to develop
git push origin develop

# 6. Open a Pull Request on GitHub: develop → main
#    At least one teammate must review before merging
```

### Commit Message Format

| Format | Example |
|--------|---------|
| `Name - Area - Description` | `Precious - Ingestion - Add HDFS loading and joining scripts` |

Common area labels: `Ingestion`, `Processing`, `Modelling`, `Results`, `Docs`, `Fix`

### Rules
- Never push directly to `main`
- Always pull `develop` before starting work
- Only commit scripts assigned to you
- Never edit another member's scripts
- Every member must have meaningful commits in the Git history

---

## Key Results

(to be updated...)
