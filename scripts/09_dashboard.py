import pandas as pd
import matplotlib

# Use non-interactive backend — required when running via SSH without a display screen connected

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import json
import os

print("=" * 60)
print("STEP 9: BUILDING THE RISK PROFILE DASHBOARD")
print("=" * 60)

# Load Cluster Profiles
# Find the CSV file saved by the final model script

profile_pd = pd.read_csv(
    "/home/hadoop/project/outputs/cluster_profiles.csv"
)


# Load the model summary JSON

with open("/home/hadoop/project/outputs/model_summary.json") as f:
    summary = json.load(f)

high_risk = summary["high_risk_cluster"]
low_risk = summary["low_risk_cluster"]
silhouette = summary["silhouette_score"]

print(f"High risk cluster: {high_risk}")
print(f"Low risk cluster:  {low_risk}")
print(f"Silhouette score:  {silhouette}")
print()

# Prepare Data 
# Sort so cluster 1 always comes first

profile_pd = profile_pd.sort_values("cluster").reset_index(drop=True)

# Assign meaningful labels to each cluster

def get_label(cluster_num):
    if cluster_num == high_risk:
        return "Cluster 1\nHIGH RISK"
    else:
        return "Cluster 2\nLOW RISK"

profile_pd["label"] = profile_pd["cluster"].apply(get_label)

# Define colors — red for high risk, green for low risk

def get_color(cluster_num):
    if cluster_num == high_risk:
        return "#E74C3C"   # red
    else:
        return "#2ECC71"   # green

profile_pd["color"] = profile_pd["cluster"].apply(get_color)
colors = profile_pd["color"].tolist()
labels = profile_pd["label"].tolist()

print("Labels assigned:")
print(profile_pd[["cluster", "label", "mortality_pct"]])
print()

# Build the dashboard
# Set overall style
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"

# Create figure with 3 rows and 3 columns of subplots
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("white")

# Main title
fig.suptitle(
    "Patient Risk Stratification Dashboard\n"
    "Nigeria University of Technology and Management — Group 4\n"
    f"Bisecting K-Means Clustering (K=2) | "
    f"Silhouette Score: {silhouette} | "
    f"MIMIC-III Demo Dataset",
    fontsize=14,
    fontweight="bold",
    y=0.98,
    linespacing=1.6
)

# Grid layout — 3 rows, 3 columns
# Last row spans all 3 columns for the heatmap
gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    hspace=0.55,
    wspace=0.38
)

# Chart 1: Pie Chart-Patient Distribution
ax1 = fig.add_subplot(gs[0, 0])
wedges, texts, autotexts = ax1.pie(
    profile_pd["patient_count"],
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 9},
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
for autotext in autotexts:
    autotext.set_fontweight("bold")
ax1.set_title(
    "Patient Distribution\nby Risk Group",
    fontweight="bold",
    fontsize=11,
    pad=10
)

#Chart 2: Mortality Rate
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(
    labels,
    profile_pd["mortality_pct"],
    color=colors,
    edgecolor="white",
    linewidth=2,
    width=0.5
)
ax2.set_title(
    "In-Hospital Mortality Rate",
    fontweight="bold",
    fontsize=11
)
ax2.set_ylabel("Mortality Rate (%)", fontsize=10)
ax2.set_ylim(0, 65)
# Add value labels on top of each bar
for bar, val in zip(bars, profile_pd["mortality_pct"]):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"{val:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=12
    )
ax2.tick_params(axis="x", labelsize=9)

# CHART 3: AVERAGE AGE
ax3 = fig.add_subplot(gs[0, 2])
bars = ax3.bar(
    labels,
    profile_pd["avg_age"],
    color=colors,
    edgecolor="white",
    linewidth=2,
    width=0.5
)
ax3.set_title(
    "Average Patient Age",
    fontweight="bold",
    fontsize=11
)
ax3.set_ylabel("Age (years)", fontsize=10)
ax3.set_ylim(0, 100)
for bar, val in zip(bars, profile_pd["avg_age"]):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=12
    )
ax3.tick_params(axis="x", labelsize=9)

# CHART 4: HEART RATE
ax4 = fig.add_subplot(gs[1, 0])
bars = ax4.bar(
    labels,
    profile_pd["avg_heart_rate"],
    color=colors,
    edgecolor="white",
    linewidth=2,
    width=0.5
)
# Add clinical reference line
ax4.axhline(
    y=100,
    color="orange",
    linestyle="--",
    alpha=0.8,
    linewidth=1.5,
    label="Tachycardia threshold (100 bpm)"
)
ax4.set_title(
    "Average Heart Rate",
    fontweight="bold",
    fontsize=11
)
ax4.set_ylabel("Heart Rate (bpm)", fontsize=10)
ax4.set_ylim(0, 120)
ax4.legend(fontsize=7, loc="upper right")
for bar, val in zip(bars, profile_pd["avg_heart_rate"]):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=12
    )
ax4.tick_params(axis="x", labelsize=9)

# CHART 5: SYSTOLIC BLOOD PRESSURE
ax5 = fig.add_subplot(gs[1, 1])
bars = ax5.bar(
    labels,
    profile_pd["avg_systolic_bp"],
    color=colors,
    edgecolor="white",
    linewidth=2,
    width=0.5
)
# Clinical reference lines
ax5.axhline(
    y=140,
    color="orange",
    linestyle="--",
    alpha=0.8,
    linewidth=1.5,
    label="Hypertension (140 mmHg)"
)
ax5.axhline(
    y=90,
    color="red",
    linestyle="--",
    alpha=0.8,
    linewidth=1.5,
    label="Hypotension (90 mmHg)"
)
ax5.set_title(
    "Average Systolic Blood Pressure",
    fontweight="bold",
    fontsize=11
)
ax5.set_ylabel("Systolic BP (mmHg)", fontsize=10)
ax5.set_ylim(0, 170)
ax5.legend(fontsize=7, loc="upper right")
for bar, val in zip(bars, profile_pd["avg_systolic_bp"]):
    ax5.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=12
    )
ax5.tick_params(axis="x", labelsize=9)

# Chart 6: ICU Length of stay

ax6 = fig.add_subplot(gs[1, 2])
bars = ax6.bar(
    labels,
    profile_pd["avg_los_days"],
    color=colors,
    edgecolor="white",
    linewidth=2,
    width=0.5
)
ax6.set_title(
    "Average ICU Length of Stay",
    fontweight="bold",
    fontsize=11
)
ax6.set_ylabel("Days in ICU", fontsize=10)
ax6.set_ylim(0, 8)
for bar, val in zip(bars, profile_pd["avg_los_days"]):
    ax6.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f"{val:.2f}d",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=12
    )
ax6.tick_params(axis="x", labelsize=9)

# Chart 7: Heatmap — spans full bottom row
ax7 = fig.add_subplot(gs[2, :])

# Build heatmap data
heatmap_data = profile_pd.set_index("label")[[
    "avg_age",
    "avg_los_days",
    "avg_heart_rate",
    "avg_systolic_bp",
    "avg_diastolic_bp",
    "avg_resp_rate",
    "avg_spo2",
    "mortality_pct"
]]

heatmap_data.columns = [
    "Age (yrs)",
    "ICU Stay (days)",
    "Heart Rate (bpm)",
    "Systolic BP (mmHg)",
    "Diastolic BP (mmHg)",
    "Resp Rate (br/min)",
    "SpO2 (%)",
    "Mortality (%)"
]

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn_r",
    ax=ax7,
    linewidths=0.5,
    linecolor="white",
    annot_kws={"size": 12, "weight": "bold"},
    cbar_kws={"label": "Value", "shrink": 0.8}
)

ax7.set_title(
    "Risk Profile Heatmap — All Clinical Features by Cluster\n"
    "(Red = Higher values, Green = Lower values)",
    fontweight="bold",
    fontsize=11,
    pad=10
)
ax7.tick_params(axis="y", rotation=0, labelsize=10)
ax7.tick_params(axis="x", labelsize=10)

# Save the dashboard
output_path = "/home/hadoop/project/outputs/risk_dashboard.png"
plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none"
)

print(f"Dashboard saved to: {output_path}")
print()
print("=" * 60)
print("DASHBOARD COMPLETE")
