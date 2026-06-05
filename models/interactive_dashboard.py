import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="ICU Risk Stratification Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #FFFFFF;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
        border-radius: 8px;
        padding: 16px;
    }

    /* Headers */
    h1, h2, h3 {
        color: #1B3A6B;
        font-family: Arial, sans-serif;
    }

    /* High risk color */
    .high-risk {
        color: #E74C3C;
        font-weight: bold;
        font-size: 1.4em;
    }

    /* Low risk color */
    .low-risk {
        color: #2ECC71;
        font-weight: bold;
        font-size: 1.4em;
    }

    /* Result box */
    .result-box {
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin: 16px 0;
    }

    .result-high {
        background-color: #FDF0F0;
        border: 2px solid #E74C3C;
    }

    .result-low {
        background-color: #EAF7EE;
        border: 2px solid #2ECC71;
    }

    /* Divider */
    hr {
        border: 1px solid #E9ECEF;
        margin: 24px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── CLUSTER DATA ──────────────────────────────────────────
# These are the actual results from our Bisecting K-Means
# model trained on the MIMIC-III demo dataset

cluster_data = pd.DataFrame({
    "Cluster": ["Cluster 1: HIGH RISK", "Cluster 2: LOW RISK"],
    "Patients": [34, 35],
    "Mortality (%)": [50.0, 8.6],
    "Avg Age (yrs)": [72.9, 67.2],
    "Avg ICU Stay (days)": [5.25, 3.44],
    "Avg Heart Rate (bpm)": [81.8, 89.2],
    "Avg Systolic BP (mmHg)": [108.1, 131.4],
    "Avg Diastolic BP (mmHg)": [54.8, 69.7],
    "Avg Resp Rate (br/min)": [19.1, 19.8],
    "Avg SpO2 (%)": [96.8, 96.7],
})

# Cluster centre values (scaled) — from our trained model
# These are the mean values used for distance calculation
HIGH_RISK_CENTRE = {
    "age": 72.9, "los": 5.25, "heart_rate": 81.8,
    "systolic_bp": 108.1, "diastolic_bp": 54.8,
    "respiratory_rate": 19.1, "spo2": 96.8
}
LOW_RISK_CENTRE = {
    "age": 67.2, "los": 3.44, "heart_rate": 89.2,
    "systolic_bp": 131.4, "diastolic_bp": 69.7,
    "respiratory_rate": 19.8, "spo2": 96.7
}

# Standard deviations from our dataset (for normalisation)
STD_VALUES = {
    "age": 15.2, "los": 6.2, "heart_rate": 10.4,
    "systolic_bp": 13.9, "diastolic_bp": 7.5,
    "respiratory_rate": 3.3, "spo2": 1.4
}
MEAN_VALUES = {
    "age": 69.9, "los": 4.5, "heart_rate": 85.6,
    "systolic_bp": 117.6, "diastolic_bp": 64.3,
    "respiratory_rate": 19.5, "spo2": 96.8
}

# ── PREDICTION FUNCTION ───────────────────────────────────
def predict_cluster(age, los, heart_rate, systolic_bp,
                    diastolic_bp, respiratory_rate, spo2):
    """
    Predicts which cluster a patient belongs to using
    Euclidean distance to each cluster centre.
    Replicates the StandardScaler + K-Means prediction
    from our Spark pipeline.
    """
    patient = {
        "age": age, "los": los, "heart_rate": heart_rate,
        "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
        "respiratory_rate": respiratory_rate, "spo2": spo2
    }

    # Scale the patient features (same as StandardScaler in pipeline)
    scaled = {}
    for key, value in patient.items():
        scaled[key] = (value - MEAN_VALUES[key]) / STD_VALUES[key]

    # Scale cluster centres
    high_scaled = {}
    low_scaled = {}
    for key in patient:
        high_scaled[key] = (
            HIGH_RISK_CENTRE[key] - MEAN_VALUES[key]
        ) / STD_VALUES[key]
        low_scaled[key] = (
            LOW_RISK_CENTRE[key] - MEAN_VALUES[key]
        ) / STD_VALUES[key]

    # Calculate Euclidean distance to each cluster centre
    dist_high = np.sqrt(sum(
        (scaled[k] - high_scaled[k]) ** 2 for k in scaled
    ))
    dist_low = np.sqrt(sum(
        (scaled[k] - low_scaled[k]) ** 2 for k in scaled
    ))

    # Return the nearest cluster
    if dist_high < dist_low:
        return "HIGH RISK", dist_high, dist_low
    else:
        return "LOW RISK", dist_high, dist_low

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/hospital.png",
        width=60
    )
    st.markdown("## 🏥 ICU Risk Dashboard")
    st.markdown("**Group 4**")
    st.markdown("Big Data Technologies 2025/2026")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "Select View",
        ["📊 Cluster Overview",
         "🔍 Patient Risk Predictor",
         "📋 Data Explorer"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown("**Algorithm:** Bisecting K-Means")
    st.markdown("**Optimal K:** 2")
    st.markdown("**Silhouette Score:** 0.2449")
    st.markdown("**Dataset:** MIMIC-III Demo")
    st.markdown("**Patients clustered:** 69")

# ══════════════════════════════════════════════════════════
# PAGE 1: CLUSTER OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "📊 Cluster Overview":

    st.title("ICU Patient Risk Stratification Dashboard")
    st.markdown(
        "Healthcare Risk Stratification using Bisecting K-Means "
        "Clustering on MIMIC-III Clinical Data | Group 4"
    )
    st.markdown("---")

    # ── KEY METRICS ───────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Patients Clustered",
            value="69",
            delta="from 100 in dataset"
        )
    with col2:
        st.metric(
            label="High Risk Mortality",
            value="50.0%",
            delta="-41.4% vs Low Risk",
            delta_color="inverse"
        )
    with col3:
        st.metric(
            label="Silhouette Score",
            value="0.2449",
            delta="K=2 optimal"
        )
    with col4:
        st.metric(
            label="Optimal Clusters (K)",
            value="2",
            delta="from K=2-8 tested"
        )

    st.markdown("---")

    # ── ROW 1: PIE + MORTALITY ────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Distribution")
        fig_pie = go.Figure(data=[go.Pie(
            labels=["HIGH RISK (Cluster 1)",
                    "LOW RISK (Cluster 2)"],
            values=[34, 35],
            hole=0.4,
            marker_colors=["#E74C3C", "#2ECC71"],
            textinfo="percent+label",
            textfont_size=13,
        )])
        fig_pie.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=320
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("In-Hospital Mortality Rate")
        fig_mort = go.Figure(data=[go.Bar(
            x=["HIGH RISK", "LOW RISK"],
            y=[50.0, 8.6],
            marker_color=["#E74C3C", "#2ECC71"],
            text=["50.0%", "8.6%"],
            textposition="outside",
            textfont_size=16,
            width=0.4
        )])
        fig_mort.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            yaxis_title="Mortality Rate (%)",
            yaxis_range=[0, 65],
            margin=dict(t=20, b=20, l=20, r=20),
            height=320
        )
        st.plotly_chart(fig_mort, use_container_width=True)

    # ── ROW 2: VITALS COMPARISON ──────────────────────────
    st.subheader("Vital Signs Comparison")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig_hr = go.Figure()
        fig_hr.add_trace(go.Bar(
            x=["HIGH RISK", "LOW RISK"],
            y=[81.8, 89.2],
            marker_color=["#E74C3C", "#2ECC71"],
            text=["81.8 bpm", "89.2 bpm"],
            textposition="outside",
            width=0.4
        ))
        fig_hr.add_hline(
            y=100, line_dash="dash",
            line_color="orange",
            annotation_text="Tachycardia (100 bpm)"
        )
        fig_hr.update_layout(
            title="Avg Heart Rate",
            paper_bgcolor="white",
            plot_bgcolor="white",
            yaxis_title="bpm",
            yaxis_range=[0, 120],
            margin=dict(t=40, b=20, l=20, r=20),
            height=300
        )
        st.plotly_chart(fig_hr, use_container_width=True)

    with col2:
        fig_bp = go.Figure()
        fig_bp.add_trace(go.Bar(
            x=["HIGH RISK", "LOW RISK"],
            y=[108.1, 131.4],
            marker_color=["#E74C3C", "#2ECC71"],
            text=["108.1 mmHg", "131.4 mmHg"],
            textposition="outside",
            width=0.4
        ))
        fig_bp.add_hline(
            y=140, line_dash="dash",
            line_color="orange",
            annotation_text="Hypertension (140)"
        )
        fig_bp.add_hline(
            y=90, line_dash="dash",
            line_color="red",
            annotation_text="Hypotension (90)"
        )
        fig_bp.update_layout(
            title="Avg Systolic Blood Pressure",
            paper_bgcolor="white",
            plot_bgcolor="white",
            yaxis_title="mmHg",
            yaxis_range=[0, 170],
            margin=dict(t=40, b=20, l=20, r=20),
            height=300
        )
        st.plotly_chart(fig_bp, use_container_width=True)

    with col3:
        fig_los = go.Figure()
        fig_los.add_trace(go.Bar(
            x=["HIGH RISK", "LOW RISK"],
            y=[5.25, 3.44],
            marker_color=["#E74C3C", "#2ECC71"],
            text=["5.25 days", "3.44 days"],
            textposition="outside",
            width=0.4
        ))
        fig_los.update_layout(
            title="Avg ICU Length of Stay",
            paper_bgcolor="white",
            plot_bgcolor="white",
            yaxis_title="Days",
            yaxis_range=[0, 8],
            margin=dict(t=40, b=20, l=20, r=20),
            height=300
        )
        st.plotly_chart(fig_los, use_container_width=True)

    # ── ROW 3: HEATMAP ────────────────────────────────────
    st.subheader("Risk Profile Heatmap")

    heatmap_data = pd.DataFrame({
        "Age (yrs)":        [72.9, 67.2],
        "ICU Stay (days)":  [5.25, 3.44],
        "Heart Rate (bpm)": [81.8, 89.2],
        "Systolic BP":      [108.1, 131.4],
        "Diastolic BP":     [54.8, 69.7],
        "Resp Rate":        [19.1, 19.8],
        "SpO2 (%)":         [96.8, 96.7],
        "Mortality (%)":    [50.0, 8.6],
    }, index=["HIGH RISK", "LOW RISK"])

    fig_heatmap = px.imshow(
        heatmap_data,
        text_auto=".1f",
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
    )
    fig_heatmap.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=20, b=20, l=20, r=20),
        height=220,
        coloraxis_showscale=True,
    )
    fig_heatmap.update_traces(textfont_size=14)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # ── TOP DIAGNOSES ─────────────────────────────────────
    st.markdown("---")
    st.subheader("Top Diagnoses by Cluster")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔴 High Risk Cluster")
        diagnoses_high = pd.DataFrame({
            "Diagnosis": [
                "Sepsis", "Hypotension",
                "Upper GI Bleed",
                "Metastatic Melanoma/Anaemia",
                "Mediastinal Adenopathy"
            ],
            "Count": [3, 2, 2, 2, 2]
        })
        fig_dh = px.bar(
            diagnoses_high,
            x="Count", y="Diagnosis",
            orientation="h",
            color_discrete_sequence=["#E74C3C"]
        )
        fig_dh.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(t=10, b=10, l=10, r=10),
            height=250,
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_dh, use_container_width=True)

    with col2:
        st.markdown("#### 🟢 Low Risk Cluster")
        diagnoses_low = pd.DataFrame({
            "Diagnosis": [
                "Pneumonia", "Shortness of Breath",
                "Asthma/COPD",
                "Fever/UTI",
                "Cerebrovascular Accident"
            ],
            "Count": [4, 3, 2, 1, 1]
        })
        fig_dl = px.bar(
            diagnoses_low,
            x="Count", y="Diagnosis",
            orientation="h",
            color_discrete_sequence=["#2ECC71"]
        )
        fig_dl.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(t=10, b=10, l=10, r=10),
            height=250,
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_dl, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 2: PATIENT RISK PREDICTOR
# ══════════════════════════════════════════════════════════
elif page == "🔍 Patient Risk Predictor":

    st.title("🔍 Patient Risk Predictor")
    st.markdown(
        "Enter a patient's vital signs below to predict "
        "their ICU risk cluster using the trained "
        "Bisecting K-Means model."
    )
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Patient Vital Signs")
        st.markdown("*Adjust the sliders to match the patient's readings:*")
        st.markdown(" ")

        age = st.slider(
            "Age (years)",
            min_value=18, max_value=100,
            value=70, step=1
        )
        los = st.slider(
            "ICU Length of Stay (days)",
            min_value=0.1, max_value=40.0,
            value=3.0, step=0.1
        )
        heart_rate = st.slider(
            "Heart Rate (bpm)",
            min_value=30, max_value=200,
            value=85, step=1
        )
        systolic_bp = st.slider(
            "Systolic Blood Pressure (mmHg)",
            min_value=60, max_value=220,
            value=120, step=1
        )
        diastolic_bp = st.slider(
            "Diastolic Blood Pressure (mmHg)",
            min_value=30, max_value=140,
            value=70, step=1
        )
        respiratory_rate = st.slider(
            "Respiratory Rate (breaths/min)",
            min_value=5, max_value=50,
            value=18, step=1
        )
        spo2 = st.slider(
            "SpO2 — Oxygen Saturation (%)",
            min_value=70, max_value=100,
            value=97, step=1
        )

        predict_btn = st.button(
            "🔍 Predict Risk Cluster",
            use_container_width=True,
            type="primary"
        )

    with col2:
        st.subheader("Prediction Result")

        if predict_btn:
            result, dist_high, dist_low = predict_cluster(
                age, los, heart_rate, systolic_bp,
                diastolic_bp, respiratory_rate, spo2
            )

            # Confidence score
            total_dist = dist_high + dist_low
            confidence = (
                (dist_low / total_dist) * 100
                if result == "HIGH RISK"
                else (dist_high / total_dist) * 100
            )

            if result == "HIGH RISK":
                st.markdown(f"""
                <div class="result-box result-high">
                    <h1 style="color:#E74C3C; margin:0;">
                        🔴 HIGH RISK
                    </h1>
                    <p style="font-size:1.1em; color:#333; margin:8px 0;">
                        This patient's profile is closest to the
                        <strong>High Risk Cluster</strong>
                    </p>
                    <p style="font-size:0.95em; color:#666;">
                        Cluster 1 | Mortality Rate: <strong>50.0%</strong>
                    </p>
                    <p style="font-size:0.95em; color:#666;">
                        Model confidence: <strong>{confidence:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### ⚠️ Clinical Indicators")
                st.error(
                    "This patient's vital signs resemble the "
                    "high-mortality cluster characterised by "
                    "**low blood pressure**, **older age**, and "
                    "**longer ICU stays** — consistent with "
                    "**septic shock** or haemodynamic instability."
                )

            else:
                st.markdown(f"""
                <div class="result-box result-low">
                    <h1 style="color:#2ECC71; margin:0;">
                        🟢 LOW RISK
                    </h1>
                    <p style="font-size:1.1em; color:#333; margin:8px 0;">
                        This patient's profile is closest to the
                        <strong>Low Risk Cluster</strong>
                    </p>
                    <p style="font-size:0.95em; color:#666;">
                        Cluster 2 | Mortality Rate: <strong>8.6%</strong>
                    </p>
                    <p style="font-size:0.95em; color:#666;">
                        Model confidence: <strong>{confidence:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### ✅ Clinical Indicators")
                st.success(
                    "This patient's vital signs resemble the "
                    "low-mortality cluster characterised by "
                    "**stable blood pressure**, **shorter ICU stays**, "
                    "and diagnoses such as **pneumonia** or "
                    "**respiratory conditions**."
                )

            # Patient vs cluster comparison radar chart
            st.markdown("#### Patient vs Cluster Profiles")

            categories = [
                "Age", "Heart Rate", "Systolic BP",
                "Diastolic BP", "Resp Rate", "SpO2"
            ]

            # Normalise to 0-100 scale for radar
            def norm(val, min_v, max_v):
                return (val - min_v) / (max_v - min_v) * 100

            ranges = {
                "Age": (18, 100),
                "Heart Rate": (30, 200),
                "Systolic BP": (60, 220),
                "Diastolic BP": (30, 140),
                "Resp Rate": (5, 50),
                "SpO2": (70, 100)
            }

            patient_vals = [
                norm(age, *ranges["Age"]),
                norm(heart_rate, *ranges["Heart Rate"]),
                norm(systolic_bp, *ranges["Systolic BP"]),
                norm(diastolic_bp, *ranges["Diastolic BP"]),
                norm(respiratory_rate, *ranges["Resp Rate"]),
                norm(spo2, *ranges["SpO2"])
            ]
            high_vals = [
                norm(72.9, *ranges["Age"]),
                norm(81.8, *ranges["Heart Rate"]),
                norm(108.1, *ranges["Systolic BP"]),
                norm(54.8, *ranges["Diastolic BP"]),
                norm(19.1, *ranges["Resp Rate"]),
                norm(96.8, *ranges["SpO2"])
            ]
            low_vals = [
                norm(67.2, *ranges["Age"]),
                norm(89.2, *ranges["Heart Rate"]),
                norm(131.4, *ranges["Systolic BP"]),
                norm(69.7, *ranges["Diastolic BP"]),
                norm(19.8, *ranges["Resp Rate"]),
                norm(96.7, *ranges["SpO2"])
            ]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=patient_vals + [patient_vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="This Patient",
                line_color="#1B3A6B",
                fillcolor="rgba(27,58,107,0.15)"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=high_vals + [high_vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="High Risk Centre",
                line_color="#E74C3C",
                fillcolor="rgba(231,76,60,0.10)"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=low_vals + [low_vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="Low Risk Centre",
                line_color="#2ECC71",
                fillcolor="rgba(46,204,113,0.10)"
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=False, range=[0, 100])
                ),
                paper_bgcolor="white",
                showlegend=True,
                margin=dict(t=20, b=20, l=20, r=20),
                height=380
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        else:
            st.info(
                "👈 Adjust the patient's vital signs on the left "
                "and click **Predict Risk Cluster** to see the result."
            )
            st.markdown(" ")
            st.markdown("#### How the prediction works")
            st.markdown("""
            1. Your inputs are **normalised** using the same
               StandardScaler parameters from the Spark pipeline
            2. The **Euclidean distance** is calculated from the
               patient's feature vector to each cluster centre
            3. The patient is assigned to the **nearest cluster**
            4. A **radar chart** shows how the patient compares
               to both cluster profiles
            """)

# ══════════════════════════════════════════════════════════
# PAGE 3: DATA EXPLORER
# ══════════════════════════════════════════════════════════
elif page == "📋 Data Explorer":

    st.title("📋 Cluster Data Explorer")
    st.markdown(
        "Explore the full cluster profile data from our "
        "Bisecting K-Means model."
    )
    st.markdown("---")

    # Full data table
    st.subheader("Full Cluster Comparison Table")
    st.dataframe(
        cluster_data.set_index("Cluster"),
        use_container_width=True,
        height=120
    )

    st.markdown("---")

    # Silhouette scores
    st.subheader("Silhouette Analysis — Finding Optimal K")
    silhouette_data = pd.DataFrame({
        "K": [2, 3, 4, 5, 6, 7, 8],
        "Silhouette Score": [
            0.2449, 0.1753, 0.1958,
            0.1666, 0.1630, 0.1398, 0.1426
        ]
    })

    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(
        x=silhouette_data["K"],
        y=silhouette_data["Silhouette Score"],
        mode="lines+markers",
        line=dict(color="#1B3A6B", width=2),
        marker=dict(size=10, color="#1B3A6B"),
        name="Silhouette Score"
    ))
    fig_sil.add_trace(go.Scatter(
        x=[2], y=[0.2449],
        mode="markers",
        marker=dict(size=16, color="#E74C3C", symbol="star"),
        name="Best K (K=2)"
    ))
    fig_sil.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Silhouette Score",
        xaxis=dict(tickmode="linear", tick0=2, dtick=1),
        height=350,
        margin=dict(t=20, b=40, l=40, r=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("---")

    # Feature importance
    st.subheader("Feature Differences Between Clusters")
    features = [
        "Age", "ICU Stay", "Heart Rate",
        "Systolic BP", "Diastolic BP",
        "Resp Rate", "SpO2", "Mortality"
    ]
    high_vals = [72.9, 5.25, 81.8, 108.1, 54.8, 19.1, 96.8, 50.0]
    low_vals  = [67.2, 3.44, 89.2, 131.4, 69.7, 19.8, 96.7, 8.6]

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(
        name="HIGH RISK",
        x=features,
        y=high_vals,
        marker_color="#E74C3C"
    ))
    fig_compare.add_trace(go.Bar(
        name="LOW RISK",
        x=features,
        y=low_vals,
        marker_color="#2ECC71"
    ))
    fig_compare.update_layout(
        barmode="group",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=380,
        margin=dict(t=20, b=40, l=40, r=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    st.markdown("---")

    # Pipeline summary
    st.subheader("Pipeline Summary")
    pipeline_data = pd.DataFrame({
        "Script": [
            "01_explore.py", "02_joining.py", "03_vitals.py",
            "04_features.py", "05_tfidf.py", "06_clustering.py",
            "07_silhouette.py", "08_final_model.py", "09_dashboard.py"
        ],
        "Stage": [
            "Exploration", "Joining", "Vitals Processing",
            "Feature Engineering", "NLP Pipeline", "Clustering",
            "Evaluation", "Final Model", "Visualisation"
        ],
        "Key Output": [
            "Data understanding", "master_patients/ (Parquet)",
            "master_with_vitals/ (partitioned)", "df_scaled/ + scaler_model",
            "tfidf_pipeline/", "predictions_k4/",
            "silhouette_results.json", "bkm_final_k2/ + profiles",
            "risk_dashboard.png"
        ]
    })
    st.dataframe(
        pipeline_data.set_index("Script"),
        use_container_width=True,
        height=360
    )