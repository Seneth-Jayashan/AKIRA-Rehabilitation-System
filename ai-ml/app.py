import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf

# Add src to path to import data loaders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

st.set_page_config(
    page_title="AKIRA AI — Model Evaluation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =================================================================
# THEME — clinical telemetry console, built around the IMU signal
# that AKIRA actually reads: ink background, signal-teal for
# healthy/correct readings, coral for flagged/incorrect ones,
# monospace for anything that is literally a number off a sensor.
# =================================================================
INK = "#080B12"
PANEL = "#0E131E"
LINE = "rgba(255,255,255,0.08)"
TEAL = "#4CE0D2"
TEAL_DIM = "rgba(76,224,210,0.35)"
CORAL = "#FF7A59"
VIOLET = "#8B7CF6"
TEXT = "#E9ECF5"
MUTED = "#8B92B0"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}
h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif !important; }}
code, .mono {{ font-family: 'JetBrains Mono', monospace !important; }}

.stApp {{
    background:
        radial-gradient(circle at 12% 0%, rgba(76,224,210,0.06) 0%, transparent 45%),
        radial-gradient(circle at 90% 15%, rgba(139,124,246,0.05) 0%, transparent 40%),
        {INK};
}}
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header[data-testid="stHeader"] {{ background: transparent; }}

section[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {LINE};
}}
section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
section[data-testid="stSidebar"] label {{ color: {MUTED} !important; }}

/* ---------- Hero ---------- */
.akira-eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 3px;
    color: {TEAL};
    text-transform: uppercase;
    margin: 0 0 10px 0;
}}
.akira-title {{
    font-size: 46px;
    font-weight: 700;
    color: {TEXT};
    margin: 0;
    letter-spacing: -0.5px;
    line-height: 1.05;
}}
.akira-title span {{ color: {TEAL}; }}
.akira-sub {{
    color: {MUTED};
    font-size: 15px;
    margin-top: 10px;
    max-width: 560px;
}}
.akira-hero {{
    background: {PANEL};
    border: 1px solid {LINE};
    border-radius: 10px;
    padding: 30px 34px 0 34px;
    margin-bottom: 26px;
    overflow: hidden;
}}
.waveform-wrap {{ margin: 22px -4px 0 -4px; opacity: 0.9; }}
.waveform-wrap svg {{ display: block; width: 100%; height: 46px; }}
.wf-trace {{
    fill: none;
    stroke: {TEAL_DIM};
    stroke-width: 1.4;
}}
.wf-pulse {{
    fill: none;
    stroke: {TEAL};
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-dasharray: 90 900;
    animation: wf-scan 5.5s linear infinite;
    filter: drop-shadow(0 0 4px rgba(76,224,210,0.6));
}}
@keyframes wf-scan {{
    0%   {{ stroke-dashoffset: 0; }}
    100% {{ stroke-dashoffset: -990; }}
}}

/* ---------- Cards / panels ---------- */
.akira-panel {{
    background: {PANEL};
    border: 1px solid {LINE};
    border-radius: 10px;
    padding: 18px 20px;
}}
.model-desc {{
    font-size: 13px;
    color: {MUTED};
    line-height: 1.5;
    margin-top: 6px;
}}
.model-tag {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 1px;
    color: {TEAL};
    border: 1px solid {TEAL_DIM};
    border-radius: 4px;
    padding: 3px 8px;
    margin-top: 12px;
}}

/* ---------- Metric readouts ---------- */
.readout {{
    background: {PANEL};
    border: 1px solid {LINE};
    border-top: 2px solid {TEAL};
    border-radius: 8px;
    padding: 16px 18px;
    text-align: left;
}}
.readout-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 30px;
    font-weight: 600;
    color: {TEXT};
    line-height: 1;
}}
.readout-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: {MUTED};
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 8px;
}}

/* ---------- Buttons ---------- */
.stButton > button {{
    background: {TEAL};
    color: {INK};
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.5px;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    transition: box-shadow 0.15s ease;
}}
.stButton > button:hover {{
    box-shadow: 0 0 0 3px rgba(76,224,210,0.25);
    color: {INK};
}}

/* ---------- Misc form elements ---------- */
div[data-baseweb="select"] > div {{
    background-color: {PANEL} !important;
    border-color: {LINE} !important;
    color: {TEXT} !important;
}}
.stDataFrame {{ border-radius: 8px; overflow: hidden; border: 1px solid {LINE}; }}
hr {{ border-color: {LINE} !important; }}

.section-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 4px;
}}
</style>
""", unsafe_allow_html=True)


def waveform_svg():
    """A small animated IMU-style trace — the literal signal AKIRA reads,
    used as the page's one signature visual instead of a generic gradient."""
    pts = "M0,23 L20,23 L28,8 L36,38 L44,14 L52,30 L60,23 L80,23 " \
          "L100,23 L108,10 L116,36 L124,16 L132,28 L140,23 L160,23 " \
          "L180,23 L188,8 L196,38 L204,14 L212,30 L220,23 L240,23 " \
          "L260,23 L268,10 L276,36 L284,16 L292,28 L300,23 L340,23"
    return f"""
    <div class="waveform-wrap">
        <svg viewBox="0 0 340 46" preserveAspectRatio="none">
            <path class="wf-trace" d="{pts}"></path>
            <path class="wf-pulse" d="{pts}"></path>
        </svg>
    </div>
    """


# =================================================================
# HERO
# =================================================================
st.markdown(f"""
<div class="akira-hero">
    <p class="akira-eyebrow">◈ Rehabilitation Motion Intelligence</p>
    <p class="akira-title">AKIRA <span>AI</span></p>
    <p class="akira-sub">Model evaluation console. Runs each trained network against a
    patient-held-out test split — subjects the model has never seen during training —
    to report honest, generalizable accuracy.</p>
    {waveform_svg()}
</div>
""", unsafe_allow_html=True)

# =================================================================
# MODEL CONFIGURATION
# =================================================================
MODELS = {
    "Exercise Recognition": {
        "path": "models/exercise_recognition_model.keras",
        "type": "multiclass",
        "tag": "MULTICLASS · 4 CH",
        "description": "Classifies the exercise being performed — Squat, Leg Extension, Walking, or Other Rehab.",
    },
    "Binary Compensation Detection": {
        "path": "models/compensation_detection_model.keras",
        "type": "binary",
        "tag": "BINARY · PASS/FAIL",
        "description": "Detects whether exercise form is Correct or Incorrect.",
    },
    "Multi-Class Compensation Diagnosis": {
        "path": "models/multiclass_compensation_model.keras",
        "type": "multiclass",
        "tag": "MULTICLASS · DIAGNOSTIC",
        "description": "Diagnoses the specific postural compensation present in a movement.",
    },
}

with st.sidebar:
    st.markdown('<p class="akira-eyebrow">◈ AKIRA AI</p>', unsafe_allow_html=True)
    st.markdown("#### Select Model")
    model_selection = st.selectbox(
        "Select Model to Evaluate", list(MODELS.keys()), label_visibility="collapsed"
    )
    config = MODELS[model_selection]
    st.markdown(f"""
    <div class="akira-panel" style="margin-top:14px;">
        <div class="section-label">Objective</div>
        <p class="model-desc">{config['description']}</p>
        <span class="model-tag">{config['tag']}</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8B92B0; font-size:12px;">Evaluation uses the same '
        'subject-aware split and random seed as training, so no test-set patient '
        'ever appears in the training data.</p>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Reconstructing patient-held-out test set…")
def get_test_data(model_name):
    if model_name == "Exercise Recognition":
        from train_model import load_data
        X_imu, X_age, y, subjects = load_data()
    elif model_name == "Binary Compensation Detection":
        from train_compensation_model import load_data
        X_imu, X_age, y, subjects = load_data()
    elif model_name == "Multi-Class Compensation Diagnosis":
        from train_multiclass_compensation_model import load_data
        X_imu, X_age, y, subjects = load_data()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    unique_subjects = np.unique(subjects)
    train_subs, test_subs = train_test_split(unique_subjects, test_size=0.2, random_state=42)
    train_subs, val_subs = train_test_split(train_subs, test_size=0.2, random_state=42)

    train_idx = np.isin(subjects, train_subs)
    test_idx = np.isin(subjects, test_subs)

    X_age_train = X_age[train_idx]
    X_age_test = X_age[test_idx]

    valid_ages = X_age_train[X_age_train > 0]
    mean_age = np.mean(valid_ages) if len(valid_ages) > 0 else 40

    X_age_train = np.where(X_age_train == -1, mean_age, X_age_train)
    X_age_test = np.where(X_age_test == -1, mean_age, X_age_test)

    age_scaler = StandardScaler()
    age_scaler.fit(X_age_train.reshape(-1, 1))

    X_imu_test = X_imu[test_idx]
    X_age_test_scaled = age_scaler.transform(X_age_test.reshape(-1, 1))
    y_test = y_encoded[test_idx]

    return X_imu_test, X_age_test_scaled, y_test, le.classes_, len(test_subs)


# =================================================================
# RUN
# =================================================================
run = st.button(f"▸ Run Evaluation — {model_selection}")

if run:
    model_path = config['path']
    if not os.path.exists(model_path):
        st.error(f"Model file not found at `{model_path}`. Please train the model first.")
    else:
        with st.spinner("Reconstructing test set…"):
            try:
                X_imu_test, X_age_test, y_test, class_names, test_patient_count = get_test_data(model_selection)
            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.stop()

        with st.spinner("Running inference…"):
            model = tf.keras.models.load_model(model_path)
            y_pred_prob = model.predict({'imu_input': X_imu_test, 'age_input': X_age_test})

            if config["type"] == "binary":
                y_pred_classes = (y_pred_prob > 0.5).astype(int).flatten()
            else:
                y_pred_classes = np.argmax(y_pred_prob, axis=1)

            acc = accuracy_score(y_test, y_pred_classes)

        # ---------------- Readouts ----------------
        st.markdown('<p class="section-label" style="margin-top:6px;">Results</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="readout">
                <div class="readout-value">{acc * 100:.2f}%</div>
                <div class="readout-label">Test Accuracy</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="readout">
                <div class="readout-value">{test_patient_count}</div>
                <div class="readout-label">Unseen Test Patients</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="readout">
                <div class="readout-value">{len(y_test)}</div>
                <div class="readout-label">Total Test Windows</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        vcol1, vcol2 = st.columns([1, 1])

        # ---------------- Confusion matrix ----------------
        with vcol1:
            st.markdown('<p class="section-label">Confusion Matrix</p>', unsafe_allow_html=True)
            akira_cmap = LinearSegmentedColormap.from_list(
                "akira_signal", ["#0E131E", "#123B3A", "#1F6F68", TEAL]
            )
            fig, ax = plt.subplots(figsize=(7, 5.4))
            fig.patch.set_facecolor(INK)
            ax.set_facecolor(INK)
            cm = confusion_matrix(y_test, y_pred_classes)
            sns.heatmap(
                cm, annot=True, fmt='d', cmap=akira_cmap, cbar=False,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"color": "#E9ECF5", "fontsize": 11, "fontfamily": "monospace"},
                linewidths=1, linecolor=INK, ax=ax,
            )
            ax.set_ylabel('Actual', color=MUTED, fontsize=11)
            ax.set_xlabel('Predicted', color=MUTED, fontsize=11)
            ax.tick_params(colors=MUTED, labelsize=9)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)

        # ---------------- Classification report ----------------
        with vcol2:
            st.markdown('<p class="section-label">Classification Report</p>', unsafe_allow_html=True)
            report = classification_report(
                y_test, y_pred_classes, target_names=class_names,
                output_dict=True, zero_division=0
            )
            df_report = pd.DataFrame(report).transpose()
            report_cmap = LinearSegmentedColormap.from_list(
                "akira_report", ["#0E131E", "#123B3A", "#1F6F68", TEAL]
            )
            styled = (
                df_report.style
                .format("{:.3f}")
                .background_gradient(cmap=report_cmap, subset=['f1-score'])
                .set_properties(**{
                    'font-family': 'JetBrains Mono, monospace',
                    'font-size': '12.5px',
                })
            )
            st.dataframe(styled, use_container_width=True)