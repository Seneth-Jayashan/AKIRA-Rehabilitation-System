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
    page_title="AKIRA AI | Advanced Neural Evaluation",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =================================================================
# PREMIUM AI THEME
# High-end glassmorphism, glowing gradients, futuristic typography.
# =================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* Base Styles */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Dynamic Animated Background */
.stApp {
    background-color: #030303;
    background-image: 
        radial-gradient(circle at 15% 50%, rgba(112, 0, 255, 0.08) 0%, transparent 50%),
        radial-gradient(circle at 85% 30%, rgba(0, 240, 255, 0.08) 0%, transparent 50%);
    background-attachment: fixed;
    color: #FFFFFF;
}

/* Hide Default Streamlit Elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] { background: transparent !important; }

/* Sidebar styling (Glassmorphism) */
section[data-testid="stSidebar"] {
    background: rgba(10, 10, 12, 0.6) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
section[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

/* AKIRA AI LOGO IN HERO */
.hero-container {
    padding: 3rem 2rem;
    border-radius: 24px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
}

.hero-container::before {
    content: "";
    position: absolute;
    top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle at center, rgba(0, 240, 255, 0.1) 0%, transparent 40%);
    animation: rotate 20s linear infinite;
    pointer-events: none;
}

@keyframes rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.akira-logo {
    font-size: 5rem;
    font-weight: 800;
    letter-spacing: -2px;
    margin: 0;
    background: linear-gradient(135deg, #00F0FF 0%, #7000FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0px 0px 10px rgba(0, 240, 255, 0.3));
}

.akira-tagline {
    font-size: 1.2rem;
    font-weight: 300;
    color: #94A3B8;
    margin-top: -10px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Glass Panels */
.glass-panel {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.3s ease;
}
.glass-panel:hover {
    border: 1px solid rgba(0, 240, 255, 0.3);
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.1);
    transform: translateY(-2px);
}

.glass-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #00F0FF;
    margin-bottom: 0.5rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.glass-text {
    font-size: 0.95rem;
    color: #CBD5E1;
    line-height: 1.6;
}

/* Readouts (Metrics) */
.metric-box {
    background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0) 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-box::after {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00F0FF, transparent);
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    color: #FFFFFF;
    text-shadow: 0 0 10px rgba(255,255,255,0.2);
}
.metric-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 0.5rem;
}

/* Button Customization */
.stButton > button {
    background: linear-gradient(90deg, #00F0FF 0%, #7000FF 100%) !important;
    color: white !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    width: 100%;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(112, 0, 255, 0.4) !important;
}
.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 6px 25px rgba(0, 240, 255, 0.5) !important;
}

/* Select Box */
div[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: white !important;
}

hr {
    border-color: rgba(255,255,255,0.05) !important;
}

/* Dataframe Styling */
.stDataFrame {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.05);
}

.section-header {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    background: linear-gradient(90deg, #FFFFFF 0%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# =================================================================
# HERO SECTION
# =================================================================
st.markdown(f"""
<div class="hero-container">
    <h1 class="akira-logo">AKIRA AI</h1>
    <p class="akira-tagline">Advanced Neural Network Evaluation Console</p>
</div>
""", unsafe_allow_html=True)


# =================================================================
# MODEL CONFIGURATION
# =================================================================
MODELS = {
    "Exercise Recognition": {
        "path": "models/exercise_recognition_model.keras",
        "type": "multiclass",
        "tag": "MULTICLASS · IMU · 4 CH",
        "description": "Classifies the exercise being performed — Squat, Leg Extension, Walking, or Other Rehab.",
    },
    "Binary Compensation Detection": {
        "path": "models/compensation_detection_model.keras",
        "type": "binary",
        "tag": "BINARY · PASS/FAIL",
        "description": "Detects whether the patient's exercise form is Correct or Incorrect.",
    },
    "Multi-Class Compensation Diagnosis": {
        "path": "models/multiclass_compensation_model.keras",
        "type": "multiclass",
        "tag": "MULTICLASS · DIAGNOSTIC",
        "description": "Diagnoses the specific postural compensation present in a movement.",
    },
}

with st.sidebar:
    st.markdown('<h2 style="color: #00F0FF; font-weight: 800; text-align: center; margin-bottom: 2rem;">◈ AKIRA AI</h2>', unsafe_allow_html=True)
    
    st.markdown("<p style='color:#94A3B8; font-weight: 600; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;'>Select Neural Network</p>", unsafe_allow_html=True)
    model_selection = st.selectbox(
        "Select Model to Evaluate", list(MODELS.keys()), label_visibility="collapsed"
    )
    config = MODELS[model_selection]
    
    st.markdown(f"""
    <div class="glass-panel" style="margin-top: 2rem;">
        <div class="glass-title">Active Model</div>
        <div class="glass-text" style="color:#00F0FF; font-weight: 600; font-size: 0.8rem; margin-bottom: 0.5rem;">{config['tag']}</div>
        <div class="glass-text">{config['description']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style="margin-top: 2rem; padding: 1rem; border-radius: 8px; background: rgba(0, 240, 255, 0.05); border-left: 3px solid #00F0FF;">
            <p style="color:#CBD5E1; font-size:0.8rem; margin:0;">
                <b>Strict Evaluation:</b> Test split utilizes a rigid subject-aware isolation approach. 
                Test patients have never been exposed to the model during training.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Initializing data pipelines and reconstructing held-out tensors…")
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
run = st.button(f"INITIALIZE INFERENCE  //  {model_selection.upper()}")

if run:
    model_path = config['path']
    if not os.path.exists(model_path):
        st.error(f"Neural network weights not found at `{model_path}`. Train the model before evaluation.")
    else:
        with st.spinner("Reconstructing test set tensors…"):
            try:
                X_imu_test, X_age_test, y_test, class_names, test_patient_count = get_test_data(model_selection)
            except Exception as e:
                st.error(f"Pipeline Exception: {e}")
                st.stop()

        with st.spinner("Executing neural inference…"):
            model = tf.keras.models.load_model(model_path)
            y_pred_prob = model.predict({'imu_input': X_imu_test, 'age_input': X_age_test})

            if config["type"] == "binary":
                y_pred_classes = (y_pred_prob > 0.5).astype(int).flatten()
            else:
                y_pred_classes = np.argmax(y_pred_prob, axis=1)

            acc = accuracy_score(y_test, y_pred_classes)

        # ---------------- Readouts ----------------
        st.markdown('<div class="section-header">Performance Metrics</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{acc * 100:.2f}<span style="font-size:1.5rem; color:#00F0FF;">%</span></div>
                <div class="metric-label">Evaluation Accuracy</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{test_patient_count}</div>
                <div class="metric-label">Unseen Patients</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{len(y_test):,}</div>
                <div class="metric-label">Analyzed Windows</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        vcol1, vcol2 = st.columns([1, 1])

        # ---------------- Confusion matrix ----------------
        with vcol1:
            st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
            
            # Custom Cyber Neon Colormap
            akira_cmap = LinearSegmentedColormap.from_list(
                "akira_neon", ["#030303", "#120a30", "#3700b3", "#7000FF", "#00F0FF"]
            )
            
            fig, ax = plt.subplots(figsize=(7, 5.4))
            fig.patch.set_facecolor('#030303')
            ax.set_facecolor('#030303')
            
            cm = confusion_matrix(y_test, y_pred_classes)
            sns.heatmap(
                cm, annot=True, fmt='d', cmap=akira_cmap, cbar=False,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"color": "#FFFFFF", "fontsize": 12, "fontfamily": "monospace", "fontweight": "bold"},
                linewidths=2, linecolor='#030303', ax=ax,
            )
            ax.set_ylabel('Actual Label', color='#94A3B8', fontsize=11, fontweight='bold')
            ax.set_xlabel('Predicted Label', color='#94A3B8', fontsize=11, fontweight='bold')
            ax.tick_params(colors='#CBD5E1', labelsize=10)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.tight_layout()
            
            st.pyplot(fig)

        # ---------------- Classification report ----------------
        with vcol2:
            st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
            report = classification_report(
                y_test, y_pred_classes, target_names=class_names,
                output_dict=True, zero_division=0
            )
            df_report = pd.DataFrame(report).transpose()
            
            report_cmap = LinearSegmentedColormap.from_list(
                "akira_report", ["#030303", "#120a30", "#3700b3", "#7000FF", "#00F0FF"]
            )
            
            styled = (
                df_report.style
                .format("{:.3f}")
                .background_gradient(cmap=report_cmap, subset=['f1-score'])
                .set_properties(**{
                    'font-family': 'JetBrains Mono, monospace',
                    'font-size': '13px',
                    'color': '#FFFFFF',
                    'background-color': '#0A0A0C',
                    'border-color': 'rgba(255,255,255,0.05)'
                })
            )
            st.dataframe(styled, width='stretch')