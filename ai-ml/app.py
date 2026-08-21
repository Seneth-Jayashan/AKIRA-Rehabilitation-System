import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf

# Add src to path to import data loaders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))



st.set_page_config(page_title="AKIRA Model Evaluation", layout="wide")

st.title("🤖 AKIRA: Model Evaluation Dashboard")
st.markdown("Evaluate the accuracy of the trained AI models on Unseen Patients (Test Set).")

# Model Configuration Mapping
MODELS = {
    "Exercise Recognition": {
        "path": "models/exercise_recognition_model.keras",
        "type": "multiclass",
        "description": "Classifies the exercise being performed (Squat, Leg Extension, Walking, Other Rehab)."
    },
    "Binary Compensation Detection": {
        "path": "models/compensation_detection_model.keras",
        "type": "binary",
        "description": "Detects if an exercise form is 'Correct' or 'Incorrect'."
    },
    "Multi-Class Compensation Diagnosis": {
        "path": "models/multiclass_compensation_model.keras",
        "type": "multiclass",
        "description": "Diagnoses specific postural compensations."
    }
}

model_selection = st.sidebar.selectbox("Select Model to Evaluate", list(MODELS.keys()))
config = MODELS[model_selection]

st.sidebar.markdown(f"**Goal:** {config['description']}")

@st.cache_data(show_spinner="Loading and Processing Data... (This takes a minute)")
def get_test_data(model_name):
    # Depending on model_name, we use the specific loader
    if model_name == "Exercise Recognition":
        from train_model import load_data
        X_imu, X_age, y, subjects = load_data()
    elif model_name == "Binary Compensation Detection":
        from train_compensation_model import load_data
        X_imu, X_age, y, subjects = load_data()
    elif model_name == "Multi-Class Compensation Diagnosis":
        from train_multiclass_compensation_model import load_data
        X_imu, X_age, y, subjects = load_data()
    
    # 2. Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 3. Subject-Aware Train/Test Split (Same random_state as training scripts)
    unique_subjects = np.unique(subjects)
    train_subs, test_subs = train_test_split(unique_subjects, test_size=0.2, random_state=42)
    train_subs, val_subs = train_test_split(train_subs, test_size=0.2, random_state=42)
    
    train_idx = np.isin(subjects, train_subs)
    test_idx = np.isin(subjects, test_subs)
    
    X_age_train = X_age[train_idx]
    X_age_test = X_age[test_idx]
    
    # Standardize Age exactly as in training
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


if st.button(f"Evaluate {model_selection}"):
    model_path = config['path']
    if not os.path.exists(model_path):
        st.error(f"Model file not found at `{model_path}`. Please train the model first.")
    else:
        with st.spinner("Reconstructing Test Set..."):
            try:
                X_imu_test, X_age_test, y_test, class_names, test_patient_count = get_test_data(model_selection)
            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.stop()
                
        with st.spinner("Running Inference..."):
            model = tf.keras.models.load_model(model_path)
            
            y_pred_prob = model.predict({'imu_input': X_imu_test, 'age_input': X_age_test})
            
            if config["type"] == "binary":
                y_pred_classes = (y_pred_prob > 0.5).astype(int).flatten()
            else:
                y_pred_classes = np.argmax(y_pred_prob, axis=1)
                
            acc = accuracy_score(y_test, y_pred_classes)
            
        # Display Metrics
        st.subheader("Evaluation Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Test Accuracy", f"{acc * 100:.2f}%")
        col2.metric("Unseen Test Patients", test_patient_count)
        col3.metric("Total Test Windows", len(y_test))
        
        st.divider()
        
        # Two columns for visuals
        vcol1, vcol2 = st.columns([1, 1])
        
        with vcol1:
            st.markdown("### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred_classes)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            
        with vcol2:
            st.markdown("### Classification Report")
            report = classification_report(y_test, y_pred_classes, target_names=class_names, output_dict=True, zero_division=0)
            df_report = pd.DataFrame(report).transpose()
            # Style the dataframe for better display
            st.dataframe(df_report.style.format("{:.3f}").background_gradient(cmap='Greens', subset=['f1-score']))
