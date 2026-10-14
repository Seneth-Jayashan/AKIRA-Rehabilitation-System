"""
AKIRA Knee Rehabilitation: Edge-AI Web Interface (Backend API)
Interactive dashboard to select between 6 Hybrid Deep Learning Models,
evaluate on the 20% held-out test dataset, and inspect real-time predictions.

Framework: FastAPI + Uvicorn + TensorFlow / Keras
The UI (TailwindCSS + Chart.js) lives in ../web and is served as static files.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pydantic import BaseModel

import tensorflow as tf
from tensorflow import keras
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from scipy import signal

from math_feature_extractor import Clinical16FeatureExtractor
extractor_engine = Clinical16FeatureExtractor(fs=100.0)

# Define paths (self-contained within package)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
WEB_DIR = os.path.join(os.path.dirname(BASE_DIR), "web")

TEST_NPZ_PATH = os.path.join(DATASETS_DIR, "test_20percent_dataset.npz")
TEST_CATALOG_PATH = os.path.join(DATASETS_DIR, "test_20percent_catalog.csv")
METRICS_JSON_PATH = os.path.join(MODELS_DIR, "test_metrics_summary.json")

app = FastAPI(title="AKIRA Knee Rehabilitation AI Web Interface")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for fast response times
CACHE = {
    "test_data": None,
    "test_catalog": None,
    "metrics": None,
    "loaded_models": {},
    "knee_angles": None
}

EXERCISE_NAMES = ["Walking", "Seated Knee Extension", "Mini Squat", "Step-Up"]
QUALITY_NAMES = ["Good (Correct Form)", "Needs Improvement (Lag/Deficit)", "Bad / Compensatory Fault"]

def compute_knee_angle_trajectory(X_seq_sample, fs=100.0, dt=0.01):
    """
    Computes sagittal knee flexion/extension angle from 12-ch IMU trial (200, 12).
    Ch 3,4,5: Thigh Gyro X,Y,Z | Ch 9,10,11: Shin Gyro X,Y,Z
    """
    thigh_gyr = X_seq_sample[:, 3:6]
    shin_gyr = X_seq_sample[:, 9:12]
    rel_gyr = shin_gyr - thigh_gyr
    dom_axis = int(np.argmax(np.std(rel_gyr, axis=0)))
    omega_knee = rel_gyr[:, dom_axis]
    raw_angle = np.cumsum(omega_knee) * dt
    b_hp, a_hp = signal.butter(1, 0.05 / (0.5 * fs), btype='high')
    knee_angle = signal.filtfilt(b_hp, a_hp, raw_angle)
    knee_angle = knee_angle - np.min(knee_angle)
    return knee_angle

def load_data_and_models():
    """Initializes global cache with dataset and metadata."""
    if CACHE["test_data"] is None and os.path.exists(TEST_NPZ_PATH):
        CACHE["test_data"] = np.load(TEST_NPZ_PATH)
        print(f"Loaded test dataset: {CACHE['test_data']['X_seq_raw'].shape[0]} samples.")

    if CACHE["test_catalog"] is None and os.path.exists(TEST_CATALOG_PATH):
        CACHE["test_catalog"] = pd.read_csv(TEST_CATALOG_PATH)
        print(f"Loaded test catalog: {len(CACHE['test_catalog'])} rows.")

    if CACHE["metrics"] is None and os.path.exists(METRICS_JSON_PATH):
        with open(METRICS_JSON_PATH, "r") as f:
            CACHE["metrics"] = json.load(f)
        print(f"Loaded benchmark metrics for {len(CACHE['metrics'])} models.")

    if CACHE["knee_angles"] is None and CACHE["test_data"] is not None:
        X_seq = CACHE["test_data"]["X_seq_raw"]
        angles = []
        for i in range(len(X_seq)):
            angles.append(compute_knee_angle_trajectory(X_seq[i]))
        CACHE["knee_angles"] = np.array(angles)

def get_or_load_model(model_key: str):
    """Loads a Keras model on demand and caches it in RAM."""
    if model_key in CACHE["loaded_models"]:
        return CACHE["loaded_models"][model_key]

    model_filename = f"{model_key}.keras"
    model_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_filename}")

    print(f"Loading {model_key} into memory...")
    model = keras.models.load_model(model_path)
    CACHE["loaded_models"][model_key] = model
    return model

@app.on_event("startup")
def startup_event():
    load_data_and_models()

# Request body models
class PredictRequest(BaseModel):
    model_key: str
    sample_index: int

@app.get("/api/models")
def get_models_list():
    """Returns the list of 6 available hybrid models with descriptions."""
    load_data_and_models()
    models_info = [
        {
            "key": "hybrid_1_cnn",
            "title": "Hybrid 1: 1D-CNN + Kinematics",
            "subtitle": "1D-CNN Feature Fusion (Edge Model)",
            "description": "Extracts multi-scale local sensor waveforms using 1D convolutional layers fused with 16 clinical features. Ultra-low latency on edge MCUs and Android (~8 ms).",
            "params": 55879,
            "latency_ms": "~8 ms",
            "edge_ready": True
        },
        {
            "key": "hybrid_2_bilstm",
            "title": "Hybrid 2: BiLSTM + Kinematics",
            "subtitle": "Bi-Directional Recurrent Sequence Fusion (Benchmark Winner)",
            "description": "Models forward and backward temporal movement cycles, capturing flexion/extension coordination with highest precision (98.2% accuracy).",
            "params": 103559,
            "latency_ms": "~15 ms",
            "edge_ready": True
        },
        {
            "key": "hybrid_3_cnn_bilstm",
            "title": "Hybrid 3: CNN-BiLSTM + Kinematics",
            "subtitle": "Architectural Hybrid: 1D-CNN + BiLSTM Fusion",
            "description": "Combines 1D-CNN micro-pattern extractors with BiLSTM temporal rhythm layers and dense clinical feature embeddings (AKIRA Specification).",
            "params": 97223,
            "latency_ms": "~16 ms",
            "edge_ready": True
        },
        {
            "key": "hybrid_4_resnet",
            "title": "Hybrid 4: 1D-ResNet + Kinematics",
            "subtitle": "Residual Skip-Connection Deep Fusion",
            "description": "Employs residual identity shortcut blocks to retain raw acceleration impulses alongside 16 clinical biomechanical features.",
            "params": 159431,
            "latency_ms": "~18 ms",
            "edge_ready": False
        },
        {
            "key": "hybrid_5_self_attention",
            "title": "Hybrid 5: Self-Attention + Kinematics",
            "subtitle": "Transformer Multi-Head Attention Fusion",
            "description": "Uses 4 multi-head attention mechanisms to automatically attend to critical inflection points in the movement trajectory.",
            "params": 42055,
            "latency_ms": "~12 ms",
            "edge_ready": True
        },
        {
            "key": "hybrid_6_gated_attention",
            "title": "Hybrid 6: Gated Attention + Kinematics",
            "subtitle": "Cross-Modal Co-Attention Dynamic Gating",
            "description": "Clinical biomechanical features dynamically gate and weigh the raw temporal sensor channels before joint fusion.",
            "params": 43335,
            "latency_ms": "~11 ms",
            "edge_ready": True
        }
    ]
    return models_info

@app.get("/api/metrics/{model_key}")
def get_model_metrics(model_key: str):
    """Returns evaluation metrics on the 20% held-out test set for the specified model."""
    load_data_and_models()
    if CACHE["metrics"] and model_key in CACHE["metrics"]:
        m = CACHE["metrics"][model_key]
        return {
            "key": model_key,
            "title": m["title"],
            "params": m["params"],
            "exercise_acc": round(m["exercise_acc"] * 100, 2),
            "exercise_f1": round(m["exercise_f1"] * 100, 2),
            "quality_acc": round(m["quality_acc"] * 100, 2),
            "quality_f1": round(m["quality_f1"] * 100, 2),
            "combined_score": round(m["combined_score"] * 100, 2),
            "cm_exercise": m["cm_exercise"],
            "cm_quality": m["cm_quality"],
            "class_names": EXERCISE_NAMES,
            "quality_names": QUALITY_NAMES
        }
    raise HTTPException(status_code=404, detail=f"Metrics not found for model: {model_key}")

@app.get("/api/all_metrics")
def get_all_metrics():
    """Returns comparison table data across all 6 models."""
    load_data_and_models()
    summary = []
    if CACHE["metrics"]:
        for k, v in CACHE["metrics"].items():
            summary.append({
                "key": k,
                "title": v["title"],
                "params": v["params"],
                "exercise_acc": round(v["exercise_acc"] * 100, 2),
                "exercise_f1": round(v["exercise_f1"] * 100, 2),
                "quality_acc": round(v["quality_acc"] * 100, 2),
                "quality_f1": round(v["quality_f1"] * 100, 2),
                "combined_score": round(v["combined_score"] * 100, 2)
            })
    summary.sort(key=lambda x: x["combined_score"], reverse=True)
    return summary

@app.get("/api/test_samples")
def get_test_samples(exercise: Optional[str] = None, quality: Optional[int] = None):
    """Returns the list of 142 test samples with optional filters."""
    load_data_and_models()
    df = CACHE["test_catalog"]
    if df is None:
        raise HTTPException(status_code=500, detail="Test catalog not loaded.")

    samples = []
    for idx, row in df.iterrows():
        # Filtering
        if exercise and exercise.lower() != "all" and row["exercise_name"].lower() != exercise.lower():
            continue
        if quality is not None and quality != -1 and int(row["quality_grade"]) != quality:
            continue

        samples.append({
            "index": int(idx),
            "file_id": str(row["file_id"]),
            "subject": str(row["subject"]),
            "source_dataset": str(row["source_dataset"]),
            "exercise_name": str(row["exercise_name"]),
            "class_id": int(row["class_id"]),
            "quality_grade": int(row["quality_grade"]),
            "quality_name": QUALITY_NAMES[int(row["quality_grade"])],
            "rom_deg": round(float(row["rom_deg"]), 1),
            "duration_sec": round(float(row["duration_sec"]), 2)
        })
    return samples

@app.get("/api/sample/{sample_idx}")
def get_sample_detail(sample_idx: int):
    """Returns comprehensive kinematic waveforms and 16 clinical features computed dynamically on-the-fly."""
    load_data_and_models()
    if CACHE["test_data"] is None or CACHE["test_catalog"] is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")

    N = len(CACHE["test_catalog"])
    if sample_idx < 0 or sample_idx >= N:
        raise HTTPException(status_code=400, detail=f"Index must be between 0 and {N-1}")

    row = CACHE["test_catalog"].iloc[sample_idx]
    X_seq_raw = CACHE["test_data"]["X_seq_raw"][sample_idx]     # (200, 12) pure raw sensor stream

    # -------------------------------------------------------------------------
    # STEP 1: COMPUTE KNEE ANGLE AND 16 FEATURES LIVE ON THE FLY FROM RAW DATA
    # -------------------------------------------------------------------------
    t0_math = time.time()
    thigh_gyro_y = X_seq_raw[:, 4]
    shin_gyro_y  = X_seq_raw[:, 10]
    knee_angle = extractor_engine.compute_knee_angle(thigh_gyro_y, shin_gyro_y)
    feat_dict, feat_vec = extractor_engine.extract_16_features(knee_angle)
    math_latency_ms = round((time.time() - t0_math) * 1000, 2)

    # Feature names and clinical units
    feat_names = [
        ("Knee Flexion ROM", round(float(feat_vec[0]), 1), "deg", "Max angle minus min angle reached (Live Computed)"),
        ("Peak Flexion Depth", round(float(feat_vec[1]), 1), "deg", "Maximum flexion angle during repetition (Live Computed)"),
        ("Extension Deficit (Quad Lag)", round(float(feat_vec[2]), 1), "deg", "Degrees from full extension (0°) (Live Computed)"),
        ("Peak Angular Velocity", round(float(feat_vec[3]), 1), "deg/s", "Maximum rotational joint speed (Live Computed)"),
        ("Mean Angular Velocity", round(float(feat_vec[4]), 1), "deg/s", "Average movement speed across cycle (Live Computed)"),
        ("SPARC Smoothness Metric", round(float(feat_vec[5]), 2), "index", "Clinical motor coordination (Live Computed)"),
        ("Repetition Count", round(float(feat_vec[6]), 1), "reps", "Detected movement cycles in window"),
        ("Repetition Duration", round(float(feat_vec[7]), 2), "s", "Average time spent per cycle"),
        ("Angle Standard Deviation", round(float(feat_vec[8]), 1), "deg", "Dispersion of joint excursion"),
        ("Velocity Standard Deviation", round(float(feat_vec[9]), 1), "deg/s", "Variability in movement cadence"),
        ("Angle Interquartile Range", round(float(feat_vec[10]), 1), "deg", "Middle 50% movement range"),
        ("Max Flexion Velocity", round(float(feat_vec[11]), 1), "deg/s", "Top speed during bending phase"),
        ("Max Extension Velocity", round(float(feat_vec[12]), 1), "deg/s", "Top speed during straightening phase"),
        ("Mean Knee Angle", round(float(feat_vec[13]), 1), "deg", "Center angle of movement arc"),
        ("Angle RMS", round(float(feat_vec[14]), 1), "deg", "Root-mean-square joint magnitude"),
        ("Normalized Smoothness Rate", round(float(feat_vec[15]), 2), "/s", "SPARC normalized by duration")
    ]

    # All 12 raw channels with metadata
    channels_info = [
        {"id": 1, "name": "Thigh Acc X", "loc": "Thigh", "type": "Acc", "unit": "g", "color": "#2563eb", "data": [round(float(v), 3) for v in X_seq_raw[:, 0]]},
        {"id": 2, "name": "Thigh Acc Y", "loc": "Thigh", "type": "Acc", "unit": "g", "color": "#0284c7", "data": [round(float(v), 3) for v in X_seq_raw[:, 1]]},
        {"id": 3, "name": "Thigh Acc Z", "loc": "Thigh", "type": "Acc", "unit": "g", "color": "#06b6d4", "data": [round(float(v), 3) for v in X_seq_raw[:, 2]]},
        {"id": 4, "name": "Thigh Gyro X", "loc": "Thigh", "type": "Gyro", "unit": "deg/s", "color": "#d97706", "data": [round(float(v), 2) for v in X_seq_raw[:, 3]]},
        {"id": 5, "name": "Thigh Gyro Y", "loc": "Thigh", "type": "Gyro", "unit": "deg/s", "color": "#f59e0b", "data": [round(float(v), 2) for v in X_seq_raw[:, 4]]},
        {"id": 6, "name": "Thigh Gyro Z", "loc": "Thigh", "type": "Gyro", "unit": "deg/s", "color": "#fbbf24", "data": [round(float(v), 2) for v in X_seq_raw[:, 5]]},
        {"id": 7, "name": "Shin Acc X", "loc": "Shin", "type": "Acc", "unit": "g", "color": "#059669", "data": [round(float(v), 3) for v in X_seq_raw[:, 6]]},
        {"id": 8, "name": "Shin Acc Y", "loc": "Shin", "type": "Acc", "unit": "g", "color": "#10b981", "data": [round(float(v), 3) for v in X_seq_raw[:, 7]]},
        {"id": 9, "name": "Shin Acc Z", "loc": "Shin", "type": "Acc", "unit": "g", "color": "#34d399", "data": [round(float(v), 3) for v in X_seq_raw[:, 8]]},
        {"id": 10, "name": "Shin Gyro X", "loc": "Shin", "type": "Gyro", "unit": "deg/s", "color": "#dc2626", "data": [round(float(v), 2) for v in X_seq_raw[:, 9]]},
        {"id": 11, "name": "Shin Gyro Y", "loc": "Shin", "type": "Gyro", "unit": "deg/s", "color": "#ef4444", "data": [round(float(v), 2) for v in X_seq_raw[:, 10]]},
        {"id": 12, "name": "Shin Gyro Z", "loc": "Shin", "type": "Gyro", "unit": "deg/s", "color": "#f87171", "data": [round(float(v), 2) for v in X_seq_raw[:, 11]]}
    ]

    raw_channels_summary = []
    for ch in channels_info:
        raw_channels_summary.append({
            "id": ch["id"],
            "name": ch["name"],
            "loc": ch["loc"],
            "type": ch["type"],
            "unit": ch["unit"],
            "color": ch["color"],
            "min": round(float(np.min(ch["data"])), 2),
            "max": round(float(np.max(ch["data"])), 2),
            "mean": round(float(np.mean(ch["data"])), 2)
        })

    raw_matrix_rows = []
    for t_idx in range(min(25, len(X_seq_raw))):
        raw_matrix_rows.append({
            "t": round(t_idx * 0.01, 2),
            "values": [round(float(X_seq_raw[t_idx, c]), 2) for c in range(12)]
        })

    time_axis = [round(t * 0.01, 2) for t in range(200)]

    return {
        "index": sample_idx,
        "file_id": str(row["file_id"]),
        "subject": str(row["subject"]),
        "source_dataset": str(row["source_dataset"]),
        "exercise_name": str(row["exercise_name"]),
        "class_id": int(row["class_id"]),
        "quality_grade": int(row["quality_grade"]),
        "quality_name": QUALITY_NAMES[int(row["quality_grade"])],
        "duration_sec": round(float(row["duration_sec"]), 2),
        "math_latency_ms": math_latency_ms,
        "features": [{"name": fn[0], "value": fn[1], "unit": fn[2], "desc": fn[3]} for fn in feat_names],
        "time_axis": time_axis,
        "knee_angle": [round(float(a), 2) for a in knee_angle],
        "all_12_channels": channels_info,
        "raw_channels_summary": raw_channels_summary,
        "raw_matrix_rows": raw_matrix_rows,
        "thigh_acc_y": [round(float(a), 3) for a in X_seq_raw[:, 1]],
        "shin_acc_y": [round(float(a), 3) for a in X_seq_raw[:, 7]],
        "thigh_gyro_z": [round(float(g), 2) for g in X_seq_raw[:, 5]],
        "shin_gyro_z": [round(float(g), 2) for g in X_seq_raw[:, 11]]
    }

@app.post("/api/predict")
def run_model_inference(req: PredictRequest):
    """Executes single-sample live end-to-end inference from raw sensor data."""
    load_data_and_models()
    sample_idx = req.sample_index
    model_key = req.model_key

    if CACHE["test_data"] is None or CACHE["test_catalog"] is None:
        raise HTTPException(status_code=500, detail="Data not loaded.")

    N = len(CACHE["test_catalog"])
    if sample_idx < 0 or sample_idx >= N:
        raise HTTPException(status_code=400, detail=f"Index must be between 0 and {N-1}")

    model = get_or_load_model(model_key)

    # 1. Grab pure raw 12-channel sequence:
    X_seq_raw = CACHE["test_data"]["X_seq_raw"][sample_idx]

    # 2. RUN MATH MODEL ON RAW SENSOR DATA TO EXTRACT 16 CLINICAL FEATURES ON-THE-FLY:
    t0_math = time.time()
    thigh_gyro_y = X_seq_raw[:, 4]
    shin_gyro_y  = X_seq_raw[:, 10]
    knee_angle = extractor_engine.compute_knee_angle(thigh_gyro_y, shin_gyro_y)
    feat_dict, feat_vec = extractor_engine.extract_16_features(knee_angle)
    math_latency_ms = round((time.time() - t0_math) * 1000, 2)

    # 3. NORMALIZE ON-THE-FLY:
    scaler_mean = CACHE["test_data"]["scaler_mean"]
    scaler_scale = CACHE["test_data"]["scaler_scale"]
    X_feat_sample = ((feat_vec - scaler_mean) / (scaler_scale + 1e-6)).reshape(1, 16)

    means = CACHE["test_data"]["means"]
    stds = CACHE["test_data"]["stds"]
    X_seq_sample = ((X_seq_raw - means) / (stds + 1e-6)).reshape(1, 200, 12)

    # 4. RUN AI MODEL INFERENCE:
    t0_ai = time.time()
    preds = model.predict(
        {"input_a_raw_seq": X_seq_sample, "input_b_clinical_feat": X_feat_sample},
        verbose=0
    )
    ai_latency_ms = round((time.time() - t0_ai) * 1000, 2)
    total_latency_ms = round(math_latency_ms + ai_latency_ms, 2)

    prob_class = preds[0][0].tolist()
    prob_qual = preds[1][0].tolist()

    pred_class_idx = int(np.argmax(prob_class))
    pred_qual_idx = int(np.argmax(prob_qual))

    row = CACHE["test_catalog"].iloc[sample_idx]
    true_class_idx = int(row["class_id"])
    true_qual_idx = int(row["quality_grade"])

    class_match = (pred_class_idx == true_class_idx)
    qual_match = (pred_qual_idx == true_qual_idx)

    # Clinical interpretation rules based on freshly computed features:
    rom_val = float(feat_vec[0])
    lag_val = float(feat_vec[2])
    sparc_val = float(feat_vec[5])

    feedback = []
    if pred_class_idx == 2:  # Mini Squat
        if rom_val > 50.0:
            feedback.append(f"⚠️ Excessive squat depth detected ({rom_val:.1f}° flexion). For early/mid-stage Tibial Plateau ORIF, maintain depth within safe 30°-45° zone to prevent excessive articular shear stress.")
        else:
            feedback.append(f"✅ Knee flexion depth within safe orthopedic limits for post-tibial plateau repair ({rom_val:.1f}°).")
    elif pred_class_idx == 1:  # Seated Extension
        if lag_val > 8.0:
            feedback.append(f"⚠️ Quadriceps extension lag of {lag_val:.1f}° detected. Target active terminal knee extension with isometric holds.")
        else:
            feedback.append("✅ Full terminal knee extension achieved with zero quadriceps lag.")

    if pred_qual_idx == 0:
        feedback.append("✅ Smooth motor control without compensatory tremors (SPARC: {:.2f}). Form verified as Good.".format(sparc_val))
    elif pred_qual_idx == 1:
        feedback.append("⚠️ Sub-optimal execution speed or slight hesitation during reversal phase (Moderate Quality).")
    else:
        feedback.append("❌ Pathological movement compensation detected (Jerk / sudden drop / compensatory trunk sway). Pause and re-align.")

    return {
        "model_key": model_key,
        "sample_index": sample_idx,
        "pipeline_latency": {
            "math_ms": math_latency_ms,
            "ai_ms": ai_latency_ms,
            "total_ms": total_latency_ms,
            "on_the_fly": True
        },
        "predicted_exercise": {
            "class_id": pred_class_idx,
            "name": EXERCISE_NAMES[pred_class_idx],
            "confidence": round(prob_class[pred_class_idx] * 100, 1),
            "probabilities": [round(p * 100, 1) for p in prob_class],
            "class_names": EXERCISE_NAMES,
            "matches_ground_truth": class_match,
            "true_class_id": true_class_idx,
            "true_name": EXERCISE_NAMES[true_class_idx]
        },
        "predicted_quality": {
            "grade_id": pred_qual_idx,
            "name": QUALITY_NAMES[pred_qual_idx],
            "confidence": round(prob_qual[pred_qual_idx] * 100, 1),
            "probabilities": [round(p * 100, 1) for p in prob_qual],
            "grade_names": QUALITY_NAMES,
            "matches_ground_truth": qual_match,
            "true_grade_id": true_qual_idx,
            "true_name": QUALITY_NAMES[true_qual_idx]
        },
        "clinical_feedback": feedback
    }

@app.get("/", include_in_schema=False)
def get_dashboard_ui():
    """Serves the single-page interactive clinical dashboard (web/index.html)."""
    return FileResponse(os.path.join(WEB_DIR, "index.html"))

# Static assets (css/, js/) — mounted last so /api/* routes take priority
app.mount("/", StaticFiles(directory=WEB_DIR), name="web")

if __name__ == "__main__":
    print("=" * 70)
    print("AKIRA Edge-AI Knee Rehabilitation Dashboard Starting...")
    print("Server URL: http://127.0.0.1:8000")
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
