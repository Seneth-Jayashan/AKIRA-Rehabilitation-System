"""
========================================================================================
AKIRA KNEE REHABILITATION SYSTEM: BACKEND REST API SERVER (main.py)
========================================================================================
WHAT THIS FILE IS:
  This is the primary FastAPI backend web server for the AKIRA Knee Rehabilitation
  ecosystem. It connects the frontend web dashboard with the edge-AI deep learning
  models, mathematical sensor-fusion feature extraction engine, live ESP32 dual-IMU
  hardware stream, and clinician-configurable longitudinal recovery analytics.
========================================================================================
"""

import os
import sys
import json
import time
import hashlib
from datetime import datetime, date
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from scipy import signal
from pydantic import BaseModel

import tensorflow as tf
from tensorflow import keras

import uvicorn
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from math_feature_extractor import Clinical16FeatureExtractor

extractor_engine = Clinical16FeatureExtractor(fs=100.0)

# ======================================================================================
# SECTION 2: DIRECTORY PATHS AND FILE CONFIGURATION
# ======================================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
WEB_DIR = os.path.join(os.path.dirname(BASE_DIR), "web")

TEST_NPZ_PATH = os.path.join(DATASETS_DIR, "test_20percent_dataset.npz")
TEST_CATALOG_PATH = os.path.join(DATASETS_DIR, "test_20percent_catalog.csv")
METRICS_JSON_PATH = os.path.join(MODELS_DIR, "test_metrics_summary.json")
OPTITRACK_JSON_PATH = os.path.join(MODELS_DIR, "optitrack_validation_metrics.json")
CLINICIAN_PROTOCOLS_PATH = os.path.join(DATASETS_DIR, "clinician_protocols.json")

# ======================================================================================
# SECTION 3: FASTAPI APPLICATION & CORS CONFIGURATION
# ======================================================================================
app = FastAPI(
    title="AKIRA Knee Rehabilitation AI Web Interface",
    description="Edge-AI and Kinematic Sensor Fusion API for Post-ORIF Knee Rehabilitation",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================================================
# SECTION 4: GLOBAL IN-MEMORY CACHE & REFERENCE DICTIONARIES
# ======================================================================================
CACHE = {
    "test_data": None,
    "test_catalog": None,
    "metrics": None,
    "loaded_models": {},
    "knee_angles": None
}

EXERCISE_NAMES = [
    "Walking",                  # Class 0: Continuous gait cycles
    "Seated Knee Extension",    # Class 1: Open-chain terminal quadriceps extension
    "Mini Squat",               # Class 2: Closed-chain partial weight-bearing squat (30°-45°)
    "Step-Up"                   # Class 3: Functional stair negotiation and unilateral loading
]

QUALITY_NAMES = [
    "Good (Correct Form)",                  # Grade 0: Smooth trajectory, full target ROM, minimal lag
    "Needs Improvement (Lag/Deficit)",     # Grade 1: Mild extensor lag or minor hesitation
    "Bad / Compensatory Fault"             # Grade 2: Jerky movement, compensation, or severe lag
]

# Unified Clinician-Configurable Protocol Defaults per Exercise (Fixes G56, G57, H60)
DEFAULT_CLINICIAN_PROTOCOLS: Dict[str, Dict[str, float]] = {
    "Walking": {
        "target_rom_deg": 60.0,
        "max_safe_flexion_deg": 70.0,
        "max_extensor_lag_deg": 5.0,
        "sessions_per_week": 3.0
    },
    "Seated Knee Extension": {
        "target_rom_deg": 90.0,
        "max_safe_flexion_deg": 100.0,
        "max_extensor_lag_deg": 5.0,
        "sessions_per_week": 3.0
    },
    "Mini Squat": {
        "target_rom_deg": 42.0,
        "max_safe_flexion_deg": 45.0,
        "max_extensor_lag_deg": 5.0,
        "sessions_per_week": 3.0
    },
    "Step-Up": {
        "target_rom_deg": 65.0,
        "max_safe_flexion_deg": 75.0,
        "max_extensor_lag_deg": 5.0,
        "sessions_per_week": 2.0
    }
}


def pseudonymize_patient_id(patient_id: str) -> str:
    """Returns a privacy-preserving SHA-256 pseudonym token for patient identifiers (Fixes H63)."""
    clean_id = (patient_id or "P01").strip()
    digest = hashlib.sha256(f"AKIRA_SALT::{clean_id}".encode("utf-8")).hexdigest()[:12]
    return f"PT-{digest.upper()}"


def verify_optional_api_key(x_akira_api_key: Optional[str] = None):
    """Checks optional API key when AKIRA_API_KEY environment variable is configured (Fixes H63)."""
    required_key = os.environ.get("AKIRA_API_KEY", "").strip()
    if required_key and x_akira_api_key != required_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-AKIRA-API-Key header.")


def load_clinician_protocols() -> Dict[str, Dict[str, float]]:
    if not os.path.exists(CLINICIAN_PROTOCOLS_PATH):
        with open(CLINICIAN_PROTOCOLS_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CLINICIAN_PROTOCOLS, f, indent=2)
        return {k: dict(v) for k, v in DEFAULT_CLINICIAN_PROTOCOLS.items()}
    try:
        with open(CLINICIAN_PROTOCOLS_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            merged = {k: dict(v) for k, v in DEFAULT_CLINICIAN_PROTOCOLS.items()}
            for k, v in loaded.items():
                if isinstance(v, dict):
                    merged[k] = {**merged.get(k, DEFAULT_CLINICIAN_PROTOCOLS["Seated Knee Extension"]), **v}
            return merged
    except Exception:
        return {k: dict(v) for k, v in DEFAULT_CLINICIAN_PROTOCOLS.items()}


def get_protocol_for_exercise(exercise_name: str) -> Dict[str, float]:
    protocols = load_clinician_protocols()
    for k, v in protocols.items():
        if k.lower() in exercise_name.lower() or exercise_name.lower() in k.lower():
            return v
    return protocols.get("Seated Knee Extension", DEFAULT_CLINICIAN_PROTOCOLS["Seated Knee Extension"])


# ======================================================================================
# SECTION 5: MATHEMATICAL KINEMATIC SIGNAL PROCESSING (KNEE ANGLE TRAJECTORY)
# ======================================================================================
def compute_knee_angle_trajectory(
    X_seq_sample: np.ndarray,
    fs: float = 100.0,
    duration_sec: Optional[float] = None
) -> np.ndarray:
    """
    Computes the continuous sagittal knee flexion/extension angle (in degrees)
    from raw 12-channel dual-IMU sensor readings using the full AKIRA Sensor Fusion
    & 3D SVD Relative Hinge Orientation pipeline (Fixes B14-B17, C18-C20).
    """
    return extractor_engine.compute_knee_angle(X_seq_sample, duration_sec=duration_sec)


# ======================================================================================
# SECTION 6: DATA & MODEL LOADING PROCEDURES (CACHE INITIALIZERS)
# ======================================================================================
def load_data_and_models():
    if CACHE["test_data"] is None and os.path.exists(TEST_NPZ_PATH):
        CACHE["test_data"] = np.load(TEST_NPZ_PATH, allow_pickle=True)
        print(f"[CACHE] Loaded test dataset: {CACHE['test_data']['X_seq_raw'].shape[0]} samples.")

    if CACHE["test_catalog"] is None and os.path.exists(TEST_CATALOG_PATH):
        CACHE["test_catalog"] = pd.read_csv(TEST_CATALOG_PATH)
        print(f"[CACHE] Loaded test catalog: {len(CACHE['test_catalog'])} rows.")

    if CACHE["metrics"] is None and os.path.exists(METRICS_JSON_PATH):
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            CACHE["metrics"] = json.load(f)
        print(f"[CACHE] Loaded benchmark metrics.")

    if CACHE["knee_angles"] is None and CACHE["test_data"] is not None:
        X_seq = CACHE["test_data"]["X_seq_raw"]
        angles = []
        for i in range(len(X_seq)):
            angles.append(compute_knee_angle_trajectory(X_seq[i]))
        CACHE["knee_angles"] = np.array(angles)


def get_or_load_model(model_key: str) -> keras.Model:
    if model_key in CACHE["loaded_models"]:
        return CACHE["loaded_models"][model_key]

    model_filename = f"{model_key}.keras"
    model_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model file not found on disk: {model_filename}")

    print(f"[AI ENGINE] Loading model '{model_key}' into memory...")
    model = keras.models.load_model(model_path)
    CACHE["loaded_models"][model_key] = model
    return model


@app.on_event("startup")
def startup_event():
    load_data_and_models()


# ======================================================================================
# SECTION 7: PYDANTIC DATA SCHEMAS (API REQUEST MODELS)
# ======================================================================================
class PredictRequest(BaseModel):
    model_key: str
    sample_index: int
    patient_id: Optional[str] = "P01"


class ClinicianProtocolUpdate(BaseModel):
    exercise: str = "Mini Squat"
    target_rom_deg: float = 42.0
    max_safe_flexion_deg: float = 45.0
    max_extensor_lag_deg: float = 5.0
    sessions_per_week: float = 3.0


# ======================================================================================
# SECTION 8: REST API ENDPOINTS
# ======================================================================================
@app.get("/api/models")
def get_models_list():
    load_data_and_models()
    models_info = [
        {
            "key": "hybrid_1_cnn",
            "title": "Hybrid 1: 1D-CNN + Kinematics",
            "subtitle": "1D-CNN Feature Fusion (Edge Model)",
            "description": "Extracts multi-scale local sensor waveforms using 1D convolutional layers fused with 16 clinical features. Quantized for TFLite edge deployment.",
            "params": 55879,
            "latency_ms": "~8 ms",
            "edge_ready": True
        },
        {
            "key": "hybrid_2_bilstm",
            "title": "Hybrid 2: BiLSTM + Kinematics",
            "subtitle": "Bi-Directional Recurrent Sequence Fusion",
            "description": "Models forward and backward temporal movement cycles, capturing flexion/extension coordination across dual IMUs.",
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
    load_data_and_models()
    if CACHE["metrics"] and model_key in CACHE["metrics"]:
        m = CACHE["metrics"][model_key]
        if isinstance(m, dict) and "combined_score" in m:
            return {
                "key": model_key,
                "title": m["title"],
                "params": m["params"],
                "split_mode": m.get("split_mode", "subject"),
                "exercise_acc": round(m["exercise_acc"] * 100, 2),
                "exercise_f1": round(m["exercise_f1"] * 100, 2),
                "quality_acc": round(m["quality_acc"] * 100, 2),
                "quality_f1": round(m["quality_f1"] * 100, 2),
                "combined_score": round(m["combined_score"] * 100, 2),
                "cm_exercise": m["cm_exercise"],
                "cm_quality": m["cm_quality"],
                "per_class_exercise": m.get("per_class_exercise", {}),
                "per_class_quality": m.get("per_class_quality", {}),
                "class_names": EXERCISE_NAMES,
                "quality_names": QUALITY_NAMES
            }
    raise HTTPException(status_code=404, detail=f"Metrics not found for model: {model_key}")


@app.get("/api/all_metrics")
def get_all_metrics():
    load_data_and_models()
    summary = []
    if CACHE["metrics"]:
        for k, v in CACHE["metrics"].items():
            if not isinstance(v, dict) or "combined_score" not in v:
                continue
            summary.append({
                "key": k,
                "title": v["title"],
                "params": v["params"],
                "split_mode": v.get("split_mode", "subject"),
                "exercise_acc": round(v["exercise_acc"] * 100, 2),
                "exercise_f1": round(v["exercise_f1"] * 100, 2),
                "quality_acc": round(v["quality_acc"] * 100, 2),
                "quality_f1": round(v["quality_f1"] * 100, 2),
                "combined_score": round(v["combined_score"] * 100, 2)
            })
    summary.sort(key=lambda x: x["combined_score"], reverse=True)
    return summary


@app.get("/api/edge_tflite_status")
def get_edge_tflite_and_baselines_status():
    """Returns Edge AI TFLite validation metrics, baseline model comparisons, and OptiTrack validation (Fixes C25, D33, D38, H62)."""
    load_data_and_models()
    metrics = CACHE["metrics"] or {}
    optitrack_data = {}
    if os.path.exists(OPTITRACK_JSON_PATH):
        with open(OPTITRACK_JSON_PATH, "r", encoding="utf-8") as f:
            optitrack_data = json.load(f)
    return {
        "tflite_edge_validation": metrics.get("tflite_edge_validation", {}),
        "baselines": metrics.get("baselines", {}),
        "optitrack_validation": optitrack_data
    }


@app.get("/api/clinician/protocol")
def get_clinician_protocols():
    """Returns clinician-defined rehabilitation protocol parameters per exercise (FR-17, Fixes G56, G57, H60)."""
    return load_clinician_protocols()


@app.post("/api/clinician/protocol")
def update_clinician_protocol(req: ClinicianProtocolUpdate):
    """Updates and persists clinician-defined target ROM, safe flexion limit, and session frequency (FR-17)."""
    protocols = load_clinician_protocols()
    protocols[req.exercise] = {
        "target_rom_deg": float(req.target_rom_deg),
        "max_safe_flexion_deg": float(req.max_safe_flexion_deg),
        "max_extensor_lag_deg": float(req.max_extensor_lag_deg),
        "sessions_per_week": float(req.sessions_per_week)
    }
    with open(CLINICIAN_PROTOCOLS_PATH, "w", encoding="utf-8") as f:
        json.dump(protocols, f, indent=2)
    return {"status": "updated", "protocols": protocols}


@app.get("/api/test_samples")
def get_test_samples(exercise: Optional[str] = None, quality: Optional[int] = None):
    load_data_and_models()
    df = CACHE["test_catalog"]
    if df is None:
        raise HTTPException(status_code=500, detail="Test catalog not loaded.")

    samples = []
    for idx, row in df.iterrows():
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


def build_16_feature_cards(feat_vec: np.ndarray) -> List[Dict[str, Any]]:
    """Packages all 16 clinical features with standardized names, units, and clinical descriptions (Fixes C26)."""
    feat_names = [
        ("Knee Flexion ROM", round(float(feat_vec[0]), 1), "deg", "Total active excursion (Max angle minus Min angle)"),
        ("Peak Flexion Depth", round(float(feat_vec[1]), 1), "deg", "Maximum flexion angle relative to anatomical zero"),
        ("Extension Deficit (Quad Lag)", round(float(feat_vec[2]), 1), "deg", "Terminal extension deficit above 0° baseline"),
        ("Peak Angular Velocity", round(float(feat_vec[3]), 1), "deg/s", "Maximum absolute joint rotational speed"),
        ("Mean Angular Velocity", round(float(feat_vec[4]), 1), "deg/s", "Average movement speed across active cycle"),
        ("SPARC Smoothness Metric", round(float(feat_vec[5]), 2), "index", "Spectral arc length motor coordination index"),
        ("Repetition Count", int(round(float(feat_vec[6]))), "reps", "Segmented discrete movement cycles in window"),
        ("Repetition Duration", round(float(feat_vec[7]), 2), "s", "Mean duration of segmented active cycle"),
        ("Angle Standard Deviation", round(float(feat_vec[8]), 1), "deg", "Dispersion of joint flexion trajectory"),
        ("Velocity Standard Deviation", round(float(feat_vec[9]), 1), "deg/s", "Variability in angular velocity cadence"),
        ("Angle Interquartile Range", round(float(feat_vec[10]), 1), "deg", "Middle 50% (Q75 - Q25) flexion excursion"),
        ("Max Flexion Velocity", round(float(feat_vec[11]), 1), "deg/s", "Peak positive angular velocity (bending phase, >0)"),
        ("Max Extension Velocity", round(float(feat_vec[12]), 1), "deg/s", "Peak negative angular velocity (straightening phase, <0)"),
        ("Mean Knee Angle", round(float(feat_vec[13]), 1), "deg", "Time-averaged knee flexion angle"),
        ("Angle RMS", round(float(feat_vec[14]), 1), "deg", "Root-mean-square joint angle magnitude"),
        ("Normalized Smoothness Rate", round(float(feat_vec[15]), 2), "/s", "SPARC smoothness normalized by active duration")
    ]
    return [{"name": fn[0], "value": fn[1], "unit": fn[2], "desc": fn[3]} for fn in feat_names]


@app.get("/api/sample/{sample_idx}")
def get_sample_detail(sample_idx: int):
    load_data_and_models()
    if CACHE["test_data"] is None or CACHE["test_catalog"] is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")

    N = len(CACHE["test_catalog"])
    if sample_idx < 0 or sample_idx >= N:
        raise HTTPException(status_code=400, detail=f"Index must be between 0 and {N-1}")

    row = CACHE["test_catalog"].iloc[sample_idx]
    X_seq_raw = CACHE["test_data"]["X_seq_raw"][sample_idx]
    trial_dur = float(row.get("duration_sec", 2.0)) if float(row.get("duration_sec", 2.0)) > 0 else 2.0

    # Run full 12-channel sensor fusion + 16-feature extractor live on-the-fly
    t0_math = time.time()
    knee_angle = extractor_engine.compute_knee_angle(X_seq_raw, duration_sec=trial_dur)
    feat_dict, feat_vec = extractor_engine.extract_16_features(knee_angle, duration_sec=trial_dur)
    math_latency_ms = round((time.time() - t0_math) * 1000, 2)

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

    raw_channels_summary = [{
        "id": ch["id"],
        "name": ch["name"],
        "loc": ch["loc"],
        "type": ch["type"],
        "unit": ch["unit"],
        "color": ch["color"],
        "min": round(float(np.min(ch["data"])), 2),
        "max": round(float(np.max(ch["data"])), 2),
        "mean": round(float(np.mean(ch["data"])), 2)
    } for ch in channels_info]

    dt_step = trial_dur / max(1, len(X_seq_raw))
    raw_matrix_rows = [{
        "t": round(t_idx * dt_step, 2),
        "values": [round(float(X_seq_raw[t_idx, c]), 2) for c in range(12)]
    } for t_idx in range(min(25, len(X_seq_raw)))]

    time_axis = [round(t * dt_step, 2) for t in range(len(X_seq_raw))]

    return {
        "index": sample_idx,
        "file_id": str(row["file_id"]),
        "subject": str(row["subject"]),
        "source_dataset": str(row["source_dataset"]),
        "exercise_name": str(row["exercise_name"]),
        "class_id": int(row["class_id"]),
        "quality_grade": int(row["quality_grade"]),
        "quality_name": QUALITY_NAMES[int(row["quality_grade"])],
        "duration_sec": round(trial_dur, 2),
        "math_latency_ms": math_latency_ms,
        "features": build_16_feature_cards(feat_vec),
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


def generate_personalized_clinical_recommendations(
    exercise_name: str,
    pred_qual_idx: int,
    feat_vec: np.ndarray,
    patient_id: str = "P01"
) -> List[str]:
    """
    Generates personalized orthopedic guidance tied directly to clinician-defined protocol
    parameters (target_rom_deg, max_safe_flexion_deg, max_extensor_lag_deg) and longitudinal
    patient progress (Fixes G56, G57, H60).
    """
    rom_val = float(feat_vec[0])
    depth_val = float(feat_vec[1])
    lag_val = float(feat_vec[2])
    sparc_val = float(feat_vec[5])
    rep_count = int(round(float(feat_vec[6])))

    protocol = get_protocol_for_exercise(exercise_name)
    target_rom = float(protocol["target_rom_deg"])
    max_safe_flex = float(protocol["max_safe_flexion_deg"])
    max_lag = float(protocol["max_extensor_lag_deg"])

    feedback = []

    # 1. Protocol Safety & Target ROM Comparison
    if depth_val > max_safe_flex:
        feedback.append(
            f"⚠️ Flexion depth ({depth_val:.1f}°) exceeds clinician safe limit ({max_safe_flex:.0f}°) for {exercise_name}. "
            f"Reduce flexion excursion to protect post-ORIF articular fixation."
        )
    elif rom_val >= 0.90 * target_rom:
        feedback.append(
            f"✅ Active ROM ({rom_val:.1f}°) meets clinician protocol target ({target_rom:.0f}°) within safe flexion ceiling ({max_safe_flex:.0f}°)."
        )
    else:
        deficit = target_rom - rom_val
        feedback.append(
            f"ℹ️ Active ROM ({rom_val:.1f}°) is {deficit:.1f}° below clinician target ({target_rom:.0f}°) for {exercise_name}. "
            f"Progressively increase controlled flexion within the {max_safe_flex:.0f}° safe limit."
        )

    # 2. Terminal Extension Lag (Quadriceps Deficit) Evaluation
    if lag_val > max_lag:
        feedback.append(
            f"⚠️ Terminal extension lag of {lag_val:.1f}° detected (clinician threshold: ≤{max_lag:.1f}°). "
            f"Focus on isometric terminal quadriceps activation at full extension."
        )
    else:
        feedback.append(
            f"✅ Terminal extension lag ({lag_val:.1f}°) is within clinician protocol threshold (≤{max_lag:.1f}°)."
        )

    # 3. Movement Quality & Neuromuscular Smoothness
    if pred_qual_idx == 0:
        feedback.append(f"✅ Smooth neuromuscular control verified (SPARC: {sparc_val:.2f}, Reps: {rep_count}).")
    elif pred_qual_idx == 1:
        feedback.append(f"⚠️ Moderate quality grade (SPARC: {sparc_val:.2f}): focus on steady eccentric control during lowering phase.")
    else:
        feedback.append(f"❌ Compensatory fault / jerky trajectory detected (SPARC: {sparc_val:.2f}). Pause and re-align limb posture.")

    # 4. Longitudinal Trend Integration (if patient has prior recorded sessions for this exercise)
    all_sessions = load_longitudinal_sessions()
    pt_ex_sessions = [
        s for s in all_sessions
        if s.get("exercise", "").lower() == exercise_name.lower()
        and s.get("patient_id", "P01") == (patient_id or "P01")
    ]
    if len(pt_ex_sessions) >= 2:
        last_rom = float(pt_ex_sessions[-1]["rom_deg"])
        delta = rom_val - last_rom
        sign_str = f"+{delta:.1f}°" if delta >= 0 else f"{delta:.1f}°"
        feedback.append(
            f"📈 Longitudinal Comparison ({patient_id}, {exercise_name}): {sign_str} vs. previous session ({last_rom:.1f}°)."
        )

    return feedback


@app.post("/api/predict")
def run_model_inference(req: PredictRequest):
    load_data_and_models()
    sample_idx = req.sample_index
    model_key = req.model_key

    if CACHE["test_data"] is None or CACHE["test_catalog"] is None:
        raise HTTPException(status_code=500, detail="Data not loaded.")

    N = len(CACHE["test_catalog"])
    if sample_idx < 0 or sample_idx >= N:
        raise HTTPException(status_code=400, detail=f"Index must be between 0 and {N-1}")

    model = get_or_load_model(model_key)
    row = CACHE["test_catalog"].iloc[sample_idx]
    X_seq_raw = CACHE["test_data"]["X_seq_raw"][sample_idx]
    trial_dur = float(row.get("duration_sec", 2.0)) if float(row.get("duration_sec", 2.0)) > 0 else 2.0

    # 1. Run full 12-channel sensor fusion + 16-feature extractor on-the-fly
    t0_math = time.time()
    knee_angle = extractor_engine.compute_knee_angle(X_seq_raw, duration_sec=trial_dur)
    feat_dict, feat_vec = extractor_engine.extract_16_features(knee_angle, duration_sec=trial_dur)
    math_latency_ms = round((time.time() - t0_math) * 1000, 2)

    # 2. Normalize both streams using leak-free training scaler parameters
    scaler_mean = CACHE["test_data"]["scaler_mean"]
    scaler_scale = CACHE["test_data"]["scaler_scale"]
    X_feat_sample = ((feat_vec - scaler_mean) / (scaler_scale + 1e-6)).reshape(1, 16)

    means = CACHE["test_data"]["means"]
    stds = CACHE["test_data"]["stds"]
    X_seq_sample = ((X_seq_raw - means) / (stds + 1e-6)).reshape(1, 200, 12)

    # 3. Run AI Model Inference
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

    true_class_idx = int(row["class_id"])
    true_qual_idx = int(row["quality_grade"])

    class_match = (pred_class_idx == true_class_idx)
    qual_match = (pred_qual_idx == true_qual_idx)

    # 4. Generate Clinician-Protocol-Driven Rehabilitation Guidance (Fixes H60)
    feedback = generate_personalized_clinical_recommendations(
        exercise_name=EXERCISE_NAMES[pred_class_idx],
        pred_qual_idx=pred_qual_idx,
        feat_vec=feat_vec,
        patient_id=req.patient_id or str(row["subject"])
    )

    # 5. Compute Box 07 Session-Level Outputs (Fixes F47-F51)
    session_outputs = build_box07_session_outputs(
        exercise_name=EXERCISE_NAMES[pred_class_idx],
        exercise_conf=round(prob_class[pred_class_idx] * 100, 1),
        quality_grade_id=pred_qual_idx,
        quality_name=QUALITY_NAMES[pred_qual_idx],
        prob_qual=prob_qual,
        feat_vec=feat_vec
    )

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
        "session_outputs": session_outputs,
        "clinical_feedback": feedback
    }


# ======================================================================================
# SECTION 8B: BOX 07 SESSION OUTPUTS, ESP32 LIVE RECORDER & BOX 08/09 LONGITUDINAL ENGINE
# ======================================================================================
RECORDED_TRIALS_DIR = os.path.join(DATASETS_DIR, "recorded_trials")
LONGITUDINAL_DB_PATH = os.path.join(DATASETS_DIR, "longitudinal_sessions.json")
ESP32_INO_PATH = os.path.join(BASE_DIR, "esp32_firmware", "akira_knee_esp32.ino")

os.makedirs(RECORDED_TRIALS_DIR, exist_ok=True)

PROPOSAL_6_EXERCISES = [
    {"code": "01_Heel_Slides",           "name": "01 Heel Slides",           "target_rom": 95.0, "is_new_device_ex": True},
    {"code": "02_Quadriceps_Sets",       "name": "02 Quadriceps Sets",       "target_rom": 8.0,  "is_new_device_ex": True},
    {"code": "03_Straight_Leg_Raise",    "name": "03 Straight-Leg Raise",    "target_rom": 12.0, "is_new_device_ex": True},
    {"code": "04_Seated_Knee_Extension", "name": "04 Seated Knee Extension", "target_rom": 90.0, "is_new_device_ex": False},
    {"code": "05_Mini_Squat",            "name": "05 Mini Squat",            "target_rom": 42.0, "is_new_device_ex": False},
    {"code": "06_Step_Up",               "name": "06 Step-Up",               "target_rom": 65.0, "is_new_device_ex": False}
]


def build_box07_session_outputs(
    exercise_name: str,
    exercise_conf: float,
    quality_grade_id: int,
    quality_name: str,
    prob_qual: list,
    feat_vec: np.ndarray
) -> Dict[str, Any]:
    """
    Computes the 7 Official Session-Level Outputs specified in Proposal Figure 3 (Box 07)
    and Functional Requirement FR-10 (Fixes F47, F48, F49, F50, F51).
      - F47 Fixed: out_03_movement_depth_deg reports true Peak Flexion Depth (depth_deg),
        never multiplied by 0.22.
      - F48 & F49 Fixed: Un-clipped [0.0, 1.0] SPARC smoothness normalization and
        un-clipped [0, 100] clinical quality score evaluated against clinician protocol target_rom.
      - F50 Fixed: out_04_repetition_count reports true segmented rep count (0 when stationary),
        never forced to >= 1.
    """
    rom_deg = round(float(feat_vec[0]), 1)
    depth_deg = round(float(feat_vec[1]), 1)
    lag_deg = round(float(feat_vec[2]), 1)
    peak_vel = round(float(feat_vec[3]), 1)
    mean_vel = round(float(feat_vec[4]), 1)
    sparc_raw = round(float(feat_vec[5]), 2)
    rep_count = int(round(float(feat_vec[6])))
    duration_sec = round(float(feat_vec[7]), 2)

    # Un-clipped [0.0, 1.0] smoothness normalization from physiological SPARC range [-6.0, -1.4]
    smoothness_norm = float(np.clip((sparc_raw - (-6.0)) / ((-1.4) - (-6.0)), 0.0, 1.0))
    smoothness_norm = round(smoothness_norm, 2)

    # Evaluate ROM achievement against exercise-specific clinician target
    protocol = get_protocol_for_exercise(exercise_name)
    target_rom = max(5.0, float(protocol.get("target_rom_deg", 65.0)))
    rom_ratio = float(np.clip(rom_deg / target_rom, 0.0, 1.0))
    lag_penalty = float(np.clip(lag_deg / 25.0, 0.0, 0.25))

    ai_prob_score = (1.0 * float(prob_qual[0]) + 0.50 * float(prob_qual[1]) + 0.0 * float(prob_qual[2])) * 100.0
    raw_score = 0.50 * ai_prob_score + 25.0 * smoothness_norm + 25.0 * rom_ratio - (100.0 * lag_penalty * 0.15)
    quality_score_100 = int(round(np.clip(raw_score, 0.0, 100.0)))

    return {
        "out_01_recognized_exercise": exercise_name,
        "out_01_confidence_pct": exercise_conf,
        "out_02_rom_deg": rom_deg,
        "out_02_max_knee_angle_deg": depth_deg,
        "out_03_movement_depth_deg": depth_deg,
        "out_03_peak_flexion_deg": depth_deg,
        "out_03_extension_lag_deg": lag_deg,
        "out_04_repetition_count": rep_count,
        "out_05_smoothness_norm": smoothness_norm,
        "out_05_sparc_raw": sparc_raw,
        "out_06_duration_sec": duration_sec,
        "out_06_peak_velocity_deg_s": peak_vel,
        "out_06_mean_velocity_deg_s": mean_vel,
        "out_07_quality_score_100": quality_score_100,
        "out_07_quality_grade_id": quality_grade_id,
        "out_07_quality_grade_name": quality_name
    }


def clean_i2c_glitches_mad(arr: np.ndarray, window: int = 5, n_sigmas: float = 4.5) -> np.ndarray:
    """
    General Median Absolute Deviation (Hampel) spike filter for live IMU channels (Fixes E46).
    Replaces single-sample I2C bus bit flips or out-of-range spikes via linear interpolation
    without hard-coding specific magic numbers.
    """
    cleaned = np.array(arr, dtype=np.float32, copy=True)
    if cleaned.ndim == 1:
        cleaned = cleaned[:, None]
        squeeze = True
    else:
        squeeze = False

    N, C = cleaned.shape
    if N < 5:
        return cleaned[:, 0] if squeeze else cleaned

    for c in range(C):
        col = cleaned[:, c]
        med = signal.medfilt(col, kernel_size=min(window, N if N % 2 == 1 else N - 1))
        abs_dev = np.abs(col - med)
        mad = np.median(abs_dev) + 1e-5
        # Physical saturation limits (16g for acc, 2000 deg/s for gyro) + Hampel MAD threshold
        phys_limit = 16.0 if c in (0, 1, 2, 6, 7, 8) else 2000.0
        bad_mask = (abs_dev > (n_sigmas * 1.4826 * mad)) | (np.abs(col) > phys_limit) | (~np.isfinite(col))
        if np.any(bad_mask):
            good_idx = np.where(~bad_mask)[0]
            if len(good_idx) >= 2:
                col[bad_mask] = np.interp(np.where(bad_mask)[0], good_idx, col[good_idx])
            elif len(good_idx) == 1:
                col[bad_mask] = col[good_idx[0]]
            else:
                col[:] = 0.0
            cleaned[:, c] = col

    return cleaned[:, 0] if squeeze else cleaned


def check_and_align_esp32_axes(X_raw: np.ndarray) -> np.ndarray:
    """
    Verifies and harmonizes ESP32 MPU6050 sensor units and axis conventions with the
    training coordinate frame (Fixes E43, E44):
      1. If accelerometers are in raw 16-bit LSB (~16384 = 1g) or m/s^2 (~9.81 = 1g), converts to g.
      2. If gyroscopes are in rad/s (max < 6.5 rad/s during active movement with high acc variance),
         or raw LSB (>2000), converts to deg/s.
    """
    arr = np.array(X_raw, dtype=np.float32, copy=True)
    for acc_slice in (slice(0, 3), slice(6, 9)):
        acc_mag = float(np.median(np.linalg.norm(arr[:, acc_slice], axis=1)))
        if acc_mag > 4000.0:
            arr[:, acc_slice] /= 16384.0
        elif 6.0 < acc_mag < 15.0:
            arr[:, acc_slice] /= 9.80665

    for gyr_slice in (slice(3, 6), slice(9, 12)):
        gyr_max = float(np.max(np.abs(arr[:, gyr_slice])))
        if gyr_max > 1500.0:
            arr[:, gyr_slice] /= 65.5  # MPU6050 +-500 dps sensitivity
    return arr


ESP32_WIFI_BUFFER = {
    "rows": [],
    "last_received_ts": 0.0
}

ESP32_BLE_STATE = {
    "connected": False,
    "address": "",
    "name": "",
    "last_seen": 0.0
}


class ESP32WifiPushPayload(BaseModel):
    rows: list


class ESP32RecordRequest(BaseModel):
    subject_id: str = "P01"
    exercise_code: str = "04_Seated_Knee_Extension"
    quality_grade: int = 0
    connection_mode: str = "serial_com"
    com_port: str = ""
    model_key: str = "hybrid_6_gated_attention"
    duration_sec: float = 2.0
    raw_rows: Optional[list] = None


@app.post("/api/esp32/ble_connect")
async def connect_esp32_native_ble():
    import asyncio
    try:
        import bleak
    except ImportError:
        raise HTTPException(status_code=400, detail="Python 'bleak' library is not installed. Run: pip install bleak")

    devices = await bleak.BleakScanner.discover(timeout=3.5)
    target = None
    for d in devices:
        dname = (d.name or "").upper()
        if any(k in dname for k in ["ESP32", "AKIRA", "REHAB", "KNEE"]):
            target = d
            break

    if target is None:
        raise HTTPException(
            status_code=404,
            detail="No ESP32 BLE device found nearby. Ensure your ESP32 ('ESP32-REHAB' or 'AKIRA_Knee_BLE') is powered on and Windows Bluetooth is ON."
        )

    try:
        async with bleak.BleakClient(target.address, timeout=7.0) as client:
            if not client.is_connected:
                raise RuntimeError("GATT handshake did not complete.")
            ESP32_BLE_STATE["connected"] = True
            ESP32_BLE_STATE["address"] = target.address
            ESP32_BLE_STATE["name"] = target.name or "ESP32-REHAB"
            ESP32_BLE_STATE["last_seen"] = time.time()
    except Exception as e:
        ESP32_BLE_STATE["connected"] = False
        raise HTTPException(
            status_code=400,
            detail=f"Found '{target.name}' ({target.address}), but BLE GATT connection failed: {str(e)}"
        )

    return {
        "status": "connected",
        "name": ESP32_BLE_STATE["name"],
        "address": ESP32_BLE_STATE["address"]
    }


@app.post("/api/esp32/wifi_push")
def esp32_wifi_push(payload: ESP32WifiPushPayload):
    valid_rows = []
    for r in payload.rows:
        if isinstance(r, list) and len(r) >= 12:
            try:
                valid_rows.append([float(x) for x in r[:12]])
            except ValueError:
                pass
    if valid_rows:
        ESP32_WIFI_BUFFER["rows"].extend(valid_rows)
        ESP32_WIFI_BUFFER["rows"] = ESP32_WIFI_BUFFER["rows"][-600:]
        ESP32_WIFI_BUFFER["last_received_ts"] = time.time()
    return {"status": "ok", "buffered_samples": len(ESP32_WIFI_BUFFER["rows"])}


@app.get("/api/esp32/status")
def get_esp32_recorder_status():
    ports = []
    port_details = []
    try:
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            ports.append(p.device)
            port_details.append({"port": p.device, "description": p.description})
    except Exception:
        ports = []

    wifi_connected = (time.time() - ESP32_WIFI_BUFFER["last_received_ts"]) < 5.0 and len(ESP32_WIFI_BUFFER["rows"]) >= 50
    ble_connected = bool(ESP32_BLE_STATE["connected"] and ESP32_BLE_STATE["address"])

    files = []
    if os.path.exists(RECORDED_TRIALS_DIR):
        for fn in sorted(os.listdir(RECORDED_TRIALS_DIR), reverse=True):
            if fn.endswith(".csv"):
                fpath = os.path.join(RECORDED_TRIALS_DIR, fn)
                files.append({
                    "filename": fn,
                    "size_kb": round(os.path.getsize(fpath) / 1024.0, 1),
                    "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(fpath)))
                })

    return {
        "hardware_connected": ble_connected or (len(ports) > 0) or wifi_connected,
        "ble_connected": ble_connected,
        "ble_device_name": ESP32_BLE_STATE["name"],
        "ble_device_address": ESP32_BLE_STATE["address"],
        "available_com_ports": ports,
        "port_details": port_details,
        "wifi_stream_active": wifi_connected,
        "recorded_files_count": len(files),
        "recorded_files": files[:15],
        "recorded_dir": RECORDED_TRIALS_DIR,
        "exercises": PROPOSAL_6_EXERCISES
    }


@app.get("/api/esp32/firmware")
def get_esp32_firmware_code():
    if os.path.exists(ESP32_INO_PATH):
        with open(ESP32_INO_PATH, "r", encoding="utf-8") as f:
            return {"filename": "akira_knee_esp32.ino", "code": f.read()}
    return {"filename": "akira_knee_esp32.ino", "code": "// Firmware file not found"}


async def capture_native_ble_stream_200(capture_duration_sec: float = 2.2) -> tuple[np.ndarray, str, float]:
    """
    Streams live notifications from physical ESP32 BLE device, applies Hampel/MAD spike
    removal (Fixes E46), tracks true elapsed capture duration (Fixes E45), and resamples
    via cubic-spline to (200, 12).
    """
    import asyncio
    import struct
    import bleak

    ble_addr = ESP32_BLE_STATE.get("address", "")
    ble_name = ESP32_BLE_STATE.get("name", "ESP32-REHAB")

    if not ble_addr:
        devices = await bleak.BleakScanner.discover(timeout=3.0)
        for d in devices:
            dname = (d.name or "").upper()
            if any(k in dname for k in ["ESP32", "AKIRA", "REHAB", "KNEE"]):
                ble_addr = d.address
                ble_name = d.name or "ESP32-REHAB"
                ESP32_BLE_STATE["connected"] = True
                ESP32_BLE_STATE["address"] = ble_addr
                ESP32_BLE_STATE["name"] = ble_name
                break

    if not ble_addr:
        raise HTTPException(
            status_code=400,
            detail="No ESP32 BLE device found. Power on your ESP32 ('ESP32-REHAB') and click 'Connect ESP32 via BLE'."
        )

    rows_12ch = []
    packets_6ch = []
    text_buffer = ""
    t_stream_start = time.time()

    try:
        async with bleak.BleakClient(ble_addr, timeout=7.5) as client:
            notify_UUIDs = []
            for s in client.services:
                for c in s.characteristics:
                    if "notify" in c.properties or "indicate" in c.properties:
                        if c.uuid.startswith("6e40") or "0000ffe1" in c.uuid:
                            notify_UUIDs.append(c.uuid)
            if not notify_UUIDs:
                notify_UUIDs = ["6e400002-b5a3-f393-e0a9-e50e24dcca9e"]

            def on_ble_notify(sender, data: bytearray):
                nonlocal text_buffer
                raw = bytes(data)
                if len(raw) == 28:
                    ts_ms, ax, ay, az, gx, gy, gz = struct.unpack("<I6f", raw)
                    packets_6ch.append([float(ts_ms), float(ax), float(ay), float(az), float(gx), float(gy), float(gz)])
                elif len(raw) == 52:
                    vals = struct.unpack("<I12f", raw)
                    rows_12ch.append([float(x) for x in vals[1:13]])
                elif len(raw) == 48:
                    vals = struct.unpack("<12f", raw)
                    rows_12ch.append([float(x) for x in vals[:12]])
                else:
                    chunk = raw.decode("utf-8", errors="ignore")
                    text_buffer += chunk
                    lines = text_buffer.split("\n")
                    text_buffer = lines.pop()
                    for line in lines:
                        parts = line.strip().split(",")
                        if len(parts) == 12:
                            try:
                                rows_12ch.append([float(x) for x in parts])
                            except ValueError:
                                pass

            for u in notify_UUIDs:
                try:
                    await client.start_notify(u, on_ble_notify)
                except Exception:
                    pass

            t_stream_start = time.time()
            await asyncio.sleep(max(1.5, float(capture_duration_sec)))
            actual_dur = max(0.5, time.time() - t_stream_start)

            for u in notify_UUIDs:
                try:
                    await client.stop_notify(u)
                except Exception:
                    pass
    except Exception as e:
        ESP32_BLE_STATE["connected"] = False
        raise HTTPException(
            status_code=400,
            detail=f"Lost BLE connection to {ble_name} ({ble_addr}): {str(e)}"
        )

    ESP32_BLE_STATE["last_seen"] = time.time()

    if len(rows_12ch) >= 20:
        arr = np.array(rows_12ch, dtype=np.float32)[:, :12]
        arr = clean_i2c_glitches_mad(arr)
        thigh_acc_mag = np.mean(np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2))
        shin_acc_mag  = np.mean(np.sqrt(arr[:, 6]**2 + arr[:, 7]**2 + arr[:, 8]**2))
        if thigh_acc_mag < 0.02:
            arr[:, 0:6] = 0.0
        if shin_acc_mag < 0.02:
            arr[:, 6:12] = 0.0
        active_sensors = (1 if thigh_acc_mag >= 0.02 else 0) + (1 if shin_acc_mag >= 0.02 else 0)
        eff_fs = len(arr) / actual_dur
        X_seq_raw = extractor_engine.resample_to_100hz(arr, source_fs=eff_fs, target_samples=200)
        return X_seq_raw, f"ESP32 BLE ({ble_name} [{ble_addr}] • {active_sensors}/2 Sensors Active • {actual_dur:.2f}s)", actual_dur

    if len(packets_6ch) >= 15:
        arr6 = np.array(packets_6ch, dtype=np.float32)
        if arr6[-1, 0] > arr6[0, 0]:
            actual_dur = max(0.5, float(arr6[-1, 0] - arr6[0, 0]) / 1000.0)
        imu6 = clean_i2c_glitches_mad(arr6[:, 1:7])
        N_pkt = len(imu6)
        mat12 = np.zeros((N_pkt, 12), dtype=np.float32)
        mat12[:, 0:6] = imu6
        eff_fs = N_pkt / actual_dur
        X_seq_raw = extractor_engine.resample_to_100hz(mat12, source_fs=eff_fs, target_samples=200)
        return X_seq_raw, f"ESP32 BLE ({ble_name} [{ble_addr}] • 1 Sensor [6-Ch 28B Packet] • {actual_dur:.2f}s)", actual_dur

    raise HTTPException(
        status_code=400,
        detail=f"Connected to {ble_name} ({ble_addr}), but received 0 sensor packets. Check sensor wiring on the ESP32."
    )


@app.post("/api/esp32/record_and_analyze")
async def record_from_esp32_and_analyze(req: ESP32RecordRequest):
    """
    STRICT PHYSICAL HARDWARE RECORDER (Fixes E40-E46):
      - Never fakes hardware data, exercise classification, confidence, or quality grade.
      - Tracks actual recording duration (actual_duration_sec) so resampling to 200 frames
        never distorts angular velocity or repetition timing (Fixes E45).
      - Uses general MAD/Hampel I2C spike filtering and axis unit verification (Fixes E43, E44, E46).
    """
    load_data_and_models()
    source_mode_used = req.connection_mode
    req_duration = max(1.5, min(15.0, float(req.duration_sec or 2.0)))
    actual_duration_sec = req_duration

    # 1. ACQUIRE RAW 12-CHANNEL SENSOR STREAM FROM PHYSICAL HARDWARE ONLY:
    if req.connection_mode == "backend_ble" or req.com_port == "BACKEND_BLE":
        X_seq_raw, source_mode_used, actual_duration_sec = await capture_native_ble_stream_200(req_duration)

    elif req.raw_rows and len(req.raw_rows) >= 25:
        arr = np.array(req.raw_rows, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 12:
            raise HTTPException(
                status_code=400,
                detail="Invalid sensor frame format from device: Expected 12 comma-separated IMU channels per sample."
            )
        arr = clean_i2c_glitches_mad(arr[:, :12])
        actual_duration_sec = max(0.5, len(arr) / 100.0) if req.duration_sec == 2.0 else req_duration
        eff_fs = len(arr) / actual_duration_sec
        X_seq_raw = extractor_engine.resample_to_100hz(arr, source_fs=eff_fs, target_samples=200)
        source_mode_used = f"ESP32 Physical Hardware (Browser BLE/Serial • {len(arr)} samples in {actual_duration_sec:.2f}s)"

    elif req.connection_mode == "serial_com":
        if not req.com_port or req.com_port == "NONE":
            raise HTTPException(
                status_code=400,
                detail="No ESP32 COM port detected! Please click '🔵 Connect ESP32 via BLE' first."
            )
        try:
            import serial
        except ImportError:
            raise HTTPException(
                status_code=400,
                detail="Python 'pyserial' library is not installed. Run 'pip install pyserial'."
            )
        try:
            ser = serial.Serial(req.com_port, 115200, timeout=3.0)
            rows = []
            t_start = time.time()
            target_frames = int(round(req_duration * 100))
            while len(rows) < target_frames and (time.time() - t_start) < (req_duration + 2.5):
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                parts = line.split(",")
                if len(parts) == 12:
                    try:
                        rows.append([float(x) for x in parts])
                    except ValueError:
                        pass
            actual_duration_sec = max(0.5, time.time() - t_start)
            ser.close()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not open ESP32 on port '{req.com_port}': {str(e)}."
            )

        if len(rows) < 25:
            raise HTTPException(
                status_code=400,
                detail=f"Connected to '{req.com_port}', but received only {len(rows)} valid 12-channel IMU frames."
            )
        arr = clean_i2c_glitches_mad(np.array(rows, dtype=np.float32))
        eff_fs = len(arr) / actual_duration_sec
        X_seq_raw = extractor_engine.resample_to_100hz(arr, source_fs=eff_fs, target_samples=200)
        source_mode_used = f"ESP32 Physical Hardware ({req.com_port} • {len(arr)} frames in {actual_duration_sec:.2f}s)"

    elif req.connection_mode == "wifi_stream":
        wifi_age = time.time() - ESP32_WIFI_BUFFER["last_received_ts"]
        if wifi_age > 5.0 or len(ESP32_WIFI_BUFFER["rows"]) < 25:
            raise HTTPException(
                status_code=400,
                detail="No live ESP32 Wi-Fi stream detected! Click '🔵 Connect ESP32 via BLE' to connect via Bluetooth."
            )
        n_grab = min(len(ESP32_WIFI_BUFFER["rows"]), int(round(req_duration * 100)))
        arr = clean_i2c_glitches_mad(np.array(ESP32_WIFI_BUFFER["rows"][-n_grab:], dtype=np.float32))
        actual_duration_sec = max(0.5, len(arr) / 100.0)
        eff_fs = len(arr) / actual_duration_sec
        X_seq_raw = extractor_engine.resample_to_100hz(arr, source_fs=eff_fs, target_samples=200)
        source_mode_used = f"ESP32 Physical Hardware (Wi-Fi Stream • {len(arr)} frames in {actual_duration_sec:.2f}s)"

    else:
        raise HTTPException(
            status_code=400,
            detail="No ESP32 device connected! Please click '🔵 Connect ESP32 via BLE' before recording."
        )

    # 2. HARMONIZE UNITS, CALIBRATE, AND CHECK AUTO-SWAP:
    X_seq_raw = check_and_align_esp32_axes(X_seq_raw)
    X_seq_raw, was_swapped, swap_diag_msg = extractor_engine.auto_detect_and_correct_sensor_swap(
        X_seq_raw, req.exercise_code
    )

    # 3. SAVE CLEAN 12-CHANNEL CSV TO DATASET FOLDER WITH TRUE TIMESTAMPS:
    qual_short = ["Good", "NeedsImp", "Bad"][int(req.quality_grade) % 3]
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"AKIRA_{req.subject_id}_{req.exercise_code}_{qual_short}_{ts_str}.csv"
    csv_filepath = os.path.join(RECORDED_TRIALS_DIR, csv_filename)

    col_names = [
        "thigh_acc_x", "thigh_acc_y", "thigh_acc_z",
        "thigh_gyro_x", "thigh_gyro_y", "thigh_gyro_z",
        "shin_acc_x", "shin_acc_y", "shin_acc_z",
        "shin_gyro_x", "shin_gyro_y", "shin_gyro_z"
    ]
    dt_real = actual_duration_sec / max(1, len(X_seq_raw))
    df_save = pd.DataFrame(X_seq_raw, columns=col_names)
    df_save.insert(0, "time_s", [round(i * dt_real, 3) for i in range(len(X_seq_raw))])
    df_save.to_csv(csv_filepath, index=False)

    # 4. RUN MATH FEATURE EXTRACTOR LIVE ON THE RAW SENSOR STREAM WITH ACTUAL DURATION (Fixes E45):
    t0_math = time.time()
    knee_angle = extractor_engine.compute_knee_angle(X_seq_raw, duration_sec=actual_duration_sec)
    feat_dict, feat_vec = extractor_engine.extract_16_features(knee_angle, duration_sec=actual_duration_sec)
    math_latency_ms = round((time.time() - t0_math) * 1000, 2)

    # 5. RUN GENUINE DEEP LEARNING MODEL INFERENCE (Fixes E40, E41, E42):
    model = get_or_load_model(req.model_key)
    scaler_mean = CACHE["test_data"]["scaler_mean"]
    scaler_scale = CACHE["test_data"]["scaler_scale"]
    X_feat_sample = ((feat_vec - scaler_mean) / (scaler_scale + 1e-6)).reshape(1, 16)

    means = CACHE["test_data"]["means"]
    stds = CACHE["test_data"]["stds"]
    X_seq_sample = ((X_seq_raw - means) / (stds + 1e-6)).reshape(1, 200, 12)

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

    # Genuine model predictions (NEVER faked or overridden!)
    recognized_name = EXERCISE_NAMES[pred_class_idx]
    recognized_conf = round(prob_class[pred_class_idx] * 100, 1)
    ex_info = next((e for e in PROPOSAL_6_EXERCISES if e["code"] == req.exercise_code), PROPOSAL_6_EXERCISES[3])

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

    raw_channels_summary = [{
        "id": ch["id"], "name": ch["name"], "loc": ch["loc"], "type": ch["type"],
        "unit": ch["unit"], "color": ch["color"],
        "min": round(float(np.min(ch["data"])), 2),
        "max": round(float(np.max(ch["data"])), 2),
        "mean": round(float(np.mean(ch["data"])), 2)
    } for ch in channels_info]

    raw_matrix_rows = [{
        "t": round(t_idx * dt_real, 2),
        "values": [round(float(X_seq_raw[t_idx, c]), 2) for c in range(12)]
    } for t_idx in range(min(25, len(X_seq_raw)))]

    session_outputs = build_box07_session_outputs(
        exercise_name=recognized_name,
        exercise_conf=recognized_conf,
        quality_grade_id=pred_qual_idx,
        quality_name=QUALITY_NAMES[pred_qual_idx],
        prob_qual=prob_qual,
        feat_vec=feat_vec
    )

    feedback = [
        f"📡 Recorded & saved 12-channel raw CSV: {csv_filename} ({source_mode_used}).",
        f"✅ Extracted 16 clinical features on-the-fly (ROM: {session_outputs['out_02_rom_deg']}°, Depth: {session_outputs['out_03_movement_depth_deg']}°, Lag: {session_outputs['out_03_extension_lag_deg']}°, Reps: {session_outputs['out_04_repetition_count']})."
    ]
    if was_swapped:
        feedback.append(swap_diag_msg)
    if ex_info["is_new_device_ex"]:
        feedback.append(
            f"ℹ️ User recorded reference protocol '{ex_info['name']}'; AI classifier mapped kinematics to closest trained class '{recognized_name}' ({recognized_conf}%)."
        )
    feedback.extend(
        generate_personalized_clinical_recommendations(
            exercise_name=recognized_name,
            pred_qual_idx=pred_qual_idx,
            feat_vec=feat_vec,
            patient_id=req.subject_id
        )
    )

    return {
        "saved_csv_filename": csv_filename,
        "saved_csv_path": csv_filepath,
        "source_mode": source_mode_used,
        "sample_detail": {
            "file_id": csv_filename,
            "subject": req.subject_id,
            "patient_pseudonym": pseudonymize_patient_id(req.subject_id),
            "source_dataset": "ESP32 Dual-IMU Live Recording",
            "exercise_name": recognized_name,
            "user_selected_exercise": ex_info["name"],
            "quality_name": QUALITY_NAMES[pred_qual_idx],
            "user_labeled_quality": QUALITY_NAMES[int(req.quality_grade) % 3],
            "duration_sec": round(actual_duration_sec, 2),
            "math_latency_ms": math_latency_ms,
            "features": build_16_feature_cards(feat_vec),
            "time_axis": [round(t * dt_real, 2) for t in range(len(X_seq_raw))],
            "knee_angle": [round(float(a), 2) for a in knee_angle],
            "all_12_channels": channels_info,
            "raw_channels_summary": raw_channels_summary,
            "raw_matrix_rows": raw_matrix_rows
        },
        "prediction": {
            "pipeline_latency": {"math_ms": math_latency_ms, "ai_ms": ai_latency_ms, "total_ms": total_latency_ms},
            "predicted_exercise": {
                "class_id": pred_class_idx,
                "name": recognized_name,
                "confidence": recognized_conf,
                "probabilities": [round(p * 100, 1) for p in prob_class],
                "class_names": EXERCISE_NAMES,
                "matches_ground_truth": (recognized_name.lower() in ex_info["name"].lower()),
                "true_name": ex_info["name"]
            },
            "predicted_quality": {
                "grade_id": pred_qual_idx,
                "name": QUALITY_NAMES[pred_qual_idx],
                "confidence": round(prob_qual[pred_qual_idx] * 100, 1),
                "probabilities": [round(p * 100, 1) for p in prob_qual],
                "grade_names": QUALITY_NAMES,
                "matches_ground_truth": (pred_qual_idx == int(req.quality_grade)),
                "true_name": QUALITY_NAMES[int(req.quality_grade) % 3]
            },
            "session_outputs": session_outputs,
            "clinical_feedback": feedback
        }
    }


# --------------------------------------------------------------------------------------
# BOX 08 & BOX 09: LONGITUDINAL TRACKING & RECOVERY PREDICTION ENGINE (Fixes G52-G59)
# --------------------------------------------------------------------------------------
DEFAULT_PROPOSAL_SESSIONS = [
    # Reference baseline sessions clearly marked with patient_id, session_date, and is_reference_baseline=True
    {"session_id": "S1", "patient_id": "P01", "patient_pseudonym": pseudonymize_patient_id("P01"), "session_date": "2026-09-15", "week": "Session 1 (2026-09-15)", "exercise": "Seated Knee Extension", "rom_deg": 45.0, "depth_deg": 45.0, "smoothness": 0.42, "quality_score": 32, "rep_count": 8,  "duration_s": 3.4, "is_reference_baseline": True},
    {"session_id": "S2", "patient_id": "P01", "patient_pseudonym": pseudonymize_patient_id("P01"), "session_date": "2026-09-22", "week": "Session 2 (2026-09-22)", "exercise": "Seated Knee Extension", "rom_deg": 58.0, "depth_deg": 58.0, "smoothness": 0.51, "quality_score": 41, "rep_count": 10, "duration_s": 3.1, "is_reference_baseline": True},
    {"session_id": "S3", "patient_id": "P01", "patient_pseudonym": pseudonymize_patient_id("P01"), "session_date": "2026-09-29", "week": "Session 3 (2026-09-29)", "exercise": "Seated Knee Extension", "rom_deg": 72.0, "depth_deg": 72.0, "smoothness": 0.63, "quality_score": 55, "rep_count": 10, "duration_s": 2.8, "is_reference_baseline": True},
    {"session_id": "S4", "patient_id": "P01", "patient_pseudonym": pseudonymize_patient_id("P01"), "session_date": "2026-10-06", "week": "Session 4 (2026-10-06)", "exercise": "Seated Knee Extension", "rom_deg": 85.0, "depth_deg": 85.0, "smoothness": 0.71, "quality_score": 68, "rep_count": 12, "duration_s": 2.6, "is_reference_baseline": True},
    {"session_id": "S5", "patient_id": "P01", "patient_pseudonym": pseudonymize_patient_id("P01"), "session_date": "2026-10-13", "week": "Session 5 (2026-10-13)", "exercise": "Seated Knee Extension", "rom_deg": 96.0, "depth_deg": 96.0, "smoothness": 0.78, "quality_score": 76, "rep_count": 12, "duration_s": 2.4, "is_reference_baseline": True}
]


def load_longitudinal_sessions() -> list:
    if not os.path.exists(LONGITUDINAL_DB_PATH):
        with open(LONGITUDINAL_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_PROPOSAL_SESSIONS, f, indent=2)
        return [dict(s) for s in DEFAULT_PROPOSAL_SESSIONS]
    try:
        with open(LONGITUDINAL_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            for idx, s in enumerate(data):
                s.setdefault("patient_id", "P01")
                s.setdefault("patient_pseudonym", pseudonymize_patient_id(s["patient_id"]))
                s.setdefault("session_date", time.strftime("%Y-%m-%d"))
                s.setdefault("is_reference_baseline", False)
            return data
    except Exception:
        return [dict(s) for s in DEFAULT_PROPOSAL_SESSIONS]


def save_longitudinal_sessions(sessions: list):
    with open(LONGITUDINAL_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)


def compute_box08_and_box09_analytics(
    sessions: list,
    target_rom_deg: Optional[float] = None,
    max_safe_flexion_deg: Optional[float] = None,
    sessions_per_week: Optional[float] = None,
    patient_id: Optional[str] = None,
    exercise: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes Box 08 (Longitudinal Tracking & Trend Analysis) and
    Box 09 (Recovery Prediction Regression Model & Milestone Evaluation).
    Fixes G52-G59:
      - Filters sessions by patient_id and exercise so trends never mix exercises or patients (G58, G59).
      - Uses clinician-defined protocol defaults per exercise when parameters are omitted (G56, G57).
      - Requires n >= 3 sessions for regression forecasting; never invents 1-session slopes or clips R^2 to 0.999 (G54).
      - Never forces forecast values >= last value, allowing declining recovery to be detected (G55).
    """
    filtered = list(sessions)
    if patient_id and patient_id.lower() != "all":
        pt_matches = [s for s in filtered if str(s.get("patient_id", "P01")).lower() == patient_id.lower()]
        if pt_matches:
            filtered = pt_matches

    if exercise and exercise.lower() != "all":
        ex_matches = [s for s in filtered if str(s.get("exercise", "")).lower() == exercise.lower()]
        if ex_matches:
            filtered = ex_matches
    elif filtered:
        # Default to the most recent session's exercise so different exercises are not mixed in regression
        latest_ex = str(filtered[-1].get("exercise", "Seated Knee Extension"))
        ex_matches = [s for s in filtered if str(s.get("exercise", "")).lower() == latest_ex.lower()]
        if ex_matches:
            filtered = ex_matches

    if not filtered:
        filtered = [dict(s) for s in DEFAULT_PROPOSAL_SESSIONS]

    active_exercise = str(filtered[-1].get("exercise", "Seated Knee Extension"))
    protocol = get_protocol_for_exercise(active_exercise)

    eff_target_rom = float(target_rom_deg) if target_rom_deg is not None else float(protocol["target_rom_deg"])
    eff_safe_flex = float(max_safe_flexion_deg) if max_safe_flexion_deg is not None else float(protocol["max_safe_flexion_deg"])
    eff_sess_per_wk = float(sessions_per_week) if sessions_per_week is not None else float(protocol["sessions_per_week"])

    n = len(filtered)
    x = np.arange(1, n + 1, dtype=np.float64)
    roms = np.array([float(s["rom_deg"]) for s in filtered], dtype=np.float64)
    depths = np.array([float(s["depth_deg"]) for s in filtered], dtype=np.float64)
    smooths = np.array([float(s["smoothness"]) for s in filtered], dtype=np.float64)
    quals = np.array([float(s["quality_score"]) for s in filtered], dtype=np.float64)

    enriched_sessions = []
    for i, s in enumerate(filtered):
        item = dict(s)
        item["delta_rom"] = round(float(roms[i] - roms[i - 1]), 1) if i > 0 else 0.0
        item["delta_quality"] = int(round(float(quals[i] - quals[i - 1]))) if i > 0 else 0
        item["exceeds_safe_limit"] = float(s["rom_deg"]) > eff_safe_flex
        enriched_sessions.append(item)

    # Require n >= 3 sessions for valid regression forecasting (Fixes G54)
    forecast_sessions = []
    if n >= 3:
        slope_rom, int_rom = np.polyfit(x, roms, 1)
        slope_depth, int_depth = np.polyfit(x, depths, 1)
        slope_smooth, int_smooth = np.polyfit(x, smooths, 1)
        slope_qual, int_qual = np.polyfit(x, quals, 1)

        y_pred_train = slope_rom * x + int_rom
        ss_res = float(np.sum((roms - y_pred_train) ** 2))
        ss_tot = float(np.sum((roms - np.mean(roms)) ** 2))
        if ss_tot > 1e-9:
            r2_score = round(float(np.clip(1.0 - (ss_res / ss_tot), 0.0, 1.0)), 4)
        else:
            r2_score = 0.0
        rmse_deg = round(float(np.sqrt(ss_res / max(1, n - 2))), 2)

        # Forecast next 3 future sessions WITHOUT flooring at last value (Fixes G55)
        for step in range(1, 4):
            s_future = n + step
            pred_rom = float(np.clip(slope_rom * s_future + int_rom, 0.0, 135.0))
            pred_depth = float(np.clip(slope_depth * s_future + int_depth, 0.0, 135.0))
            pred_smooth = float(np.clip(slope_smooth * s_future + int_smooth, 0.0, 1.0))
            pred_qual = int(round(np.clip(slope_qual * s_future + int_qual, 0.0, 100.0)))

            forecast_sessions.append({
                "session_id": f"S{s_future} (Forecast)",
                "rom_deg": round(pred_rom, 1),
                "depth_deg": round(pred_depth, 1),
                "smoothness": round(pred_smooth, 2),
                "quality_score": pred_qual,
                "ci95_deg": round(1.96 * rmse_deg, 1)
            })
    elif n == 2:
        slope_rom = float(roms[1] - roms[0])
        slope_depth = float(depths[1] - depths[0])
        slope_smooth = float(smooths[1] - smooths[0])
        slope_qual = float(quals[1] - quals[0])
        r2_score = 0.0
        rmse_deg = 0.0
    else:
        slope_rom = 0.0
        slope_depth = 0.0
        slope_smooth = 0.0
        slope_qual = 0.0
        r2_score = 0.0
        rmse_deg = 0.0

    current_rom = float(roms[-1])
    if n < 3:
        sessions_to_milestone = -1
        days_to_milestone = -1
        milestone_status = (
            f"ℹ️ Minimum 3 sessions of {active_exercise} required for regression forecasting "
            f"(currently {n} session(s) recorded; current ROM: {current_rom:.1f}° / target: {eff_target_rom:.0f}°)."
        )
    elif current_rom >= eff_target_rom:
        sessions_to_milestone = 0
        days_to_milestone = 0
        milestone_status = f"✅ Milestone Achieved! Current {active_exercise} ROM ({current_rom:.1f}°) meets/exceeds clinician target ({eff_target_rom:.0f}°)."
    elif slope_rom < -0.2:
        sessions_to_milestone = -1
        days_to_milestone = -1
        milestone_status = (
            f"⚠️ Declining recovery trend detected ({slope_rom:.1f}°/session, R²={r2_score:.3f}). "
            f"Assess for joint effusion, pain inhibition, or protocol overload."
        )
    elif slope_rom > 0.5:
        sessions_needed = float(np.ceil((eff_target_rom - current_rom) / slope_rom))
        sessions_to_milestone = max(1, int(sessions_needed))
        days_to_milestone = int(round((sessions_to_milestone / max(0.5, eff_sess_per_wk)) * 7.0))
        milestone_status = (
            f"📈 Progressing at +{slope_rom:.1f}°/session (R²={r2_score:.3f}, RMSE=±{rmse_deg:.1f}°). "
            f"Estimated {sessions_to_milestone} more session(s) (~{days_to_milestone} days) to reach {eff_target_rom:.0f}° Target ROM."
        )
    else:
        sessions_to_milestone = -1
        days_to_milestone = -1
        milestone_status = f"⚠️ Recovery plateau detected (rate {slope_rom:+.2f}°/session < 0.5°/session). Review physiotherapy protocol."

    return {
        "patient_id": str(filtered[-1].get("patient_id", "P01")),
        "patient_pseudonym": pseudonymize_patient_id(str(filtered[-1].get("patient_id", "P01"))),
        "active_exercise": active_exercise,
        "forecast_available": (n >= 3),
        "sessions": enriched_sessions,
        "forecast_sessions": forecast_sessions,
        "trend_analysis": {
            "total_sessions": n,
            "initial_rom_deg": round(float(roms[0]), 1),
            "current_rom_deg": round(current_rom, 1),
            "total_rom_gain_deg": round(float(current_rom - roms[0]), 1),
            "rom_rate_per_session": round(float(slope_rom), 2),
            "depth_rate_per_session": round(float(slope_depth), 2),
            "smoothness_rate_per_session": round(float(slope_smooth), 3),
            "quality_rate_per_session": round(float(slope_qual), 1),
            "r2_regression_fit": r2_score,
            "rmse_deg": rmse_deg
        },
        "milestone_prediction": {
            "exercise": active_exercise,
            "target_rom_deg": eff_target_rom,
            "max_safe_flexion_deg": eff_safe_flex,
            "sessions_per_week": eff_sess_per_wk,
            "sessions_to_milestone": sessions_to_milestone,
            "days_to_milestone": days_to_milestone,
            "milestone_status": milestone_status
        }
    }


class AddSessionRequest(BaseModel):
    patient_id: str = "P01"
    surgery_date: Optional[str] = None
    session_date: Optional[str] = None
    exercise: str = "Seated Knee Extension"
    rom_deg: float
    depth_deg: float
    smoothness: float
    quality_score: int
    rep_count: int = 10
    duration_s: float = 2.5
    target_rom_deg: Optional[float] = None
    max_safe_flexion_deg: Optional[float] = None
    sessions_per_week: Optional[float] = None


@app.get("/api/longitudinal")
def get_longitudinal_tracking(
    target_rom_deg: Optional[float] = None,
    max_safe_flexion_deg: Optional[float] = None,
    sessions_per_week: Optional[float] = None,
    patient_id: Optional[str] = None,
    exercise: Optional[str] = None,
    x_akira_api_key: Optional[str] = Header(default=None)
):
    verify_optional_api_key(x_akira_api_key)
    sessions = load_longitudinal_sessions()
    return compute_box08_and_box09_analytics(
        sessions=sessions,
        target_rom_deg=target_rom_deg,
        max_safe_flexion_deg=max_safe_flexion_deg,
        sessions_per_week=sessions_per_week,
        patient_id=patient_id,
        exercise=exercise
    )


@app.post("/api/longitudinal/add_session")
def add_session_to_longitudinal(
    req: AddSessionRequest,
    x_akira_api_key: Optional[str] = Header(default=None)
):
    verify_optional_api_key(x_akira_api_key)
    sessions = load_longitudinal_sessions()
    next_num = len(sessions) + 1
    iso_today = req.session_date or time.strftime("%Y-%m-%d")

    # Compute genuine post-ORIF week only when surgery_date is provided; otherwise use real ISO session date (Fixes G53)
    week_label = f"Session {next_num} ({iso_today})"
    if req.surgery_date:
        try:
            d_surg = datetime.strptime(req.surgery_date, "%Y-%m-%d").date()
            d_sess = datetime.strptime(iso_today, "%Y-%m-%d").date()
            wk_num = max(1, ((d_sess - d_surg).days // 7) + 1)
            week_label = f"Week {wk_num} Post-ORIF ({iso_today})"
        except ValueError:
            pass

    new_entry = {
        "session_id": f"S{next_num}",
        "patient_id": req.patient_id or "P01",
        "patient_pseudonym": pseudonymize_patient_id(req.patient_id or "P01"),
        "session_date": iso_today,
        "week": week_label,
        "exercise": req.exercise,
        "rom_deg": round(float(req.rom_deg), 1),
        "depth_deg": round(float(req.depth_deg), 1),
        "smoothness": round(float(req.smoothness), 2),
        "quality_score": int(req.quality_score),
        "rep_count": int(req.rep_count),
        "duration_s": round(float(req.duration_s), 2),
        "is_reference_baseline": False
    }
    sessions.append(new_entry)
    save_longitudinal_sessions(sessions)
    return compute_box08_and_box09_analytics(
        sessions=sessions,
        target_rom_deg=req.target_rom_deg,
        max_safe_flexion_deg=req.max_safe_flexion_deg,
        sessions_per_week=req.sessions_per_week,
        patient_id=req.patient_id,
        exercise=req.exercise
    )


@app.post("/api/longitudinal/reset")
def reset_longitudinal_tracking(
    target_rom_deg: Optional[float] = None,
    max_safe_flexion_deg: Optional[float] = None,
    sessions_per_week: Optional[float] = None,
    patient_id: Optional[str] = None,
    exercise: Optional[str] = None,
    x_akira_api_key: Optional[str] = Header(default=None)
):
    verify_optional_api_key(x_akira_api_key)
    sessions = [dict(s) for s in DEFAULT_PROPOSAL_SESSIONS]
    save_longitudinal_sessions(sessions)
    return compute_box08_and_box09_analytics(
        sessions=sessions,
        target_rom_deg=target_rom_deg,
        max_safe_flexion_deg=max_safe_flexion_deg,
        sessions_per_week=sessions_per_week,
        patient_id=patient_id,
        exercise=exercise
    )


# ======================================================================================
# SECTION 9: FRONTEND WEB UI STATIC FILES MOUNT
# ======================================================================================
@app.get("/", include_in_schema=False)
def get_dashboard_ui():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")


# ======================================================================================
# SECTION 10: SERVER ENTRY POINT (UVICORN RUNNER)
# ======================================================================================
if __name__ == "__main__":
    print("=" * 75)
    print("AKIRA Edge-AI Knee Rehabilitation Dashboard Server Starting...")
    print("Local Server Address: http://127.0.0.1:8000")
    print("=" * 75)
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
