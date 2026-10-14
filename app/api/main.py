"""
========================================================================================
AKIRA KNEE REHABILITATION SYSTEM: BACKEND REST API SERVER (main.py)
========================================================================================
WHAT THIS FILE IS:
  This is the primary FastAPI backend web server for the AKIRA Knee Rehabilitation
  ecosystem. It connects the frontend web dashboard with the edge-AI deep learning
  models and mathematical kinematic feature extraction engine.

WHAT THIS FILE DOES:
  1. Serves the interactive web dashboard (HTML, CSS, JS, Chart.js) to the browser.
  2. Provides REST API endpoints for:
     - Listing and comparing 6 Dual-Stream Hybrid Deep Learning models.
     - Loading and inspecting test samples from clinical datasets (12 IMU channels).
     - Live on-the-fly extraction of 16 orthopedic kinematic features (SPARC, ROM, etc.).
     - Running dual-head neural network inference (Exercise Classification + Quality Grading).
     - Generating automated orthopedic clinical feedback (e.g., extensor lag, squat depth).
  3. Employs an in-memory RAM cache for ultra-fast response times (<15ms latency).

TABLE OF CONTENTS:
  SECTION 1: Imports and Dependencies
  SECTION 2: Directory Paths and File Configuration
  SECTION 3: FastAPI Application & CORS Configuration
  SECTION 4: Global In-Memory Cache & Reference Dictionaries
  SECTION 5: Mathematical Kinematic Signal Processing (Knee Angle Trajectory)
  SECTION 6: Data & Model Loading Procedures (Cache Initializers)
  SECTION 7: Pydantic Data Schemas (API Request Models)
  SECTION 8: REST API Endpoints:
             - GET  /api/models           (List 6 Hybrid Architectures)
             - GET  /api/metrics/{model}  (Confusion Matrix & Performance Stats)
             - GET  /api/all_metrics      (Cross-Model Comparison Table)
             - GET  /api/test_samples     (Filterable 142 Held-out Test Trials)
             - GET  /api/sample/{idx}     (Full 12-channel Waveforms & Live Math)
             - POST /api/predict          (End-to-End Real-Time AI Inference)
  SECTION 9: Frontend Web UI Static Files Mount
  SECTION 10: Server Entry Point (Uvicorn runner)
========================================================================================
"""

# ======================================================================================
# SECTION 1: IMPORTS AND DEPENDENCIES
# ======================================================================================
import os
import sys
import json
import time
from typing import Dict, Any, Optional

# Numerical computation, signal processing, and dataframes
import numpy as np
import pandas as pd
from scipy import signal

# Pydantic for API data validation
from pydantic import BaseModel

# TensorFlow & Keras for Deep Learning model loading and prediction
import tensorflow as tf
from tensorflow import keras

# FastAPI & Uvicorn for asynchronous high-performance REST API handling
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# AKIRA Domain Module: Mathematical 16-Clinical-Feature Extractor Engine
from math_feature_extractor import Clinical16FeatureExtractor

# Instantiate the mathematical kinematic feature extractor engine at 100 Hz sampling rate
extractor_engine = Clinical16FeatureExtractor(fs=100.0)


# ======================================================================================
# SECTION 2: DIRECTORY PATHS AND FILE CONFIGURATION
# ======================================================================================
# Base path of the API package (absolute path to current directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder containing trained Keras deep learning models (.keras files and metrics)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Folder containing preprocessed clinical datasets and catalogs
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

# Folder containing static frontend UI assets (index.html, CSS styles, JS scripts)
WEB_DIR = os.path.join(os.path.dirname(BASE_DIR), "web")

# Paths to the 20% held-out test dataset files (used for validation and testing)
TEST_NPZ_PATH = os.path.join(DATASETS_DIR, "test_20percent_dataset.npz")
TEST_CATALOG_PATH = os.path.join(DATASETS_DIR, "test_20percent_catalog.csv")
METRICS_JSON_PATH = os.path.join(MODELS_DIR, "test_metrics_summary.json")


# ======================================================================================
# SECTION 3: FASTAPI APPLICATION & CORS CONFIGURATION
# ======================================================================================
# Initialize the FastAPI application instance
app = FastAPI(
    title="AKIRA Knee Rehabilitation AI Web Interface",
    description="Edge-AI and Kinematic Feature Extraction API for Post-ORIF Knee Rehabilitation",
    version="1.0.0"
)

# Enable Cross-Origin Resource Sharing (CORS) to allow requests from any frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allows all origins (e.g. localhost, local network devices)
    allow_credentials=True,
    allow_methods=["*"],       # Allows all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],       # Allows all HTTP request headers
)


# ======================================================================================
# SECTION 4: GLOBAL IN-MEMORY CACHE & REFERENCE DICTIONARIES
# ======================================================================================
# Global memory cache storing test sets, catalogs, pre-loaded models, and pre-computed data
# This prevents repeated slow disk I/O reads on every single HTTP request.
CACHE = {
    "test_data": None,        # Contains raw sequences, feature baselines, normalization stats
    "test_catalog": None,     # Pandas DataFrame containing trial metadata (subjects, quality)
    "metrics": None,          # Dictionary of cross-validation benchmark metrics for all models
    "loaded_models": {},      # Dictionary caching loaded Keras model objects by model key
    "knee_angles": None       # Cached continuous knee angle curves for the test set
}

# Exercise label mapping (Head 1 Output - 4 Classes)
EXERCISE_NAMES = [
    "Walking",                  # Class 0: Continuous gait cycles
    "Seated Knee Extension",    # Class 1: Open-chain terminal quadriceps extension
    "Mini Squat",               # Class 2: Closed-chain partial weight-bearing squat (0°-45°)
    "Step-Up"                   # Class 3: Functional stair negotiation and unilateral loading
]

# Quality grade mapping (Head 2 Output - 3 Clinical Tiers)
QUALITY_NAMES = [
    "Good (Correct Form)",                  # Grade 0: Smooth trajectory, full ROM, no lag
    "Needs Improvement (Lag/Deficit)",     # Grade 1: Mild extensor lag or minor compensatory hesitation
    "Bad / Compensatory Fault"             # Grade 2: Jerky movement, trunk compensation, or severe lag
]


# ======================================================================================
# SECTION 5: MATHEMATICAL KINEMATIC SIGNAL PROCESSING (KNEE ANGLE TRAJECTORY)
# ======================================================================================
def compute_knee_angle_trajectory(X_seq_sample: np.ndarray, fs: float = 100.0, dt: float = 0.01) -> np.ndarray:
    """
    WHAT THIS DOES:
      Computes the continuous sagittal knee flexion/extension angle (in degrees)
      directly from raw 12-channel dual-IMU sensor readings.
      
    CHANNEL MAPPING:
      - Channels 0, 1, 2: Thigh Accelerometer (X, Y, Z in g)
      - Channels 3, 4, 5: Thigh Gyroscope (X, Y, Z in deg/s)
      - Channels 6, 7, 8: Shin Accelerometer (X, Y, Z in g)
      - Channels 9, 10, 11: Shin Gyroscope (X, Y, Z in deg/s)

    MATHEMATICAL PRINCIPLE:
      1. Differential Angular Velocity:
         omega_knee(t) = Shin_Gyro(t) - Thigh_Gyro(t)
      2. Numerical Integration:
         theta_raw(t) = Integral(omega_knee(t) * dt)
      3. High-Pass Filtering:
         Removes low-frequency sensor baseline integration drift.
      4. Calibration:
         Offsets minimum baseline to 0.0° (full anatomical extension).

    PARAMETERS:
      X_seq_sample (np.ndarray): Shape (200, 12) representing 2.0s of dual-IMU stream at 100 Hz.
      fs (float): Sampling frequency in Hertz (default 100.0 Hz).
      dt (float): Sampling period in seconds (1/fs = 0.01 s).

    RETURNS:
      knee_angle (np.ndarray): 1D array of 200 float values representing knee flexion angle in degrees.
    """
    # Extract Thigh Gyroscope (Channels 3, 4, 5) and Shin Gyroscope (Channels 9, 10, 11)
    thigh_gyr = X_seq_sample[:, 3:6]
    shin_gyr = X_seq_sample[:, 9:12]

    # Calculate relative rotational velocity between shin and thigh segments
    rel_gyr = shin_gyr - thigh_gyr

    # Automatically identify the dominant flexion-extension rotational axis
    dom_axis = int(np.argmax(np.std(rel_gyr, axis=0)))
    omega_knee = rel_gyr[:, dom_axis]

    # Integrate angular velocity over time to obtain raw angle
    raw_angle = np.cumsum(omega_knee) * dt

    # Zero-phase 1st-order high-pass Butterworth filter (0.05 Hz cutoff) to eliminate drift
    b_hp, a_hp = signal.butter(1, 0.05 / (0.5 * fs), btype='high')
    knee_angle = signal.filtfilt(b_hp, a_hp, raw_angle)

    # Offset reference so 0 degrees equals anatomical full extension
    knee_angle = knee_angle - np.min(knee_angle)
    return knee_angle


# ======================================================================================
# SECTION 6: DATA & MODEL LOADING PROCEDURES (CACHE INITIALIZERS)
# ======================================================================================
def load_data_and_models():
    """
    WHAT THIS DOES:
      Populates the global in-memory CACHE dictionary with:
      - The 20% held-out test dataset (.npz file containing raw sequences and labels).
      - The test catalog CSV (metadata containing subjects, trial numbers, file IDs).
      - Pre-computed benchmark evaluation metrics for all 6 hybrid models (.json).
      - Pre-calculated continuous knee angle trajectories.
    """
    # 1. Load test dataset arrays
    if CACHE["test_data"] is None and os.path.exists(TEST_NPZ_PATH):
        CACHE["test_data"] = np.load(TEST_NPZ_PATH)
        print(f"[CACHE] Loaded test dataset: {CACHE['test_data']['X_seq_raw'].shape[0]} samples.")

    # 2. Load test catalog CSV
    if CACHE["test_catalog"] is None and os.path.exists(TEST_CATALOG_PATH):
        CACHE["test_catalog"] = pd.read_csv(TEST_CATALOG_PATH)
        print(f"[CACHE] Loaded test catalog: {len(CACHE['test_catalog'])} rows.")

    # 3. Load benchmark evaluation metrics JSON
    if CACHE["metrics"] is None and os.path.exists(METRICS_JSON_PATH):
        with open(METRICS_JSON_PATH, "r") as f:
            CACHE["metrics"] = json.load(f)
        print(f"[CACHE] Loaded benchmark metrics for {len(CACHE['metrics'])} models.")

    # 4. Pre-compute knee angle curves across test trials for fast charting
    if CACHE["knee_angles"] is None and CACHE["test_data"] is not None:
        X_seq = CACHE["test_data"]["X_seq_raw"]
        angles = []
        for i in range(len(X_seq)):
            angles.append(compute_knee_angle_trajectory(X_seq[i]))
        CACHE["knee_angles"] = np.array(angles)


def get_or_load_model(model_key: str) -> keras.Model:
    """
    WHAT THIS DOES:
      Retrieves a trained Keras model from the in-memory cache, or loads it from disk
      if it hasn't been loaded into memory yet (lazy-loading pattern).

    PARAMETERS:
      model_key (str): Identifier key for the model (e.g. 'hybrid_1_cnn', 'hybrid_2_bilstm').

    RETURNS:
      keras.Model: Loaded TensorFlow/Keras functional model ready for inference.
    """
    # Check if the model is already residing in RAM
    if model_key in CACHE["loaded_models"]:
        return CACHE["loaded_models"][model_key]

    # Resolve filename on disk
    model_filename = f"{model_key}.keras"
    model_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model file not found on disk: {model_filename}")

    # Load model and cache it in RAM
    print(f"[AI ENGINE] Loading model '{model_key}' into memory...")
    model = keras.models.load_model(model_path)
    CACHE["loaded_models"][model_key] = model
    return model


# Trigger dataset and metrics loading immediately upon FastAPI application startup
@app.on_event("startup")
def startup_event():
    """Initializes the backend cache when the server boots."""
    load_data_and_models()


# ======================================================================================
# SECTION 7: PYDANTIC DATA SCHEMAS (API REQUEST MODELS)
# ======================================================================================
class PredictRequest(BaseModel):
    """
    WHAT THIS IS:
      Pydantic schema defining the expected JSON payload for the /api/predict endpoint.
      
    FIELDS:
      model_key (str): The AI model architecture to use (e.g., 'hybrid_2_bilstm').
      sample_index (int): Index of the test sample in the held-out test catalog (0 to 141).
    """
    model_key: str
    sample_index: int


# ======================================================================================
# SECTION 8: REST API ENDPOINTS
# ======================================================================================

# --------------------------------------------------------------------------------------
# ENDPOINT 1: GET /api/models - List All Available Deep Learning Models
# --------------------------------------------------------------------------------------
@app.get("/api/models")
def get_models_list():
    """
    WHAT THIS DOES:
      Returns metadata for all 6 Dual-Stream Hybrid Deep Learning models,
      including architectural summaries, parameter counts, latencies, and edge suitability.
    """
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


# --------------------------------------------------------------------------------------
# ENDPOINT 2: GET /api/metrics/{model_key} - Model Benchmark Performance Metrics
# --------------------------------------------------------------------------------------
@app.get("/api/metrics/{model_key}")
def get_model_metrics(model_key: str):
    """
    WHAT THIS DOES:
      Returns cross-validation evaluation metrics for a specific model on the 20%
      held-out test set, including Accuracy, F1-Scores, and 2D Confusion Matrices.
    """
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


# --------------------------------------------------------------------------------------
# ENDPOINT 3: GET /api/all_metrics - Compare All 6 Models
# --------------------------------------------------------------------------------------
@app.get("/api/all_metrics")
def get_all_metrics():
    """
    WHAT THIS DOES:
      Returns comparative summary performance data across all 6 models,
      sorted by combined accuracy score descending (for leaderboard table display).
    """
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
    # Sort leaderboard by highest combined performance score
    summary.sort(key=lambda x: x["combined_score"], reverse=True)
    return summary


# --------------------------------------------------------------------------------------
# ENDPOINT 4: GET /api/test_samples - Browse and Filter Test Set Trials
# --------------------------------------------------------------------------------------
@app.get("/api/test_samples")
def get_test_samples(exercise: Optional[str] = None, quality: Optional[int] = None):
    """
    WHAT THIS DOES:
      Returns the metadata list of 142 held-out test trials with optional filtering
      by exercise type and movement quality tier.
    """
    load_data_and_models()
    df = CACHE["test_catalog"]
    if df is None:
        raise HTTPException(status_code=500, detail="Test catalog not loaded.")

    samples = []
    for idx, row in df.iterrows():
        # Filter by exercise name if specified
        if exercise and exercise.lower() != "all" and row["exercise_name"].lower() != exercise.lower():
            continue
        # Filter by quality grade if specified
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


# --------------------------------------------------------------------------------------
# ENDPOINT 5: GET /api/sample/{sample_idx} - Deep-Dive Waveforms & Live Math Features
# --------------------------------------------------------------------------------------
@app.get("/api/sample/{sample_idx}")
def get_sample_detail(sample_idx: int):
    """
    WHAT THIS DOES:
      Provides comprehensive details for a specific trial:
      1. Takes raw 12-channel IMU time series (200 time steps @ 100 Hz).
      2. Dynamically runs the Mathematical Physics Model on-the-fly to derive knee angle
         and all 16 clinical features (ROM, Depth, Extensor Lag, SPARC Smoothness, etc.).
      3. Returns time-series waveform arrays formatted for frontend Chart.js rendering.
    """
    load_data_and_models()
    if CACHE["test_data"] is None or CACHE["test_catalog"] is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")

    N = len(CACHE["test_catalog"])
    if sample_idx < 0 or sample_idx >= N:
        raise HTTPException(status_code=400, detail=f"Index must be between 0 and {N-1}")

    row = CACHE["test_catalog"].iloc[sample_idx]
    X_seq_raw = CACHE["test_data"]["X_seq_raw"][sample_idx]     # (200, 12) pure raw sensor stream

    # 1. RUN MATHEMATICAL ENGINE LIVE ON THE FLY:
    t0_math = time.time()
    thigh_gyro_y = X_seq_raw[:, 4]   # Thigh Gyro Y (Channel 4)
    shin_gyro_y  = X_seq_raw[:, 10]  # Shin Gyro Y (Channel 10)
    knee_angle = extractor_engine.compute_knee_angle(thigh_gyro_y, shin_gyro_y)
    feat_dict, feat_vec = extractor_engine.extract_16_features(knee_angle)
    math_latency_ms = round((time.time() - t0_math) * 1000, 2)

    # 2. Package all 16 clinical features with units and clinical explanations:
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

    # 3. Format all 12 raw channels with their anatomical sensor locations and units:
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

    # Summary statistics for each of the 12 channels (Min, Max, Mean)
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

    # Sample raw numerical matrix rows (first 25 time steps) for data table inspection
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


# --------------------------------------------------------------------------------------
# ENDPOINT 6: POST /api/predict - End-to-End Real-Time AI Inference
# --------------------------------------------------------------------------------------
@app.post("/api/predict")
def run_model_inference(req: PredictRequest):
    """
    WHAT THIS DOES:
      Executes a complete, live end-to-end inference pass for a single trial:
      1. Retrieves raw 12-channel IMU sequence (Input A: 200 time steps @ 100 Hz).
      2. Computes the 16 clinical biomechanical features live on the fly (Input B).
      3. Normalizes both streams using baseline population statistics.
      4. Feeds both streams into the selected Dual-Stream Hybrid Deep Learning model.
      5. Obtains dual predictions:
         - Exercise Class (Walking, Extension, Squat, Step-Up)
         - Movement Quality Tier (Good, Needs Improvement, Bad)
      6. Applies orthopedic clinical rules to provide actionable rehabilitation guidance.
    """
    load_data_and_models()
    sample_idx = req.sample_index
    model_key = req.model_key

    if CACHE["test_data"] is None or CACHE["test_catalog"] is None:
        raise HTTPException(status_code=500, detail="Data not loaded.")

    N = len(CACHE["test_catalog"])
    if sample_idx < 0 or sample_idx >= N:
        raise HTTPException(status_code=400, detail=f"Index must be between 0 and {N-1}")

    model = get_or_load_model(model_key)

    # 1. Grab raw 12-channel IMU sequence:
    X_seq_raw = CACHE["test_data"]["X_seq_raw"][sample_idx]

    # 2. RUN MATH MODEL ON RAW SENSOR DATA TO EXTRACT 16 CLINICAL FEATURES ON-THE-FLY:
    t0_math = time.time()
    thigh_gyro_y = X_seq_raw[:, 4]
    shin_gyro_y  = X_seq_raw[:, 10]
    knee_angle = extractor_engine.compute_knee_angle(thigh_gyro_y, shin_gyro_y)
    feat_dict, feat_vec = extractor_engine.extract_16_features(knee_angle)
    math_latency_ms = round((time.time() - t0_math) * 1000, 2)

    # 3. NORMALIZE BOTH STREAMS ON-THE-FLY:
    # Standardize Input B (16 clinical features) using training scaler parameters
    scaler_mean = CACHE["test_data"]["scaler_mean"]
    scaler_scale = CACHE["test_data"]["scaler_scale"]
    X_feat_sample = ((feat_vec - scaler_mean) / (scaler_scale + 1e-6)).reshape(1, 16)

    # Standardize Input A (12 sensor channels) across temporal baseline
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

    # Softmax probabilities for each head
    prob_class = preds[0][0].tolist()  # 4 class probabilities
    prob_qual = preds[1][0].tolist()   # 3 quality probabilities

    # Top prediction indices
    pred_class_idx = int(np.argmax(prob_class))
    pred_qual_idx = int(np.argmax(prob_qual))

    # Ground truth labels from catalog
    row = CACHE["test_catalog"].iloc[sample_idx]
    true_class_idx = int(row["class_id"])
    true_qual_idx = int(row["quality_grade"])

    class_match = (pred_class_idx == true_class_idx)
    qual_match = (pred_qual_idx == true_qual_idx)

    # 5. GENERATE AUTOMATED ORTHOPEDIC REHABILITATION GUIDANCE:
    rom_val = float(feat_vec[0])      # Feature 1: Total Range of Motion
    lag_val = float(feat_vec[2])      # Feature 3: Terminal Extension Deficit (Extensor Lag)
    sparc_val = float(feat_vec[5])    # Feature 6: SPARC Smoothness Metric

    feedback = []
    # Mini Squat clinical safety rules
    if pred_class_idx == 2:
        if rom_val > 50.0:
            feedback.append(
                f"⚠️ Excessive squat depth detected ({rom_val:.1f}° flexion). For early/mid-stage Tibial Plateau ORIF, "
                f"maintain depth within safe 30°-45° zone to prevent excessive articular shear stress."
            )
        else:
            feedback.append(f"✅ Knee flexion depth within safe orthopedic limits for post-tibial plateau repair ({rom_val:.1f}°).")

    # Seated Knee Extension clinical safety rules
    elif pred_class_idx == 1:
        if lag_val > 8.0:
            feedback.append(
                f"⚠️ Quadriceps extension lag of {lag_val:.1f}° detected. Target active terminal knee extension with isometric holds."
            )
        else:
            feedback.append("✅ Full terminal knee extension achieved with zero quadriceps lag.")

    # Movement Quality Assessment rules
    if pred_qual_idx == 0:
        feedback.append(f"✅ Smooth motor control without compensatory tremors (SPARC: {sparc_val:.2f}). Form verified as Good.")
    elif pred_qual_idx == 1:
        feedback.append("⚠️ Sub-optimal execution speed or slight hesitation during reversal phase (Moderate Quality).")
    else:
        feedback.append("❌ Pathological movement compensation detected (Jerk / sudden drop / compensatory trunk sway). Pause and re-align.")

    # Return structured prediction response
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


# ======================================================================================
# SECTION 9: FRONTEND WEB UI STATIC FILES MOUNT
# ======================================================================================
@app.get("/", include_in_schema=False)
def get_dashboard_ui():
    """
    WHAT THIS DOES:
      Serves the main single-page clinical dashboard user interface (web/index.html).
    """
    return FileResponse(os.path.join(WEB_DIR, "index.html"))

# Mount the static directory to serve frontend static assets (CSS, JavaScript, icons)
# NOTE: Mounted last so /api/* routes take strict precedence
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
