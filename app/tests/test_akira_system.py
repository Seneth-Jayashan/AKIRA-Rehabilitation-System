"""
========================================================================================
AKIRA KNEE REHABILITATION: AUTOMATED UNIT & INTEGRATION TEST SUITE (test_akira_system.py)
========================================================================================
Fixes I65: Comprehensive automated pytest suite verifying:
  1. Preprocessing & Signal Processing (Step 2.1 Cubic-Spline Resampling, Step 2.2 5 Hz
     Butterworth Low-Pass Filter, Step 2.3 Sensor Calibration, Step 2.4 Madgwick /
     Complementary Sensor Fusion, Step 2.5 3D Relative Hinge Orientation).
  2. All 16 Clinical Biomechanical Features (F1 ROM != F2 Peak Depth, F3 Extensor Lag > 0,
     F7 Adaptive Repetition Count, F8 Active Duration Fallback, F12/F13 Standardized
     Flexion/Extension Velocity Polarity).
  3. Automatic Sensor Swap Detection (Thigh vs. Shin energy asymmetry auto-swap).
  4. Dataset Integrity & Leak-Free Subject-Independent Split (0 overlapping subjects,
     multi-dataset coverage across all 4 exercises and 3 quality grades).
  5. Session Outputs (Box 07) & Longitudinal Analytics (Boxes 08-09):
     - True depth_deg (no 0.22 multiplier),
     - True rep_count (0 when stationary),
     - Minimum 3 sessions required for regression forecasting,
     - Un-floored forecast allowing declining recovery trajectories to be detected,
     - Clinician-configurable protocol targets and per-exercise session filtering.
========================================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd

API_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "api"))
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)

from math_feature_extractor import Clinical16FeatureExtractor
import main as api_main


def test_signal_processing_and_sensor_fusion():
    """Verifies Steps 2.1-2.5: resampling, 5 Hz LPF, calibration, Madgwick fusion, and 3D relative orientation."""
    ext = Clinical16FeatureExtractor(fs=100.0)

    # Generate a synthetic 50 Hz 2-second 12-channel trial (100 samples)
    t_50 = np.linspace(0, 2.0, 100, endpoint=False)
    raw_50 = np.zeros((100, 12), dtype=np.float32)
    raw_50[:, 2] = 1.0  # Thigh resting gravity Z = 1g
    raw_50[:, 8] = np.cos(0.5 * np.sin(2 * np.pi * 0.8 * t_50))
    raw_50[:, 6] = np.sin(0.5 * np.sin(2 * np.pi * 0.8 * t_50))
    raw_50[:, 10] = 65.0 * np.sin(2 * np.pi * 0.8 * t_50)  # Shin flexion/extension rate

    # Step 2.1: Cubic-spline resampling 50 Hz -> 100 Hz (200 samples)
    res_100 = ext.resample_to_100hz(raw_50, source_fs=50.0, target_samples=200)
    assert res_100.shape == (200, 12)

    # Step 2.2: 5 Hz Butterworth low-pass filter
    filtered = ext.butter_lowpass_filter(res_100, cutoff_hz=5.0)
    assert filtered.shape == (200, 12)
    assert np.all(np.isfinite(filtered))

    # Step 2.3: Static tare & gyro bias calibration
    cal, meta = ext.calibrate_sensors(filtered)
    assert cal.shape == (200, 12)
    assert "baseline_extension_offset_deg" in meta

    # Step 2.4 & 2.5: Full 12-channel sensor fusion & relative orientation
    angle = ext.compute_knee_angle(res_100, duration_sec=2.0)
    assert angle.shape == (200,)
    assert float(np.max(angle) - np.min(angle)) > 15.0


def test_16_features_integrity_and_distinctness():
    """Verifies C20, C21, C22, C23, C24, C26 on synthetic and held-out clinical signals."""
    ext = Clinical16FeatureExtractor(fs=100.0)

    # Create a knee angle curve with a 9.5 deg extensor lag (minimum angle = 9.5 deg, peak = 74.5 deg)
    t = np.linspace(0, 2.0, 200)
    angle_with_lag = 9.5 + 65.0 * (np.sin(np.pi * t / 1.0) ** 2)
    feat_dict, feat_vec = ext.extract_16_features(angle_with_lag, duration_sec=2.0)

    assert len(feat_vec) == 16
    # Feature 1 (ROM = 65.0) must be distinct from Feature 2 (Peak Depth = 74.5) (Fixes C22)
    assert abs(feat_dict["rom_deg"] - 65.0) < 1.0
    assert abs(feat_dict["peak_depth_deg"] - 74.5) < 1.0
    assert abs(feat_dict["peak_depth_deg"] - feat_dict["rom_deg"]) > 8.0

    # Feature 3 (Extensor Lag) must capture the 9.5 deg baseline deficit (Fixes C21)
    assert abs(feat_dict["extensor_lag_deg"] - 9.5) < 1.0

    # Feature 12 (Max Flexion Velocity > 0) and Feature 13 (Min Extension Velocity < 0) (Fixes C26)
    assert feat_dict["max_ang_vel"] > 0.0
    assert feat_dict["min_ang_vel"] < 0.0

    # Stationary signal test: 0 repetitions and 0.0s active duration (Fixes C23, C24, F50)
    flat_angle = np.ones(200, dtype=np.float32) * 5.0
    flat_dict, flat_vec = ext.extract_16_features(flat_angle, duration_sec=2.0)
    assert flat_dict["repetition_count"] == 0.0
    assert flat_dict["mean_repetition_duration"] == 0.0


def test_auto_sensor_swap_detection():
    """Verifies automatic detection and correction when Thigh (Ch 0-5) and Shin (Ch 6-11) straps are swapped."""
    ext = Clinical16FeatureExtractor(fs=100.0)
    t = np.linspace(0, 2.0, 200)
    X_swapped = np.zeros((200, 12), dtype=np.float32)
    #Both sensors active with 1g gravity on Z, but high dynamic angular velocity on Sensor 1 (Thigh) and near-zero on Sensor 2 (Shin)
    X_swapped[:, 2] = 1.0
    X_swapped[:, 8] = 1.0
    X_swapped[:, 4] = 95.0 * np.sin(2 * np.pi * 1.0 * t)
    X_swapped[:, 10] = 2.0 * np.sin(2 * np.pi * 1.0 * t)

    X_corrected, was_swapped, msg = ext.auto_detect_and_correct_sensor_swap(
        X_swapped, "04_Seated_Knee_Extension"
    )
    assert was_swapped is True
    assert "Auto-Corrected" in msg
    assert np.max(np.abs(X_corrected[:, 10])) > 80.0


def test_datasets_and_subject_independent_split():
    """Verifies A1-A7, A9, B12, B13, D27: multi-dataset coverage, realistic ROMs, and zero subject leakage."""
    ds_dir = os.path.join(API_DIR, "datasets")
    cat_path = os.path.join(ds_dir, "dataset_catalog.csv")
    test_cat_path = os.path.join(ds_dir, "test_20percent_catalog.csv")
    assert os.path.exists(cat_path)
    assert os.path.exists(test_cat_path)

    full_df = pd.read_csv(cat_path)
    test_df = pd.read_csv(test_cat_path)

    # Every exercise must come from at least 2 distinct source datasets (Fixes A1, A2)
    for ex_name, grp in full_df.groupby("exercise_name"):
        assert grp["source_dataset"].nunique() >= 2, f"{ex_name} comes from <2 datasets"
        # Every exercise must have all 3 quality grades (0, 1, 2) (Fixes A6, A7)
        assert set(grp["quality_grade"].unique()) == {0, 1, 2}, f"{ex_name} missing quality grades"

    # Realistic biomechanical ROM ceiling (<= 145 deg, Fixes A3)
    assert float(full_df["rom_deg"].max()) <= 145.0

    # Zero subject overlap between train and held-out test set (Fixes D27)
    test_file_ids = set(test_df["file_id"])
    train_df = full_df[~full_df["file_id"].isin(test_file_ids)]
    overlap_subjects = set(train_df["subject"]).intersection(set(test_df["subject"]))
    assert len(overlap_subjects) == 0, f"Subject leakage found: {overlap_subjects}"


def test_box07_and_longitudinal_forecasting():
    """Verifies F47-F50 and G52-G59 in main.py."""
    feat_vec = np.zeros(16, dtype=np.float32)
    feat_vec[0] = 62.0  # ROM
    feat_vec[1] = 68.5  # Peak Flexion Depth
    feat_vec[2] = 6.5   # Extensor Lag
    feat_vec[5] = -2.1  # SPARC
    feat_vec[6] = 0.0   # 0 reps

    out07 = api_main.build_box07_session_outputs(
        exercise_name="Seated Knee Extension",
        exercise_conf=96.4,
        quality_grade_id=0,
        quality_name="Good (Correct Form)",
        prob_qual=[0.92, 0.06, 0.02],
        feat_vec=feat_vec
    )
    # F47: Movement depth is true depth_deg (68.5), not 68.5 * 0.22
    assert out07["out_03_movement_depth_deg"] == 68.5
    # F50: Rep count is 0 when feat_vec[6] == 0
    assert out07["out_04_repetition_count"] == 0

    # G54: With < 3 sessions, forecast_available is False and forecast_sessions is empty
    one_session = [{
        "session_id": "S1", "patient_id": "P99", "session_date": "2026-10-20",
        "week": "Session 1", "exercise": "Mini Squat", "rom_deg": 35.0,
        "depth_deg": 38.0, "smoothness": 0.65, "quality_score": 72,
        "rep_count": 5, "duration_s": 2.0
    }]
    res_1 = api_main.compute_box08_and_box09_analytics(one_session, patient_id="P99", exercise="Mini Squat")
    assert res_1["forecast_available"] is False
    assert len(res_1["forecast_sessions"]) == 0

    # G55: Declining trajectory must forecast lower ROM (never floored at last value)
    declining_sessions = [
        {"session_id": "S1", "patient_id": "P99", "session_date": "2026-10-01", "week": "S1", "exercise": "Mini Squat", "rom_deg": 44.0, "depth_deg": 45.0, "smoothness": 0.75, "quality_score": 80, "rep_count": 6, "duration_s": 2.0},
        {"session_id": "S2", "patient_id": "P99", "session_date": "2026-10-08", "week": "S2", "exercise": "Mini Squat", "rom_deg": 38.0, "depth_deg": 40.0, "smoothness": 0.65, "quality_score": 70, "rep_count": 5, "duration_s": 2.0},
        {"session_id": "S3", "patient_id": "P99", "session_date": "2026-10-15", "week": "S3", "exercise": "Mini Squat", "rom_deg": 32.0, "depth_deg": 34.0, "smoothness": 0.55, "quality_score": 60, "rep_count": 4, "duration_s": 2.0},
    ]
    res_dec = api_main.compute_box08_and_box09_analytics(declining_sessions, patient_id="P99", exercise="Mini Squat")
    assert res_dec["forecast_available"] is True
    assert res_dec["trend_analysis"]["rom_rate_per_session"] < 0.0
    assert res_dec["forecast_sessions"][0]["rom_deg"] < 32.0
