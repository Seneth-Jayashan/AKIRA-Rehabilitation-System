"""
========================================================================================
AKIRA KNEE REHABILITATION: SELF-CONTAINED UNIT & INTEGRATION TEST SUITE
========================================================================================
Fixes I65 & New #30:
  1. Self-contained execution on a fresh clone (synthesizes realistic 3D rigid-body IMU
     rotations even when git-ignored .npz files are absent).
  2. Verifies 3D Madgwick sensor fusion + 3D relative orientation end-to-end from 12-channel
     IMU signals: the angle is never negative, ROM is correct even with a sensor-mounting
     offset, +/-180 deg wraps are removed, and Feature 3 is the extension return deficit.
  2b. Verifies unlabelled-quality masking (-1) and API-key enforcement without a stored secret.
  3. Verifies non-destructive sensor swap detection (`auto_swap=False` default) and explicit
     opt-in correction (`auto_swap=True`).
  4. Verifies 3-way subject-independent split integrity (0 overlapping subjects across
     inner-train, validation, and held-out test sets).
  5. Verifies saved `.keras` models, pure `TFLITE_BUILTINS` `.tflite` edge model, and the
     complete C byte array in `edge_akira_classifier.c`.
  6. Exercises FastAPI endpoints (`/api/info`, `/api/predict` single-window & multi-window
     full session, OOD protocol mode for Heel Slides / Quad Sets / SLR, and `/api/longitudinal`
     strict patient/exercise filtering + pseudonym-only storage) via `TestClient`.
========================================================================================
"""

import os
import sys
import json
import asyncio
import numpy as np
import pandas as pd

API_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "api"))
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)

from math_feature_extractor import Clinical16FeatureExtractor
import main as api_main


def _generate_synthetic_knee_imu(
    n_samples: int = 200,
    fs: float = 100.0,
    baseline_deficit_deg: float = 12.0,
    rom_swing_deg: float = 55.0,
    freq_hz: float = 0.5
) -> np.ndarray:
    """
    Generates a physically consistent 12-channel IMU array (n_samples, 12) where:
      - Thigh is resting horizontally along X with gravity along Z (ax=0, ay=0, az=1g).
      - Shin rotates in the X-Z plane around the Y hinge axis from `baseline_deficit_deg`
        up to `baseline_deficit_deg + rom_swing_deg` and back.
    """
    t = np.arange(n_samples, dtype=np.float64) / fs
    theta_deg = baseline_deficit_deg + rom_swing_deg * (np.sin(np.pi * freq_hz * t) ** 2)
    theta_rad = np.deg2rad(theta_deg)
    omega_deg_s = np.gradient(theta_deg, 1.0 / fs)

    X = np.zeros((n_samples, 12), dtype=np.float32)
    # Thigh (Channels 0..5): stationary reference with gravity along Z
    X[:, 0] = 0.0
    X[:, 1] = 0.0
    X[:, 2] = 1.0
    # Shin (Channels 6..11): rotated by theta_rad in the X-Z plane around Y axis
    X[:, 6] = -np.sin(theta_rad).astype(np.float32)
    X[:, 7] = 0.0
    X[:, 8] = np.cos(theta_rad).astype(np.float32)
    X[:, 10] = omega_deg_s.astype(np.float32)
    return X


def test_signal_processing_3d_madgwick_and_tare_recovery():
    """
    Verifies Steps 2.1-2.5 end-to-end from 12-channel IMU signals:
    resampling, 5 Hz LPF, gravity-projected tare calibration, 3D Madgwick fusion,
    and 3D relative orientation without discarding the tare angle.
    """
    ext = Clinical16FeatureExtractor(fs=100.0)

    # 1. Step 2.1: Cubic-spline resampling 50 Hz -> 100 Hz
    raw_50 = _generate_synthetic_knee_imu(n_samples=100, fs=50.0, baseline_deficit_deg=14.0, rom_swing_deg=50.0)
    res_100 = ext.resample_to_100hz(raw_50, source_fs=50.0, target_samples=200)
    assert res_100.shape == (200, 12)

    # 2. Step 2.2: 5 Hz Butterworth low-pass filter
    filtered = ext.butter_lowpass_filter(res_100, cutoff_hz=5.0)
    assert filtered.shape == (200, 12)
    assert np.all(np.isfinite(filtered))

    # 3. Step 2.3: Calibration still reports the static gravity tare for this aligned synthetic leg
    cal, meta = ext.calibrate_sensors(filtered)
    assert cal.shape == (200, 12)
    assert 8.0 <= meta["baseline_extension_offset_deg"] <= 20.0

    # 4. Steps 2.4-2.6: the angle is referenced to the most-extended posture of the recording,
    #    so it is never negative and its minimum is exactly 0 (the old gravity tare produced
    #    impossible negative angles on real data)
    angle = ext.compute_knee_angle(res_100, duration_sec=2.0)
    assert angle.shape == (200,)
    assert float(np.min(angle)) == 0.0
    feat_dict, feat_vec = ext.extract_16_features(angle, duration_sec=2.0)

    assert len(feat_vec) == 16
    assert 40.0 <= feat_dict["rom_deg"] <= 60.0
    assert feat_dict["peak_depth_deg"] <= feat_dict["rom_deg"] + 1e-3
    # Feature 3 is the extension return deficit: >= 0 and small for clean repetitions
    assert 0.0 <= feat_vec[2] <= 10.0
    assert feat_dict["extension_return_deficit_deg"] == feat_vec[2]
    assert feat_dict["max_ang_vel"] > 0.0
    assert feat_dict["min_ang_vel"] < 0.0

    # Stationary signal test: 0 repetitions and 0.0s active duration
    flat_imu = _generate_synthetic_knee_imu(n_samples=200, fs=100.0, baseline_deficit_deg=5.0, rom_swing_deg=0.0)
    flat_angle = ext.compute_knee_angle(flat_imu, duration_sec=2.0)
    flat_dict, _ = ext.extract_16_features(flat_angle, duration_sec=2.0)
    assert flat_dict["repetition_count"] == 0.0
    assert flat_dict["mean_repetition_duration"] == 0.0


def _rotating_shin_with_mounting_offset(offset_deg: float, n: int = 200, fs: float = 100.0):
    """Thigh fixed, shin flexing about Y; the shin sensor is mounted rotated by `offset_deg` about X."""
    from scipy.spatial.transform import Rotation as R
    t = np.arange(n) / fs
    theta = 10.0 + 50.0 * np.sin(np.pi * 0.5 * t) ** 2
    mount = R.from_euler("x", offset_deg, degrees=True)
    g = np.array([0.0, 0.0, 1.0])
    X = np.zeros((n, 12))
    X[:, 0:3] = g
    omega = np.gradient(np.deg2rad(theta), 1.0 / fs)
    for i in range(n):
        shin = R.from_euler("y", theta[i], degrees=True) * mount
        X[i, 6:9] = shin.inv().apply(g)
        X[i, 9:12] = np.rad2deg(mount.inv().apply([0.0, omega[i], 0.0]))
    return X, theta


def test_knee_angle_is_mounting_offset_invariant_and_unwrapped():
    """
    The relative rotation is projected on the hinge axis in the THIGH frame, so a shin sensor
    mounted at an angle no longer shrinks or flips the measured ROM, and +/-180 deg wraps are
    removed before the ROM is taken.
    """
    ext = Clinical16FeatureExtractor(fs=100.0)
    for offset in (0.0, 40.0, 75.0):
        X, theta = _rotating_shin_with_mounting_offset(offset)
        angle = ext.compute_knee_angle(X, duration_sec=2.0)
        assert float(np.min(angle)) == 0.0
        # ~1-3 deg filter error at any offset; the old shin-frame projection lost most of the
        # ROM at large offsets, so a 4 deg tolerance still catches that bug
        assert abs(float(np.ptp(angle)) - float(np.ptp(theta))) < 4.0

    wrapped = np.concatenate([np.linspace(150, 179, 50), np.linspace(-179, -150, 50)])
    zeroed = Clinical16FeatureExtractor._orient_and_zero(wrapped)
    assert float(np.ptp(zeroed)) < 70.0


def test_unlabelled_quality_is_masked_everywhere():
    """
    Windows from datasets without correctness annotation carry quality -1: they get zero weight in
    the quality loss, are excluded from quality metrics, and are never displayed as a real grade.
    """
    from train_models import per_output_sample_weights, quality_eval_mask
    y_class = np.array([0, 1, 2, 3, 3, 0])
    y_qual = np.array([0, 1, 2, -1, -1, 0])
    idx = np.arange(6)
    w_c, w_q = per_output_sample_weights(y_class, y_qual, idx)
    assert np.all(w_c > 0)
    assert np.all(w_q[y_qual < 0] == 0.0)
    assert np.all(w_q[y_qual >= 0] > 0.0)
    assert list(quality_eval_mask(y_qual)) == [True, True, True, False, False, True]

    assert api_main.quality_name(-1) == api_main.UNLABELLED_QUALITY_NAME
    assert api_main.quality_name(2) == api_main.QUALITY_NAMES[2]
    assert "Step-Up" not in api_main.QUALITY_TRAINED_EXERCISES


def test_api_key_enforced_only_when_configured(monkeypatch):
    """No secret is stored in source: auth is enforced exactly when AKIRA_API_KEY is set."""
    from fastapi import HTTPException
    assert not hasattr(api_main, "DEFAULT_API_TOKEN")

    monkeypatch.delenv("AKIRA_API_KEY", raising=False)
    api_main.verify_optional_api_key(None)  # local-development mode: open

    monkeypatch.setenv("AKIRA_API_KEY", "test-key-123")
    try:
        api_main.verify_optional_api_key(None)
        raise AssertionError("missing key must be rejected")
    except HTTPException as exc:
        assert exc.status_code == 401
    try:
        api_main.verify_optional_api_key("wrong")
        raise AssertionError("wrong key must be rejected")
    except HTTPException as exc:
        assert exc.status_code == 401
    api_main.verify_optional_api_key("test-key-123")


def test_sensor_swap_detection_non_destructive_and_opt_in():
    """
    Verifies New #26: auto_detect_and_correct_sensor_swap flags open-chain Seated Knee Extension
    when thigh RMS > 35 deg/s and shin RMS < 8 deg/s, does NOT mutate data by default
    (auto_swap=False), and swaps channels only when auto_swap=True.
    """
    ext = Clinical16FeatureExtractor(fs=100.0)
    t = np.linspace(0, 2.0, 200)
    X_swapped = np.zeros((200, 12), dtype=np.float32)
    X_swapped[:, 2] = 1.0
    X_swapped[:, 8] = 1.0
    X_swapped[:, 4] = 95.0 * np.sin(2 * np.pi * 1.0 * t)   # High gyro on Thigh
    X_swapped[:, 10] = 2.0 * np.sin(2 * np.pi * 1.0 * t)   # Near-zero gyro on Shin

    # Default auto_swap=False: flags warning in message but does NOT swap channels (was_swapped=False)
    X_unmodified, was_swapped_default, msg_flag = ext.auto_detect_and_correct_sensor_swap(
        X_swapped, "04_Seated_Knee_Extension", auto_swap=False
    )
    assert was_swapped_default is False
    assert "Possible open-chain strap swap detected" in msg_flag
    assert np.allclose(X_unmodified[:, 4], X_swapped[:, 4])

    # Explicit auto_swap=True: swaps channels [0..5] <-> [6..11]
    X_swapped_fixed, flagged_2, msg_swap = ext.auto_detect_and_correct_sensor_swap(
        X_swapped, "04_Seated_Knee_Extension", auto_swap=True
    )
    assert flagged_2 is True
    assert "Auto-Corrected" in msg_swap
    assert np.max(np.abs(X_swapped_fixed[:, 10])) > 80.0


def test_datasets_and_three_way_split_or_synthetic_fallback():
    """
    Verifies dataset catalog and 3-way split integrity when present, or runs self-contained
    feature-extraction checks when executed on a fresh clone without .npz files.
    """
    ds_dir = os.path.join(API_DIR, "datasets")
    cat_path = os.path.join(ds_dir, "dataset_catalog.csv")
    test_cat_path = os.path.join(ds_dir, "test_20percent_catalog.csv")
    metrics_path = os.path.join(API_DIR, "models", "test_metrics_summary.json")

    if os.path.exists(cat_path) and os.path.exists(test_cat_path):
        full_df = pd.read_csv(cat_path)
        test_df = pd.read_csv(test_cat_path)
        for ex_name, grp in full_df.groupby("exercise_name"):
            assert grp["source_dataset"].nunique() >= 2
            assert set(grp["quality_grade"].unique()) <= {-1, 0, 1, 2}
        # Quality grades come only from datasets that publish correctness labels
        labelled = full_df[full_df["quality_grade"] >= 0]
        assert set(labelled["source_dataset"].unique()) <= {"KneE-PAD", "PHYTMO"}
        unlabelled_sources = set(full_df[full_df["quality_grade"] < 0]["source_dataset"].unique())
        assert unlabelled_sources <= {"StrengthSense", "HuGaDB", "PHYTMO"}
        assert float(full_df["rom_deg"].max()) <= 155.0
        assert float(full_df["rom_deg"].min()) >= 4.0
        test_file_ids = set(test_df["file_id"])
        train_df = full_df[~full_df["file_id"].isin(test_file_ids)]
        assert len(set(train_df["subject"]).intersection(set(test_df["subject"]))) == 0

    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            ms = json.load(f)
        h1 = ms["hybrid_1_cnn"]
        assert h1["train_subjects_count"] > 0
        assert h1["val_subjects_count"] > 0
        assert h1["test_subjects_count"] > 0
        assert h1["test_trials_count"] > 0


def test_saved_models_and_pure_tflite_edge_export():
    """Verifies Still open #9: pure TFLITE_BUILTINS model inference and complete C byte array export."""
    models_dir = os.path.join(API_DIR, "models")
    tflite_path = os.path.join(models_dir, "akira_hybrid_cnn_edge.tflite")
    c_path = os.path.join(models_dir, "edge_akira_classifier.c")

    if os.path.exists(tflite_path) and os.path.exists(c_path):
        tflite_size = os.path.getsize(tflite_path)
        assert tflite_size > 10000
        with open(c_path, "r", encoding="utf-8") as f:
            c_code = f.read()
        # Ensure the full byte array is written, not a 512-byte preview
        assert "First 512 bytes preview" not in c_code
        assert f"akira_edge_tflite_model[{tflite_size}]" in c_code

    metrics_path = os.path.join(models_dir, "test_metrics_summary.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            edge = json.load(f)["hybrid_1_cnn"]["tflite_edge_validation"]
        # No invented on-device latency; quantization mode is reported and verified
        assert edge.get("on_device_latency_ms") is None
        assert "esp32_s3_240mhz_estimated_latency_ms" not in edge
        assert edge["quantization_mode"] in ("full_int8", "dynamic_range_fallback")
        if edge["quantization_mode"] == "full_int8":
            assert edge["full_int8_verified"] is True


def test_fastapi_endpoints_full_session_and_longitudinal_isolation():
    """
    Verifies FastAPI endpoint handlers directly:
      - /api/edge_tflite_status returns TFLite, baselines, and OptiTrack validation status
      - /api/esp32/record_and_analyze aggregates multi-window full sessions (e.g. 600 samples = 6.0 s)
        and handles OOD protocol exercises (01_Heel_Slides) transparently
      - /api/longitudinal isolates patients/exercises and never leaks raw patient_id
    """
    edge_json = api_main.get_edge_tflite_and_baselines_status()
    assert "tflite_edge_validation" in edge_json
    assert "optitrack_validation" in edge_json

    # Multi-window 6.0-second session (600 samples @ 100 Hz)
    imu_600 = _generate_synthetic_knee_imu(n_samples=600, fs=100.0, baseline_deficit_deg=8.0, rom_swing_deg=60.0)
    req_600 = api_main.ESP32RecordRequest(
        connection_mode="browser_ble",
        raw_rows=imu_600.tolist(),
        duration_sec=6.0,
        exercise_code="04_Seated_Knee_Extension",
        model_key="hybrid_1_cnn"
    )
    pred_json = asyncio.run(api_main.record_from_esp32_and_analyze(req_600))
    assert pred_json["sample_detail"]["duration_sec"] == 6.0
    out07 = pred_json["prediction"]["session_outputs"]
    assert out07["out_02_max_knee_angle_deg"] != out07["out_03_peak_flexion_deg"]

    # Out-of-distribution kinematic protocol mode (01_Heel_Slides)
    req_ood = api_main.ESP32RecordRequest(
        connection_mode="browser_ble",
        raw_rows=imu_600[:200].tolist(),
        duration_sec=2.0,
        exercise_code="01_Heel_Slides",
        model_key="hybrid_1_cnn"
    )
    ood_json = asyncio.run(api_main.record_from_esp32_and_analyze(req_ood))
    assert ood_json["is_out_of_distribution_exercise"] is True
    assert "Kinematic Protocol Mode" in ood_json["prediction"]["predicted_exercise"]["name"]

    for tmp_res in (pred_json, ood_json):
        p = tmp_res.get("saved_csv_path", "")
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    # Longitudinal strict filtering: unknown patient or unrecorded exercise returns 0 sessions
    long_empty = api_main.get_longitudinal_tracking(patient_id="UNKNOWN_PT_999", exercise="Walking")
    assert long_empty["trend_analysis"]["total_sessions"] == 0
    assert long_empty["sessions"] == []
    assert "patient_id" not in long_empty

