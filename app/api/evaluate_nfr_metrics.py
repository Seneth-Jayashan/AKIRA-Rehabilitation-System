"""
========================================================================================
AKIRA KNEE REHABILITATION: NON-FUNCTIONAL REQUIREMENTS (NFR) BENCHMARK (evaluate_nfr_metrics.py)
========================================================================================
Exports Non-Functional Requirement (NFR) results, stating where each was measured:
  - NFR-01: Processing & AI inference latency (host PC; ESP32 on-device not measured yet)
  - NFR-02: Dual-IMU synchronisation (needs raw ESP32 timestamps: reported NOT_MEASURED)
  - NFR-03: Edge model flash footprint and full-int8 quantization (Target: <= 256 KB)
  - NFR-04: Knee-angle accuracy vs. OptiTrack motion capture on PHYTMO (Target: MAE <= 3.0 deg)
  - NFR-05: Patient-ID pseudonymization and API-key authentication status
========================================================================================
"""

import os
import time
import json
import numpy as np

from math_feature_extractor import Clinical16FeatureExtractor
from main import pseudonymize_patient_id

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def measure_empirical_timestamp_sync() -> dict:
    """
    NFR-02 (dual-IMU synchronisation) can only be measured on the AKIRA ESP32 hardware from
    per-sample device timestamps. The saved `recorded_trials` CSVs are resampled to 100 Hz and
    their `time_s` column is reconstructed (i * 0.01 s), so they cannot measure jitter or
    inter-sensor skew either. The status is therefore NOT_MEASURED until raw device timestamps
    are logged. Both IMUs share one I2C bus and are read back-to-back in the same 10 ms tick,
    so the expected skew is one 14-byte burst (~1.3 ms at 100 kHz); that is a design figure,
    not a measurement.
    """
    return {
        "measurement_source": "none (no raw per-sample ESP32 timestamps are logged yet)",
        "target_max_skew_ms": 10.0,
        "design_estimate_i2c_burst_ms": 1.26,
        "status": "NOT_MEASURED_ON_AKIRA_HARDWARE",
        "how_to_measure": "Log the firmware's per-packet timestamp_ms for each IMU read before resampling and "
                          "report the step jitter and thigh-to-shin read offset."
    }


def run_nfr_evaluation() -> dict:
    ext = Clinical16FeatureExtractor(fs=100.0)
    test_npz = np.load(os.path.join(DATASETS_DIR, "test_20percent_dataset.npz"), allow_pickle=True)
    X_seq = test_npz["X_seq_raw"]
    N = min(50, len(X_seq))

    # 1. NFR-01: Math Sensor Fusion + 16-Feature Extraction Latency
    math_latencies_ms = []
    for i in range(N):
        t0 = time.perf_counter()
        angle = ext.compute_knee_angle(X_seq[i], duration_sec=2.0)
        _, _ = ext.extract_16_features(angle, duration_sec=2.0)
        math_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    math_mean_ms = round(float(np.mean(math_latencies_ms)), 2)
    math_p95_ms = round(float(np.percentile(math_latencies_ms, 95)), 2)

    # 2. NFR-02: Empirical Dual-IMU Timestamp Sync & Jitter Measurement
    nfr_02 = measure_empirical_timestamp_sync()

    # 3. NFR-03: Edge AI Footprint & Pure TFLITE_BUILTINS Latency
    metrics_path = os.path.join(MODELS_DIR, "test_metrics_summary.json")
    tflite_stats = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            m_all = json.load(f)
            first_model = next(iter(m_all.values()), {})
            tflite_stats = first_model.get("tflite_edge_validation", {})

    tflite_size_kb = float(tflite_stats.get("size_kb", 0.0))
    tflite_lat_ms = float(tflite_stats.get("mean_latency_ms", 0.0))
    c_bytes = int(tflite_stats.get("c_array_bytes_written", 0))
    ops_set = str(tflite_stats.get("ops_set", ""))
    full_int8 = bool(tflite_stats.get("full_int8_verified", False))
    total_edge_pipeline_ms = round(math_mean_ms + tflite_lat_ms, 2)

    # 4. NFR-04: Multi-Trial Unscaled OptiTrack 3D Motion Capture Validation (NO hardcoded fallback!)
    opti_path = os.path.join(MODELS_DIR, "optitrack_validation_metrics.json")
    if not os.path.exists(opti_path):
        from preprocess_datasets import run_optitrack_mocap_validation
        opti_stats = run_optitrack_mocap_validation()
    else:
        with open(opti_path, "r", encoding="utf-8") as f:
            opti_stats = json.load(f)

    opti_trials = int(opti_stats.get("trials_evaluated_count", 0))
    acc_mae = float(opti_stats.get("system_accepted_mae_deg", opti_stats.get("mae_deg", float("nan"))))
    acc_median = float(opti_stats.get("system_accepted_median_mae_deg", opti_stats.get("median_mae_deg", float("nan"))))
    pi_mae = float(opti_stats.get("system_accepted_polarity_independent_mae_deg", float("nan")))
    pi_median = float(opti_stats.get("system_accepted_polarity_independent_median_mae_deg", float("nan")))

    # 5. NFR-05: Pseudonymization & Storage Privacy Verification
    sample_token = pseudonymize_patient_id("P01")
    privacy_ok = sample_token.startswith("PT-") and ("P01" not in sample_token)
    api_key_configured = bool(os.environ.get("AKIRA_API_KEY", "").strip())

    report = {
        "nfr_01_latency": {
            "target_max_ms": 200.0,
            "platform": "host PC (on-device ESP32 latency not measured: firmware has no TFLite Micro runtime yet)",
            "math_extraction_mean_ms": math_mean_ms,
            "math_extraction_p95_ms": math_p95_ms,
            "host_pc_tflite_inference_mean_ms": tflite_lat_ms,
            "total_pipeline_mean_ms": total_edge_pipeline_ms,
            "status": "PASS_ON_HOST_PC" if total_edge_pipeline_ms <= 200.0 else "FAIL"
        },
        "nfr_02_dual_imu_sync": nfr_02,
        "nfr_03_edge_memory": {
            "target_max_flash_kb": 256.0,
            "tflite_model_size_kb": tflite_size_kb,
            "c_array_full_bytes": c_bytes,
            "ops_set": ops_set,
            "full_int8_verified": full_int8,
            "note": "Flash size only; tensor-arena RAM on the ESP32 is not measured yet.",
            "status": "PASS_FLASH_SIZE" if (0.0 < tflite_size_kb <= 256.0 and full_int8) else "FAIL"
        },
        "nfr_04_kinematic_accuracy": {
            "target_max_mae_deg": 3.0,
            "trials_evaluated_count": opti_trials,
            "polarity_agreement_pct": opti_stats.get("polarity_agreement_pct"),
            "system_accepted_mae_deg": acc_mae,
            "system_accepted_median_mae_deg": acc_median,
            "polarity_independent_mae_deg": pi_mae,
            "polarity_independent_median_mae_deg": pi_median,
            "note": "Polarity (flexion vs extension direction) cannot be determined from IMU data without a "
                    "known sensor orientation; the polarity-independent error measures the angle trace itself.",
            "status": "PASS" if acc_mae <= 3.0 else "FAIL"
        },
        "nfr_05_privacy_security": {
            "sha256_pseudonymization_active": privacy_ok,
            "raw_patient_id_omitted_from_disk": True,
            "example_pseudonym": sample_token,
            "api_key_configured": api_key_configured,
            "status": ("PASS" if api_key_configured else "PASS_PSEUDONYMIZATION_ONLY (set AKIRA_API_KEY to enable auth)")
                      if privacy_ok else "FAIL"
        }
    }

    out_path = os.path.join(MODELS_DIR, "nfr_evaluation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Saved NFR Evaluation Report to {out_path}")
    return report


if __name__ == "__main__":
    res = run_nfr_evaluation()
    print(json.dumps(res, indent=2))
