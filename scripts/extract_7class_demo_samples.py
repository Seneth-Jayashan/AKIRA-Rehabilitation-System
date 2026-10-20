"""Extract genuine held-out test CSV samples for all 7 rehabilitation classes into data/samples/."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import (
    load_optotrak_trial_12ch,
    load_ju_imu_trial_12ch,
    load_ultra_mocap_trial_12ch,
)

RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_COLS = [
    "RUA_Ax", "RUA_Ay", "RUA_Az", "RUA_Gx", "RUA_Gy", "RUA_Gz",
    "RLA_Ax", "RLA_Ay", "RLA_Az", "RLA_Gx", "RLA_Gy", "RLA_Gz",
]


def save_sample_csv(
    filename: str,
    x_12ch: np.ndarray,
    exercise_idx: int,
    exercise_name: str,
    y_angles: np.ndarray | None = None,
) -> Path:
    n_samples = len(x_12ch)
    time_sec = np.round(np.arange(n_samples) * 0.01, 3)

    data = {"time_sec": time_sec}
    for i, col in enumerate(CANONICAL_COLS):
        data[col] = np.round(x_12ch[:, i], 5)

    data["gt_exercise_class_idx"] = exercise_idx
    data["gt_exercise_name"] = exercise_name

    if y_angles is not None:
        data["gt_elbow_flex_ext_deg"] = np.round(y_angles[:, 0], 2)
        if y_angles.shape[1] > 1:
            data["gt_forearm_pro_sup_deg"] = np.round(y_angles[:, 1], 2)

    df = pd.DataFrame(data)
    out_path = SAMPLES_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"Created sample: {out_path.name} ({n_samples} frames, {n_samples*0.01:.1f}s)")
    return out_path


def main():
    print("=" * 70)
    print("Extracting Genuine Held-Out Test Samples for 7-Class Expansion")
    print("=" * 70)

    # 1. Class 1: Forearm Pronation-Supination (Optotrak vs DOT, Subject 5 - Test)
    # Task 12 is trial index 11
    b_dot_s5 = RAW_DIR / "Optotrak vs DOT" / "Optotrak vs DOT" / "Data" / "subject 5" / "B_dotComplete"
    h_opt_s5 = RAW_DIR / "Optotrak vs DOT" / "Optotrak vs DOT" / "Data" / "subject 5" / "30. Results" / "H_Joint_Angles_Aligned_FINAL"
    x_opt_c1, y_opt_c1 = load_optotrak_trial_12ch(b_dot_s5, 11, h_angles_path=h_opt_s5, target_hz=100.0)
    save_sample_csv(
        "dot_subj05_task12_prosup_test.csv",
        x_opt_c1,
        exercise_idx=1,
        exercise_name="Forearm Pronation-Supination",
        y_angles=y_opt_c1,
    )

    # 2. Class 3: Shoulder Abduction / Scaption (Optotrak vs DOT, Subject 15 - Test)
    # Task 14 is trial index 13
    b_dot_s15 = RAW_DIR / "Optotrak vs DOT" / "Optotrak vs DOT" / "Data" / "subject 15" / "B_dotComplete"
    x_opt_c3, y_opt_c3 = load_optotrak_trial_12ch(b_dot_s15, 13, target_hz=100.0)
    save_sample_csv(
        "dot_subj15_task14_shabd_test.csv",
        x_opt_c3,
        exercise_idx=3,
        exercise_name="Shoulder Abduction / Scaption",
    )

    # 3. Class 4: Shoulder Internal-External Rotation (ULTra-MoCap, Subject P12 - Test)
    u_imu_rot = RAW_DIR / "ULTra-MoCap" / "ULTra-MoCap-raw-1" / "ULTra-MoCap-raw-1" / "P12" / "Shoulder Rotation" / "IMU" / "P12_ShoulderRotation_Normal.csv"
    u_ik_rot = RAW_DIR / "ULTra-MoCap" / "ULTra-MoCap-raw-1" / "ULTra-MoCap-raw-1" / "P12" / "Shoulder Rotation" / "Processed IK" / "P12_ShoulderRotation_Normal.csv"
    x_u_rot, y_u_rot = load_ultra_mocap_trial_12ch(u_imu_rot, u_ik_rot)
    save_sample_csv(
        "ultra_mocap_p12_shoulderrot_normal_test.csv",
        x_u_rot,
        exercise_idx=4,
        exercise_name="Shoulder Internal-External Rotation",
        y_angles=y_u_rot,
    )

    # 4. Class 6: Functional Drinking / Reach-to-Grasp (Optotrak vs DOT, Subject 5 - Test)
    # Task 18 is trial index 17
    x_opt_c6, y_opt_c6 = load_optotrak_trial_12ch(b_dot_s5, 17, target_hz=100.0)
    save_sample_csv(
        "dot_subj05_task18_drinkcup_test.csv",
        x_opt_c6,
        exercise_idx=6,
        exercise_name="Functional Drinking / Reach-to-Grasp",
    )

    # 5. Class 6 Alternate: JU-IMU PickCup (Stroke Patient Stroke13 - Test)
    ju_pickcup = RAW_DIR / "JU-IMU" / "ROM" / "ROM_Stroke13_PickCup.csv"
    x_ju_cup = load_ju_imu_trial_12ch(ju_pickcup, side="R", target_hz=100.0)
    save_sample_csv(
        "ju_imu_stroke13_pickcup_test.csv",
        x_ju_cup,
        exercise_idx=6,
        exercise_name="Functional Drinking / Reach-to-Grasp",
    )

    # Update manifest.json in data/samples/
    # Read existing manifest
    manifest_p = SAMPLES_DIR / "manifest.json"
    if manifest_p.exists():
        with open(manifest_p, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"samples": []}

    existing_samples = data.get("samples", [])
    existing_ids = {s["sample_id"] for s in existing_samples}

    new_samples_meta = [
        {
            "sample_id": "dot_subj05_task12_prosup_test",
            "filename": "dot_subj05_task12_prosup_test.csv",
            "source_dataset": "Optotrak vs DOT",
            "subject_id": "subject 5 (Held-Out Test)",
            "exercise_class_idx": 1,
            "exercise_name": "Forearm Pronation-Supination",
            "duration_sec": round(len(x_opt_c1) * 0.01, 1),
            "has_ground_truth": True,
            "has_continuous_angles": True,
            "description": "Held-out test subject 5 performing 5 reps of forearm pronation/supination (Task 12) with Optotrak ground-truth angles.",
        },
        {
            "sample_id": "dot_subj15_task14_shabd_test",
            "filename": "dot_subj15_task14_shabd_test.csv",
            "source_dataset": "Optotrak vs DOT",
            "subject_id": "subject 15 (Held-Out Test)",
            "exercise_class_idx": 3,
            "exercise_name": "Shoulder Abduction / Scaption",
            "duration_sec": round(len(x_opt_c3) * 0.01, 1),
            "has_ground_truth": True,
            "has_continuous_angles": False,
            "description": "Held-out test subject 15 performing 5 reps of shoulder abduction (Task 14) at 120 Hz resampled to 100 Hz.",
        },
        {
            "sample_id": "ultra_mocap_p12_shoulderrot_normal_test",
            "filename": "ultra_mocap_p12_shoulderrot_normal_test.csv",
            "source_dataset": "ULTra-MoCap",
            "subject_id": "P12 (Held-Out Test)",
            "exercise_class_idx": 4,
            "exercise_name": "Shoulder Internal-External Rotation",
            "duration_sec": round(len(x_u_rot) * 0.01, 1),
            "has_ground_truth": True,
            "has_continuous_angles": True,
            "description": "Held-out test subject P12 performing shoulder rotation at normal speed with synchronized OpenSim IK angles.",
        },
        {
            "sample_id": "dot_subj05_task18_drinkcup_test",
            "filename": "dot_subj05_task18_drinkcup_test.csv",
            "source_dataset": "Optotrak vs DOT",
            "subject_id": "subject 5 (Held-Out Test)",
            "exercise_class_idx": 6,
            "exercise_name": "Functional Drinking / Reach-to-Grasp",
            "duration_sec": round(len(x_opt_c6) * 0.01, 1),
            "has_ground_truth": True,
            "has_continuous_angles": False,
            "description": "Held-out test subject 5 performing functional drinking from a cup ADL task (Task 18).",
        },
        {
            "sample_id": "ju_imu_stroke13_pickcup_test",
            "filename": "ju_imu_stroke13_pickcup_test.csv",
            "source_dataset": "JU-IMU",
            "subject_id": "Stroke13 (Held-Out Test)",
            "exercise_class_idx": 6,
            "exercise_name": "Functional Drinking / Reach-to-Grasp",
            "duration_sec": round(len(x_ju_cup) * 0.01, 1),
            "has_ground_truth": True,
            "has_continuous_angles": False,
            "description": "Held-out test stroke patient Stroke13 performing PickCup task (80 Hz resampled to 100 Hz).",
        },
    ]

    for sm in new_samples_meta:
        if sm["sample_id"] not in existing_ids:
            existing_samples.append(sm)

    data["samples"] = existing_samples
    data["total_samples"] = len(existing_samples)
    data["supported_exercises"] = [
        "0: Elbow Flexion-Extension",
        "1: Forearm Pronation-Supination",
        "2: Shoulder Flexion / Overhead Reach",
        "3: Shoulder Abduction / Scaption",
        "4: Shoulder Internal-External Rotation",
        "5: Crossbody Reach / Horizontal Adduction",
        "6: Functional Drinking / Reach-to-Grasp",
    ]

    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\nUpdated manifest {manifest_p.name} with {len(existing_samples)} total samples.")
    print("=" * 70)


if __name__ == "__main__":
    main()
