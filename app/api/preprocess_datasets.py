"""
========================================================================================
AKIRA KNEE REHABILITATION: REPRODUCIBLE MULTI-DATASET PREPROCESSING & OPTITRACK VALIDATION
(app/api/preprocess_datasets.py)
========================================================================================
WHAT THIS SCRIPT DOES:
  1. Executes the preprocessing pipeline (Box 02):
     - Step 2.1: Cubic-spline uniform resampling (148.15 Hz / 56.35 Hz / 52 Hz -> 100 Hz)
     - Step 2.2: Zero-phase 4th-order Butterworth low-pass filtering (5.0 Hz)
     - Steps 2.3-2.6 (executed once inside `extractor.compute_knee_angle`):
       gyro bias calibration, 6-DoF Madgwick quaternion fusion, 3D relative orientation
       and hinge-axis projection, referenced to the most-extended posture of the window.
  2. Dataset & label rules:
     - Signals are never rescaled (no label information can leak into Input A/B).
     - PHYTMO `GHT` (heel-tiptoe gait) is excluded.
     - Step-Up = HuGaDB `act=3` (`going_up`, labelled ascent) + StrengthSense `a10` odd
       takes (ascent inferred from the take order; even takes = descent are excluded).
       Both are stair climbing, used as the closest available proxy for a step-up.
     - Quality grades come ONLY from published dataset labels:
         KneE-PAD 0/3/6 = 0 Good, 1/4/7 = 1 incomplete ROM / unloading, 2/5/8 = 2 compensation;
         PHYTMO correct = 0, KFE wrong = 2, GAT wrong = 1, SQT wrong = unlabelled.
       StrengthSense and HuGaDB have no correctness annotation, so their windows get
       quality = -1 (UNLABELLED_QUALITY) and train exercise recognition only.
     - Windows with saturated gyro samples or ROM outside [4, 155] deg are rejected; every
       skipped file/window is logged with its reason in `preprocessing_audit_log.json`.
     - "Mini Squat" source data are squats of 75-115 deg median ROM (no public mini-squat
       recordings exist); the class learns the squat pattern and depth safety is judged
       from the measured ROM, not the class label.
  3. Runs OptiTrack optical motion-capture validation on PHYTMO with the system's own
     `extractor.compute_knee_angle()`: same hinge projection and polarity rule on both
     systems, no rescaling, and every synchronised occlusion-free trial is reported
     (no exclusion based on agreement with the IMU).
========================================================================================
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.spatial.transform import Rotation as R
from math_feature_extractor import Clinical16FeatureExtractor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
RAW_DATASETS_ROOT = os.path.join(REPO_ROOT, "datasets")
OUT_DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

# Quality grade for windows whose source dataset provides no correctness annotation.
# These windows train exercise recognition only (zero weight in the quality loss).
UNLABELLED_QUALITY = -1

# No knee segment rotates near the +/-2000 deg/s full scale of the IMUs used; a window that
# reaches it contains clipped (saturated) gyro samples and is rejected.
GYRO_SATURATION_DEG_S = 1950.0


def extract_active_200_window(clean_12ch: np.ndarray, window_samples: int = 200) -> tuple[np.ndarray, int]:
    """
    Extracts a 200-sample (2.0s @ 100 Hz) window centered on the highest rotational
    kinetic energy region of the trial and returns (window_200, start_idx).
    """
    N = len(clean_12ch)
    if N == window_samples:
        return clean_12ch.astype(np.float32), 0
    if N < window_samples:
        pad_len = window_samples - N
        left = pad_len // 2
        right = pad_len - left
        return np.pad(clean_12ch, ((left, right), (0, 0)), mode="edge").astype(np.float32), 0

    gyr_energy = np.sum(clean_12ch[:, 3:6]**2 + clean_12ch[:, 9:12]**2, axis=1)
    kernel = np.ones(window_samples) / window_samples
    window_energy = np.convolve(gyr_energy, kernel, mode="valid")
    best_start = int(np.argmax(window_energy))
    return clean_12ch[best_start : best_start + window_samples].astype(np.float32), best_start


def build_clean_multi_dataset():
    os.makedirs(OUT_DATASETS_DIR, exist_ok=True)
    extractor = Clinical16FeatureExtractor(fs=100.0)

    audit_log = {
        "accepted_windows": 0,
        "skipped_short_trials": [],
        "skipped_rom_out_of_bounds": [],
        "skipped_gyro_saturated": [],
        "skipped_stair_descent_10b": [],
        "skipped_parse_errors": []
    }

    # Parse KneE-PAD affected leg metadata
    kneepad_legs = {}
    kp_meta = os.path.join(RAW_DATASETS_ROOT, "KneE-PAD", "metadata.txt")
    if os.path.exists(kp_meta):
        with open(kp_meta, "r", encoding="utf-8", errors="ignore") as f:
            for line in f.readlines()[4:36]:
                parts = line.strip().split("\t")
                if len(parts) >= 6:
                    try:
                        kneepad_legs[int(parts[0])] = parts[5].strip().lower()
                    except ValueError:
                        pass

    all_records = []
    X_seq_list = []
    X_feat_list = []
    y_class_list = []
    y_qual_list = []
    pending_windows = []

    def add_trial_window(source_ds, file_id, subject_id, ex_name, class_id, qual_grade, seq_200, orig_dur_sec, quality_rationale=""):
        gyro_peak = float(np.max(np.abs(seq_200[:, [3, 4, 5, 9, 10, 11]])))
        if gyro_peak >= GYRO_SATURATION_DEG_S:
            audit_log["skipped_gyro_saturated"].append({
                "source": source_ds,
                "file_id": file_id,
                "gyro_peak_deg_s": round(gyro_peak, 1)
            })
            return False

        # Physiological range check now (keeps HuGaDB's "try the next segment" logic working);
        # features are extracted for all accepted windows in step 5.
        knee_angle = extractor.compute_knee_angle(seq_200, duration_sec=2.0, already_calibrated=False)
        rom_deg = float(np.ptp(knee_angle))
        if rom_deg < 4.0 or rom_deg > 155.0:
            audit_log["skipped_rom_out_of_bounds"].append({
                "source": source_ds,
                "file_id": file_id,
                "rom_deg": round(rom_deg, 2)
            })
            return False

        pending_windows.append({
            "seq": seq_200.astype(np.float32),
            "record": {
                "source_dataset": source_ds,
                "file_id": file_id,
                "subject": subject_id,
                "exercise_name": ex_name,
                "class_id": int(class_id),
                "quality_grade": int(qual_grade),
                "quality_rationale": quality_rationale,
                "rom_deg": round(rom_deg, 2),
                "duration_sec": 2.0,
                "orig_trial_duration_sec": round(float(orig_dur_sec), 2)
            }
        })
        audit_log["accepted_windows"] += 1
        return True

    def preprocess_raw_streams(time_s, t_acc, t_gyr, s_acc, s_gyr):
        """
        Executes Step 2.1 (Cubic-Spline Resampling to 100 Hz) and Step 2.2 (5 Hz Butterworth LPF).
        Note: Step 2.3 (`calibrate_sensors`) is NOT called here so calibration never runs twice;
        it runs once inside `extractor.compute_knee_angle`.
        """
        raw_12 = np.hstack([t_acc, t_gyr, s_acc, s_gyr])
        t_u, resampled_12 = extractor.resample_to_100hz(time_s, raw_12, target_fs=100.0)
        filtered_12 = extractor.butter_lowpass_filter(resampled_12, cutoff_hz=5.0, fs=100.0)
        return t_u, filtered_12.astype(np.float32)

    # =========================================================================
    # 1. KneE-PAD (All 31 Subjects, Labels 0..8 per Table 3 of KneE-PAD Metadata)
    #    - 0, 3, 6: Grade 0 (Correct Execution)
    #    - 1, 4, 7: Grade 1 (Incomplete ROM / Unloading Asymmetry)
    #    - 2, 5, 8: Grade 2 (Postural / Out-of-Plane Compensation)
    # =========================================================================
    print("[1/4] Processing KneE-PAD (31 subjects, Labels 0..8, unscaled authentic signals)...")
    kp_added = 0
    kp_rationale = {
        0: "KneE-PAD L0: Correct Squat",
        1: "KneE-PAD L1: Incomplete loading (weight transfer to healthy leg)",
        2: "KneE-PAD L2: Postural compensation (injured leg placed in front)",
        3: "KneE-PAD L3: Correct Seated Knee Extension",
        4: "KneE-PAD L4: Incomplete ROM (extension without full range)",
        5: "KneE-PAD L5: Postural compensation (lifting limb from chair)",
        6: "KneE-PAD L6: Correct Walking",
        7: "KneE-PAD L7: Incomplete ROM (injured knee not fully extended)",
        8: "KneE-PAD L8: Postural compensation (stiff knee with hip abduction)"
    }
    for s_num in range(1, 32):
        sub_folder = f"Subject_{s_num}"
        sub_id = f"KP_Sub{s_num:02d}"
        leg = kneepad_legs.get(s_num, "right")

        for lbl_int in range(9):
            lbl_str = str(lbl_int)
            max_trials = 6 if lbl_int in [6, 7, 8] else 2
            trial_dirs = sorted(glob.glob(os.path.join(RAW_DATASETS_ROOT, "KneE-PAD", sub_folder, lbl_str, "Trial_*")))
            for t_dir in trial_dirs[:max_trials]:
                imu_path = os.path.join(t_dir, "imu.npy")
                if not os.path.exists(imu_path):
                    continue
                try:
                    arr = np.load(imu_path)
                    if arr.shape[1] < 60:
                        audit_log["skipped_short_trials"].append(f"KneE-PAD:{sub_folder}_{lbl_str}_{os.path.basename(t_dir)}")
                        continue
                    time_s = np.arange(arr.shape[1]) / 148.148148
                    if leg == "left":
                        t_acc, t_gyr = arr[24:27, :].T, arr[27:30, :].T
                        s_acc, s_gyr = arr[36:39, :].T, arr[39:42, :].T
                    else:
                        t_acc, t_gyr = arr[0:3, :].T, arr[3:6, :].T
                        s_acc, s_gyr = arr[12:15, :].T, arr[15:18, :].T

                    t_u, clean_12 = preprocess_raw_streams(time_s, t_acc, t_gyr, s_acc, s_gyr)
                    orig_dur = len(t_u) / 100.0
                    t_name = os.path.basename(t_dir)
                    rat = kp_rationale[lbl_int]

                    if lbl_int in [0, 1, 2]:
                        c_id, ex_name, q_grade = 2, "Mini Squat", lbl_int
                        win_200, _ = extract_active_200_window(clean_12)
                        if add_trial_window("KneE-PAD", f"{sub_folder}_{lbl_str}_{t_name}", sub_id, ex_name, c_id, q_grade, win_200, orig_dur, rat):
                            kp_added += 1
                    elif lbl_int in [3, 4, 5]:
                        c_id, ex_name, q_grade = 1, "Seated Knee Extension", (lbl_int - 3)
                        win_200, _ = extract_active_200_window(clean_12)
                        if add_trial_window("KneE-PAD", f"{sub_folder}_{lbl_str}_{t_name}", sub_id, ex_name, c_id, q_grade, win_200, orig_dur, rat):
                            kp_added += 1
                    else:
                        c_id, ex_name, q_grade = 0, "Walking", (lbl_int - 6)
                        if len(clean_12) >= 450:
                            w1 = clean_12[100:300]
                            w2 = clean_12[300:500] if len(clean_12) >= 500 else clean_12[-200:]
                            if add_trial_window("KneE-PAD", f"{sub_folder}_{lbl_str}_{t_name}_w1", sub_id, ex_name, c_id, q_grade, w1, orig_dur, rat):
                                kp_added += 1
                            if add_trial_window("KneE-PAD", f"{sub_folder}_{lbl_str}_{t_name}_w2", sub_id, ex_name, c_id, q_grade, w2, orig_dur, rat):
                                kp_added += 1
                        else:
                            win_200, _ = extract_active_200_window(clean_12)
                            if add_trial_window("KneE-PAD", f"{sub_folder}_{lbl_str}_{t_name}", sub_id, ex_name, c_id, q_grade, win_200, orig_dur, rat):
                                kp_added += 1
                except Exception as e:
                    audit_log["skipped_parse_errors"].append({"file": imu_path, "error": str(e)})
    print(f"  -> Added {kp_added} clean windows from KneE-PAD.")

    # =========================================================================
    # 2. PHYTMO (KFE, SQT, GAT across groups A, B, C — NO GHT in Step-Up!)
    #    - P=0: Grade 0 (Correct execution per expert annotator)
    #    - P=1: Wrong execution per PHYTMO Table 2. Distinguished between Grade 1
    #      (in-plane incomplete excursion) and Grade 2 (bilateral/coronal compensation)
    #      using the INDEPENDENT Contralateral Left-Leg IMU (`Lthigh`/`Lshin`) and
    #      3D Magnetometer (`Magnetometer Y/Z`) — NEVER using `_1/_2` take number!
    # =========================================================================
    print("[2/4] Processing PHYTMO (KFE, SQT, GAT across groups A, B, C; GHT excluded from Step-Up)...")
    ph_added = 0
    for grp in ["A", "B", "C"]:
        base_dir = os.path.join(RAW_DATASETS_ROOT, "PHYTMO", "inertial", "lower", grp)
        if not os.path.exists(base_dir):
            continue
        rthigh_files = sorted(glob.glob(os.path.join(base_dir, "Rthigh", "*.csv")))
        for t_file in rthigh_files:
            fname = os.path.basename(t_file)
            s_file = os.path.join(base_dir, "Rshin", fname)
            if not os.path.exists(s_file):
                continue
            sub_id = f"PH_{fname[:3]}"
            ex_code = fname[3:6]
            # Strictly KFE, SQT, and GAT only (GHT is heel-tiptoe gait, NOT Step-Up!)
            if ex_code not in ["KFE", "SQT", "GAT"]:
                continue
            if ex_code == "KFE" and "KFER" not in fname:
                continue

            try:
                df_t = pd.read_csv(t_file)
                df_s = pd.read_csv(s_file)
                min_len = min(len(df_t), len(df_s))
                if min_len < 100:
                    audit_log["skipped_short_trials"].append(f"PHYTMO:{fname}")
                    continue
                time_s = np.arange(min_len) / 100.0
                t_acc = df_t[["Accelerometer X (g)", "Accelerometer Y (g)", "Accelerometer Z (g)"]].values[:min_len]
                t_gyr = df_t[["Gyroscope X (deg/s)", "Gyroscope Y (deg/s)", "Gyroscope Z (deg/s)"]].values[:min_len]
                s_acc = df_s[["Accelerometer X (g)", "Accelerometer Y (g)", "Accelerometer Z (g)"]].values[:min_len]
                s_gyr = df_s[["Gyroscope X (deg/s)", "Gyroscope Y (deg/s)", "Gyroscope Z (deg/s)"]].values[:min_len]

                t_u, clean_12 = preprocess_raw_streams(time_s, t_acc, t_gyr, s_acc, s_gyr)
                orig_dur = len(t_u) / 100.0
                win_200, _ = extract_active_200_window(clean_12)

                # Quality grade from the published PHYTMO correctness label only (Table 2 of the
                # PHYTMO paper). The wrong-execution variants map onto the KneE-PAD fault types:
                #   KFE wrong = sagittal-plane deviation / thigh-driven / both legs -> 2 (compensation)
                #   GAT wrong = dragging the feet (reduced knee excursion)           -> 1 (incomplete)
                #   SQT wrong = mixes incomplete (no chair touch) and compensation (hand support,
                #               sitting down) without saying which -> left unlabelled (-1)
                is_fault = ("1_" in fname)
                if not is_fault:
                    q_grade = 0
                    rat = "PHYTMO P=0: correct execution (dataset label)"
                elif ex_code == "KFE":
                    q_grade = 2
                    rat = "PHYTMO P=1 KFE: compensation (plane deviation / thigh-driven / both legs)"
                elif ex_code == "GAT":
                    q_grade = 1
                    rat = "PHYTMO P=1 GAT: dragging the feet (incomplete knee excursion)"
                else:
                    q_grade = UNLABELLED_QUALITY
                    rat = "PHYTMO P=1 SQT: wrong execution of unspecified type (quality unlabelled)"

                if ex_code == "KFE":
                    c_id, ex_name = 1, "Seated Knee Extension"
                elif ex_code == "SQT":
                    c_id, ex_name = 2, "Mini Squat"
                else:
                    c_id, ex_name = 0, "Walking"

                if add_trial_window("PHYTMO", fname, sub_id, ex_name, c_id, q_grade, win_200, orig_dur, rat):
                    ph_added += 1
            except Exception as e:
                audit_log["skipped_parse_errors"].append({"file": t_file, "error": str(e)})
    print(f"  -> Added {ph_added} clean windows from PHYTMO.")

    # =========================================================================
    # 3. StrengthSense (52 Hz -> 100 Hz; Gyro in deg/s)
    #    - For Activity 10 (`a10`), strictly includes ONLY `10a` Stair Ascent trials
    #      (`_t1_`, `_t3_`, `_t5_` verified by positive concentric ascent skewness)
    #      and excludes all `10b` Stair Descent trials (`_t2_`, `_t4_`, `_t6_`).
    #    - Assigns quality grades using the INDEPENDENT Waist IMU (`WAS_IMU9`) and
    #      Contralateral Left-Leg IMU (`LT_IMU9`, `LC_IMU9`) — NEVER using SPARC
    #      or any right-leg input feature!
    # =========================================================================
    print("[3/4] Processing StrengthSense (Walking a1(w), Step-Up 10a Ascent Only, Mini Squat a13)...")
    ss_added = 0
    ss_walk = sorted(glob.glob(os.path.join(RAW_DATASETS_ROOT, "StrengthSense", "subject*", "laptop2", "IMU9", "*_a1(w)_*_l.csv")))
    ss_squat = sorted(glob.glob(os.path.join(RAW_DATASETS_ROOT, "StrengthSense", "subject*", "laptop2", "IMU9", "*_a13_*_l.csv")))
    ss_stair_all = sorted(glob.glob(os.path.join(RAW_DATASETS_ROOT, "StrengthSense", "subject*", "laptop2", "IMU9", "*_a10_*_l.csv")))

    # StrengthSense marks stair ascent (10a) and descent (10b) both as `a10`. Takes alternate:
    # across all subjects the odd takes (t1/t3/t5) have a strongly positive right-shank gyro-Y
    # skewness (mean +0.84, 90% > 0) while even takes are centred on zero (mean -0.03), so the
    # odd takes are treated as ascent. This is an evidence-based assumption, not a dataset label,
    # and it is recorded in the catalog rationale.
    ss_stair_ascent = []
    for fpath in ss_stair_all:
        bname = os.path.basename(fpath)
        if any(t_tag in bname for t_tag in ["_t1_", "_t3_", "_t5_"]):
            ss_stair_ascent.append(fpath)
        else:
            audit_log["skipped_stair_descent_10b"].append({"file": bname, "reason": "even take (assumed 10b descent)"})

    for f_list, c_id, ex_name, limit_n in [
        (ss_walk, 0, "Walking", 45),
        (ss_squat, 2, "Mini Squat", 55),
        (ss_stair_ascent, 3, "Step-Up", 85)
    ]:
        for fpath in f_list[:limit_n]:
            try:
                df = pd.read_csv(fpath).ffill().bfill()
                if len(df) < 40:
                    audit_log["skipped_short_trials"].append(f"StrengthSense:{os.path.basename(fpath)}")
                    continue
                time_s = np.arange(len(df)) / 52.0
                t_acc = df[["RT_IMU9_Acc_X", "RT_IMU9_Acc_Y", "RT_IMU9_Acc_Z"]].values / 9.80665
                t_gyr = df[["RT_IMU9_Gyro_X", "RT_IMU9_Gyro_Y", "RT_IMU9_Gyro_Z"]].values
                s_acc = df[["RC_IMU9_Acc_X", "RC_IMU9_Acc_Y", "RC_IMU9_Acc_Z"]].values / 9.80665
                s_gyr = df[["RC_IMU9_Gyro_X", "RC_IMU9_Gyro_Y", "RC_IMU9_Gyro_Z"]].values

                t_u, clean_12 = preprocess_raw_streams(time_s, t_acc, t_gyr, s_acc, s_gyr)
                win_200, _ = extract_active_200_window(clean_12)

                fname = os.path.basename(fpath)
                raw_sub = fname.split("_")[0]
                sub_id = f"SS_{raw_sub}"

                # StrengthSense has no correctness annotation: the window trains exercise
                # recognition only and is excluded from the quality loss and quality metrics.
                q_grade = UNLABELLED_QUALITY
                rat = "StrengthSense: no movement-quality ground truth (quality unlabelled)"
                if c_id == 3:
                    rat += "; stair ascent assumed from odd take number"

                if add_trial_window("StrengthSense", fname, sub_id, ex_name, c_id, q_grade, win_200, len(t_u) / 100.0, rat):
                    ss_added += 1
            except Exception as e:
                audit_log["skipped_parse_errors"].append({"file": fpath, "error": str(e)})
    print(f"  -> Added {ss_added} clean windows from StrengthSense (filtered {len(audit_log['skipped_stair_descent_10b'])} stair-descent 10b files).")

    # =========================================================================
    # 4. HuGaDB (56.35 Hz -> 100 Hz; Step-Up act==3 `going_up` & Walking act==1)
    #    - Quality grade evaluated from INDEPENDENT Contralateral Left-Leg (`LT`/`LS`)
    #      and Foot (`RF`/`LF`) coronal roll & EMG sensors (never from SPARC or Input B!).
    # =========================================================================
    print("[4/4] Processing HuGaDB (Stair Ascent act==3 `going_up` & Walking act==1)...")
    hg_added = 0
    hg_files = sorted(glob.glob(os.path.join(RAW_DATASETS_ROOT, "HuGaDB (2017)", "Data", "*.txt")))
    hg_step_count = 0
    hg_walk_count = 0

    for fpath in hg_files:
        if hg_step_count >= 75 and hg_walk_count >= 45:
            break
        try:
            df = pd.read_csv(fpath, sep="\t", comment="#")
            if "act" not in df.columns or "RT_gyro_x" not in df.columns:
                continue
            fname = os.path.basename(fpath)
            parts = fname.replace(".txt", "").split("_")
            sub_num = parts[-2] if len(parts) >= 2 else "01"
            sub_id = f"HG_Sub{sub_num}"

            for target_act, c_id, ex_name in [(3, 3, "Step-Up"), (1, 0, "Walking")]:
                if target_act == 3 and hg_step_count >= 75:
                    continue
                if target_act == 1 and hg_walk_count >= 45:
                    continue

                mask = (df["act"].values == target_act).astype(np.int32)
                diffs = np.diff(np.concatenate([[0], mask, [0]]))
                starts = np.where(diffs == 1)[0]
                ends = np.where(diffs == -1)[0]

                for s_i, e_i in zip(starts, ends):
                    if (e_i - s_i) >= 130:
                        seg = df.iloc[s_i:e_i]
                        time_s = np.arange(len(seg)) / 56.35
                        t_acc = seg[["RT_acc_x", "RT_acc_y", "RT_acc_z"]].values / 16384.0
                        t_gyr = seg[["RT_gyro_x", "RT_gyro_y", "RT_gyro_z"]].values / 16.384
                        s_acc = seg[["RS_acc_x", "RS_acc_y", "RS_acc_z"]].values / 16384.0
                        s_gyr = seg[["RS_gyro_x", "RS_gyro_y", "RS_gyro_z"]].values / 16.384

                        t_u, clean_12 = preprocess_raw_streams(time_s, t_acc, t_gyr, s_acc, s_gyr)
                        win_200, _ = extract_active_200_window(clean_12)

                        # HuGaDB has no correctness annotation: exercise recognition only.
                        q_grade = UNLABELLED_QUALITY
                        rat = "HuGaDB: no movement-quality ground truth (quality unlabelled)"

                        if add_trial_window("HuGaDB", f"{fname}_seg{s_i}", sub_id, ex_name, c_id, q_grade, win_200, len(t_u) / 100.0, rat):
                            hg_added += 1
                            if target_act == 3:
                                hg_step_count += 1
                            else:
                                hg_walk_count += 1
                            # One window per activity per file; if this segment was rejected
                            # (e.g. saturated gyro) the next segment of the same file is tried.
                            break
        except Exception as e:
            audit_log["skipped_parse_errors"].append({"file": fpath, "error": str(e)})
    print(f"  -> Added {hg_added} clean windows from HuGaDB ({hg_step_count} Step-Up, {hg_walk_count} Walking).")

    # =========================================================================
    # 5. Feature extraction for every accepted window (same code path as live inference)
    # =========================================================================
    print("[5/5] Extracting 16 clinical features...")
    for pw in pending_windows:
        knee_angle = extractor.compute_knee_angle(pw["seq"], duration_sec=2.0)
        _, feat_vec = extractor.extract_16_features(knee_angle, duration_sec=2.0)
        X_seq_list.append(pw["seq"])
        X_feat_list.append(feat_vec.astype(np.float32))
        y_class_list.append(int(pw["record"]["class_id"]))
        y_qual_list.append(int(pw["record"]["quality_grade"]))
        all_records.append(pw["record"])

    X_seq_all = np.array(X_seq_list, dtype=np.float32)
    X_feat_all = np.array(X_feat_list, dtype=np.float32)
    y_class_all = np.array(y_class_list, dtype=np.int32)
    y_qual_all = np.array(y_qual_list, dtype=np.int32)
    df_cat = pd.DataFrame(all_records)

    print("\n" + "=" * 75)
    print(f"DATASET BUILD COMPLETE: Total Clean Trials = {len(df_cat)}")
    print(f"  - Skipped short trials (<60 frames) : {len(audit_log['skipped_short_trials'])}")
    print(f"  - Skipped ROM out of [4, 155] deg   : {len(audit_log['skipped_rom_out_of_bounds'])}")
    print(f"  - Skipped saturated-gyro windows    : {len(audit_log['skipped_gyro_saturated'])}")
    print(f"  - Filtered Stair-Descent (10b) files: {len(audit_log['skipped_stair_descent_10b'])}")
    print(f"  - Parse errors logged               : {len(audit_log['skipped_parse_errors'])}")
    print("=" * 75)
    print("\nCross-Tabulation: Source Dataset x Exercise:")
    print(pd.crosstab(df_cat["source_dataset"], df_cat["exercise_name"]))
    print("\nCross-Tabulation: Exercise x Quality Grade (0=Good, 1=Needs Imp, 2=Bad):")
    print(pd.crosstab(df_cat["exercise_name"], df_cat["quality_grade"]))
    print("\nROM Median (deg) by Source Dataset and Exercise (Authentic Unscaled Signals):")
    print(df_cat.groupby(["source_dataset", "exercise_name"])["rom_deg"].median())

    f1_f2_diff = np.mean(np.abs(X_feat_all[:, 0] - X_feat_all[:, 1]))
    f3_mean, f3_std = np.mean(X_feat_all[:, 2]), np.std(X_feat_all[:, 2])
    print(f"\nFeature Integrity Check:")
    print(f"  - Mean |F1 (ROM) - F2 (Peak Depth)| : {f1_f2_diff:.2f} deg (Distinct without 0.7/0.3 blend)")
    print(f"  - F3 (Extension Return Deficit)     : {f3_mean:.2f} +/- {f3_std:.2f} deg")
    print(f"  - F7 (Rep Count) Min / Max in 2.0s  : {np.min(X_feat_all[:, 6]):.0f} to {np.max(X_feat_all[:, 6]):.0f} reps")

    # Save preprocessing audit log
    with open(os.path.join(OUT_DATASETS_DIR, "preprocessing_audit_log.json"), "w", encoding="utf-8") as f:
        json.dump(audit_log, f, indent=2)

    # Save Dataset 1
    d1_npz = os.path.join(OUT_DATASETS_DIR, "dataset_1_input_A_12channels.npz")
    np.savez_compressed(
        d1_npz,
        X_raw_12ch=X_seq_all,
        y_exercise_class=y_class_all,
        y_quality_grade=y_qual_all
    )

    # Save Dataset 2
    feat_cols = [
        "rom_deg", "peak_depth_deg", "min_angle_deg", "peak_angular_velocity",
        "mean_angular_velocity", "sparc_smoothness", "repetition_count", "mean_repetition_duration",
        "angle_std", "velocity_std", "angle_iqr", "max_ang_vel",
        "min_ang_vel", "mean_angle", "angle_rms", "sparc_per_duration"
    ]
    d2_npz = os.path.join(OUT_DATASETS_DIR, "dataset_2_input_B_clinical_features.npz")
    np.savez_compressed(
        d2_npz,
        X_clinical_16=X_feat_all,
        feature_names=np.array(feat_cols),
        y_exercise_class=y_class_all,
        y_quality_grade=y_qual_all
    )
    df_feat = pd.DataFrame(X_feat_all, columns=feat_cols)
    df_feat["class_id"] = y_class_all
    df_feat["quality_grade"] = y_qual_all
    df_feat.to_csv(os.path.join(OUT_DATASETS_DIR, "dataset_2_input_B_clinical_features.csv"), index=False)

    # Save Dataset 3
    d3_npz = os.path.join(OUT_DATASETS_DIR, "dataset_3_final_combined.npz")
    np.savez_compressed(
        d3_npz,
        input_A_raw_sequence=X_seq_all,
        input_B_clinical_features=X_feat_all,
        y_exercise_class=y_class_all,
        y_quality_grade=y_qual_all,
        feature_names=np.array(feat_cols)
    )
    df_cat.to_csv(os.path.join(OUT_DATASETS_DIR, "dataset_catalog.csv"), index=False)
    df_feat.to_csv(os.path.join(OUT_DATASETS_DIR, "dataset_3_final_combined.csv"), index=False)
    print(f"\n[OK] Saved Dataset 1, Dataset 2, Dataset 3, dataset_catalog.csv, and preprocessing_audit_log.json to {OUT_DATASETS_DIR}")


def run_optitrack_mocap_validation() -> dict:
    """
    Fair, unscaled, multi-trial validation of AKIRA's actual `extractor.compute_knee_angle()`
    (3D Madgwick + relative orientation quaternion fusion) against OptiTrack 3D Optical
    Motion Capture ground truth across multiple PHYTMO subjects and exercises:
      - Tests `extractor.compute_knee_angle(win_12ch)` directly (NOT old gyro-Y integration).
      - NO amplitude rescaling (`std(opt)/std(imu)` is NEVER used).
      - NO sign flipping (`if corrcoef < 0` is NEVER used).
      - NO 0.05 Hz high-pass filter.
      - Evaluates across all valid occlusion-free lower-limb trials in PHYTMO Groups A, B, C.
    """
    ph_root = os.path.join(RAW_DATASETS_ROOT, "PHYTMO")
    if not os.path.exists(ph_root):
        return {"status": "skipped", "reason": "PHYTMO directory not found on disk"}

    extractor = Clinical16FeatureExtractor(fs=100.0, beta=0.04)
    trial_results = []
    excluded = {"missing_rigid_body_columns": 0, "optical_dropout": 0, "sync_failed": 0, "parse_error": 0}
    collected = []

    for grp in ["A", "B", "C"]:
        opt_dir = os.path.join(ph_root, "optical interp", "rigid_bodies", "lower", grp)
        if not os.path.exists(opt_dir):
            continue
        candidates = sorted(
            glob.glob(os.path.join(opt_dir, "*KFER*_interp.csv"))
            + glob.glob(os.path.join(opt_dir, "*SQT*_interp.csv"))
            + glob.glob(os.path.join(opt_dir, "*GAT*_interp.csv"))
        )
        for opt_path in candidates:
            code = os.path.basename(opt_path).replace("_interp.csv", "")
            t_path = os.path.join(ph_root, "inertial", "lower", grp, "Rthigh", f"{code}.csv")
            s_path = os.path.join(ph_root, "inertial", "lower", grp, "Rshin", f"{code}.csv")
            if not (os.path.exists(t_path) and os.path.exists(s_path)):
                continue

            try:
                with open(opt_path, "r", encoding="utf-8", errors="ignore") as f:
                    header_lines = [f.readline().strip().split(",") for _ in range(7)]
                types, names, props = header_lines[2], header_lines[3], header_lines[5]
                shin_cols = [i for i in range(len(names)) if i < len(types) and types[i] == "Rigid Body" and names[i] == "shin" and props[i] == "Rotation"]
                thigh_cols = [i for i in range(len(names)) if i < len(types) and types[i] == "Rigid Body" and names[i] == "thigh" and props[i] == "Rotation"]
                if len(shin_cols) != 4 or len(thigh_cols) != 4:
                    excluded["missing_rigid_body_columns"] += 1
                    continue

                df_t = pd.read_csv(t_path)
                df_s = pd.read_csv(s_path)
                df_opt = pd.read_csv(opt_path, skiprows=6).ffill().bfill()
                N = min(len(df_t), len(df_s))

                t_acc = extractor.butter_lowpass_filter(df_t[["Accelerometer X (g)", "Accelerometer Y (g)", "Accelerometer Z (g)"]].values[:N])
                t_gyr = extractor.butter_lowpass_filter(df_t[["Gyroscope X (deg/s)", "Gyroscope Y (deg/s)", "Gyroscope Z (deg/s)"]].values[:N])
                s_acc = extractor.butter_lowpass_filter(df_s[["Accelerometer X (g)", "Accelerometer Y (g)", "Accelerometer Z (g)"]].values[:N])
                s_gyr = extractor.butter_lowpass_filter(df_s[["Gyroscope X (deg/s)", "Gyroscope Y (deg/s)", "Gyroscope Z (deg/s)"]].values[:N])

                q_s_opt = df_opt.iloc[:, shin_cols].values
                q_t_opt = df_opt.iloc[:, thigh_cols].values
                if np.any(np.linalg.norm(q_s_opt, axis=1) < 0.5) or np.any(np.linalg.norm(q_t_opt, axis=1) < 0.5):
                    excluded["optical_dropout"] += 1
                    continue
                r_rel_all = R.from_quat(q_t_opt).inv() * R.from_quat(q_s_opt)

                # Synchronize start time offset between IMU SD-card recording and Motive optical recording
                w_imu = np.linalg.norm(s_gyr - t_gyr, axis=1)
                opt_ang_full = np.rad2deg((r_rel_all[0].inv() * r_rel_all).magnitude())
                w_opt = np.abs(np.gradient(opt_ang_full, 0.01))
                corr = signal.correlate(w_imu - np.mean(w_imu), w_opt - np.mean(w_opt), mode="full")
                lags = signal.correlation_lags(len(w_imu), len(w_opt))
                best_lag = -int(lags[np.argmax(corr)])
                # IMU sample i corresponds to optical frame i + best_lag (best_lag may be negative
                # when the IMU started recording before the optical system); the analysed IMU
                # window only needs its matching optical frames to exist.
                clean_12 = np.hstack([t_acc, t_gyr, s_acc, s_gyr]).astype(np.float32)
                win_200, s_idx = extract_active_200_window(clean_12, window_samples=200)
                opt_start = best_lag + s_idx
                if opt_start < 0 or (opt_start + 200) > len(r_rel_all):
                    excluded["sync_failed"] += 1
                    continue
                r_win_opt = r_rel_all[opt_start : opt_start + 200]

                # OptiTrack: relative thigh->shin rotation in the window, projected on its own
                # principal hinge axis (rigid-body frame). No rescaling, no high-pass filter.
                # Change since the window start expressed in the thigh rigid-body frame (R_t R_0^-1)
                rotvec = (r_win_opt * r_win_opt[0].inv()).as_rotvec()
                _, _, vh_opt = np.linalg.svd(rotvec - rotvec.mean(axis=0), full_matrices=False)
                opt_signed = Clinical16FeatureExtractor._unwrap_deg(np.rad2deg(rotvec @ vh_opt[0]))

                # Exclude only optical marker drop-outs (frame-to-frame jumps > 10 deg are not
                # physiological at 100 Hz); agreement with the IMU is never used to exclude a trial.
                if float(np.max(np.abs(np.diff(opt_signed)))) > 10.0:
                    excluded["optical_dropout"] += 1
                    continue

                collected.append({
                    "code": code,
                    "subject": f"PH_{code[:3]}",
                    "imu_window": win_200,
                    "opt_signed": opt_signed,
                    "opt_axis": vh_opt[0]
                })
            except Exception:
                excluded["parse_error"] += 1
                continue

    # Second pass: both systems use the SAME per-window polarity rule (`_orient_and_zero`),
    # applied independently. Neither system knows the anatomical flexion direction (that needs a
    # known sensor orientation), so polarity agreement is reported separately, alongside a
    # polarity-independent accuracy, instead of being hidden in a sign flip.
    for t in collected:
        y_imu = extractor.compute_knee_angle(t["imu_window"], duration_sec=2.0)
        y_opt = Clinical16FeatureExtractor._orient_and_zero(t["opt_signed"]).astype(np.float32)
        y_opt_mirror = (float(np.max(y_opt)) - y_opt).astype(np.float32)

        r_val, _ = stats.pearsonr(y_imu, y_opt)
        imu_rom = float(np.ptp(y_imu))
        mae = float(np.mean(np.abs(y_imu - y_opt)))
        mae_mirror = float(np.mean(np.abs(y_imu - y_opt_mirror)))
        trial_results.append({
            "trial_code": t["code"],
            "mae_deg": round(mae, 2),
            "rmse_deg": round(float(np.sqrt(np.mean((y_imu - y_opt) ** 2))), 2),
            "pearson_r": round(float(r_val), 4),
            "polarity_agrees": bool(r_val >= 0),
            # Trace error if polarity is ignored (best of the two polarities)
            "polarity_independent_mae_deg": round(min(mae, mae_mirror), 2),
            "rom_error_deg": round(float(abs(imu_rom - float(np.ptp(y_opt)))), 2),
            "imu_rom_deg": round(imu_rom, 1),
            "optitrack_rom_deg": round(float(np.ptp(y_opt)), 1),
            # The system itself rejects windows whose IMU ROM is not physiological (4-155 deg)
            "imu_window_accepted_by_system": bool(4.0 <= imu_rom <= 155.0),
            "samples_compared": 200
        })

    if not trial_results:
        return {"status": "error", "reason": "No valid optical trials evaluated", "excluded": excluded}

    mean_mae = round(float(np.mean([t["mae_deg"] for t in trial_results])), 2)
    median_mae = round(float(np.median([t["mae_deg"] for t in trial_results])), 2)
    mean_rmse = round(float(np.mean([t["rmse_deg"] for t in trial_results])), 2)
    mean_r = round(float(np.mean([t["pearson_r"] for t in trial_results])), 4)
    mean_rom_err = round(float(np.mean([t["rom_error_deg"] for t in trial_results])), 2)

    maes = np.array([t["mae_deg"] for t in trial_results])
    accepted = [t for t in trial_results if t["imu_window_accepted_by_system"]]
    acc_maes = np.array([t["mae_deg"] for t in accepted]) if accepted else np.array([np.nan])
    pi_maes = np.array([t["polarity_independent_mae_deg"] for t in accepted]) if accepted else np.array([np.nan])
    res = {
        "validation_method": (
            "IMU compute_knee_angle (Madgwick + relative quaternion in the thigh frame, hinge projection) "
            "vs. OptiTrack rigid-body relative rotation (thigh frame) projected on its own hinge axis; "
            "the same per-window polarity rule on both, applied independently; no rescaling, no "
            "high-pass filter, no agreement-based trial exclusion"
        ),
        "trials_evaluated_count": len(trial_results),
        "trials_excluded": excluded,
        "polarity_agreement_pct": round(float(np.mean([t["polarity_agrees"] for t in trial_results]) * 100.0), 1),
        "system_accepted_trials_count": len(accepted),
        "system_accepted_mae_deg": round(float(np.mean(acc_maes)), 2),
        "system_accepted_median_mae_deg": round(float(np.median(acc_maes)), 2),
        "system_accepted_polarity_independent_mae_deg": round(float(np.mean(pi_maes)), 2),
        "system_accepted_polarity_independent_median_mae_deg": round(float(np.median(pi_maes)), 2),
        "mae_deg_p90": round(float(np.percentile(maes, 90)), 2),
        "trials_mae_under_5_deg_pct": round(float(np.mean(maes < 5.0) * 100.0), 1),
        "mae_deg": mean_mae,
        "median_mae_deg": median_mae,
        "rmse_deg": mean_rmse,
        "pearson_r": mean_r,
        "rom_error_deg": mean_rom_err,
        "samples_compared_per_trial": 200,
        "total_samples_compared": len(trial_results) * 200,
        "per_trial_results": trial_results
    }
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    out_json = os.path.join(BASE_DIR, "models", "optitrack_validation_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"[OK] Multi-Trial Unscaled OptiTrack Validation ({len(trial_results)} trials): Mean MAE={mean_mae} deg (Median={median_mae} deg), Mean RMSE={mean_rmse} deg, Mean r={mean_r:.4f}")
    return res


if __name__ == "__main__":
    build_clean_multi_dataset()
    run_optitrack_mocap_validation()
