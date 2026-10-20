"""Dataset curation, anomaly filtering, and split-manifest generation."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data.schemas import (
    JU_IMU_14CLASS_MAP,
    JU_IMU_TO_SHARED_7CLASS,
    OPTOTRAK_10CLASS_MAP,
    OPTOTRAK_TO_SHARED_7CLASS,
    PHYTMO_3CLASS_MAP,
    PHYTMO_6CLASS_MAP,
    PHYTMO_TO_SHARED_2CLASS,
)
from src.data.splits import load_benchmark_config, verify_disjoint_subject_splits


def assign_split(subject_id: str, split_cfg: dict) -> Optional[str]:
    """Return 'train', 'val', or 'test' for a subject_id, or None if excluded."""
    if subject_id in split_cfg["train"]:
        return "train"
    if subject_id in split_cfg["val"]:
        return "val"
    if subject_id in split_cfg["test"]:
        return "test"
    return None


def curate_phytmo_upper(raw_root: Path, config: dict) -> Tuple[List[dict], List[dict]]:
    """Curate PHYTMO inertial/upper trials and record included and excluded items."""
    phytmo_upper = raw_root / "PHYTMO" / "inertial" / "upper"
    split_cfg = config["subject_splits"]["PHYTMO"]
    verify_disjoint_subject_splits(split_cfg["train"], split_cfg["val"], split_cfg["test"], "PHYTMO")

    trial_sensors: Dict[Tuple[str, str, str], Dict[str, str]] = defaultdict(dict)
    excluded: List[dict] = []

    for root, _, files in os.walk(phytmo_upper):
        sensor = os.path.basename(root)
        for fname in sorted(files):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(root, fname)
            if "Calib" in fname:
                excluded.append({
                    "dataset": "PHYTMO",
                    "item": os.path.relpath(fpath, raw_root),
                    "reason": "Static calibration recording (not an active rehabilitation exercise trial)",
                })
                continue
            m = re.match(r"^([A-E]\d{2})([A-Z]{3})([01])_(\d)\.csv$", fname)
            if not m:
                excluded.append({
                    "dataset": "PHYTMO",
                    "item": os.path.relpath(fpath, raw_root),
                    "reason": "Unrecognized filename pattern",
                })
                continue
            subj, ex_base, qual_str, series = m.group(1), m.group(2), m.group(3), m.group(4)
            ex_full = f"{ex_base}{qual_str}"
            trial_sensors[(subj, ex_full, series)][sensor] = os.path.relpath(fpath, raw_root)

    included: List[dict] = []
    for (subj, ex_full, series), s_map in sorted(trial_sensors.items()):
        trial_id = f"PHYTMO_{subj}_{ex_full}_{series}"
        if set(s_map.keys()) != {"Larm", "Lforearm", "Rarm", "Rforearm"}:
            excluded.append({
                "dataset": "PHYTMO",
                "item": trial_id,
                "reason": f"Incomplete 4-sensor set: found {sorted(s_map.keys())}",
            })
            continue
        split = assign_split(subj, split_cfg)
        if split is None:
            excluded.append({
                "dataset": "PHYTMO",
                "item": trial_id,
                "reason": f"Subject {subj} not assigned to any split",
            })
            continue
        ex_base = ex_full[:3]
        qual = int(ex_full[3])
        included.append({
            "dataset": "PHYTMO",
            "trial_id": trial_id,
            "subject_id": subj,
            "split": split,
            "exercise_code": ex_base,
            "exercise_quality_code": ex_full,
            "series": int(series),
            "y_3class": PHYTMO_3CLASS_MAP[ex_base],
            "y_6class": PHYTMO_6CLASS_MAP[ex_full],
            "y_quality": qual,
            "y_shared_2class": PHYTMO_TO_SHARED_2CLASS.get(ex_full, None),
            "sensors": s_map,
            "native_hz": 100.0,
        })

    return included, excluded


def curate_optotrak_vs_dot(raw_root: Path, config: dict) -> Tuple[List[dict], List[dict]]:
    """Curate Optotrak vs DOT trials for Track 1A (Classification) and Track 3A (Angle Regression)."""
    opto_root = raw_root / "Optotrak vs DOT" / "Optotrak vs DOT" / "Data"
    cls_cfg = config["subject_splits"]["Optotrak_vs_DOT_Classification"]
    reg_cfg = config["subject_splits"]["Optotrak_vs_DOT_AngleRegression"]
    verify_disjoint_subject_splits(cls_cfg["train"], cls_cfg["val"], cls_cfg["test"], "Optotrak_Classif")
    verify_disjoint_subject_splits(reg_cfg["train"], reg_cfg["val"], reg_cfg["test"], "Optotrak_AngleReg")

    included: List[dict] = []
    excluded: List[dict] = []

    # Log discarded raw CSVs in 'delete' / 'Fist attempt before deleting RLA' subfolders
    for root, _, files in os.walk(opto_root):
        rel_root = os.path.relpath(root, raw_root)
        if "delete" in root.lower() or "fist attempt" in root.lower():
            csv_files = [f for f in files if f.endswith(".csv")]
            if csv_files:
                excluded.append({
                    "dataset": "Optotrak vs DOT",
                    "item": f"{rel_root} ({len(csv_files)} CSV files)",
                    "reason": "Discarded / backup sensor recordings placed in 'delete' or 'Fist attempt' subfolder by dataset authors",
                })

    for s_idx in range(1, 17):
        subj = f"subject {s_idx}"
        s_dir = opto_root / subj
        b_path = s_dir / "B_dotComplete"
        h_path = s_dir / "30. Results" / "H_Joint_Angles_Aligned_FINAL"

        if subj == "subject 2":
            excluded.append({
                "dataset": "Optotrak vs DOT",
                "item": f"{subj} (all 20 trials)",
                "reason": "Sensors mounted on left arm due to right-side injury (Readme.docx); excluded from right-arm benchmarks",
            })
            continue

        split_cls = assign_split(subj, cls_cfg)
        split_reg = assign_split(subj, reg_cfg)

        if subj == "subject 8":
            excluded.append({
                "dataset": "Optotrak vs DOT (Track 3A Angle Regression only)",
                "item": f"{subj} (H_Joint_Angles_Aligned_FINAL)",
                "reason": "Missing medial epicondyle (ME) bony landmark digitization; included in Track 1A Classification but excluded from Track 3A Angle Regression",
            })

        # Trials 1..10 are calibration; Trials 11..20 are functional/ROM evaluation tasks
        excluded.append({
            "dataset": "Optotrak vs DOT (Track 1A Eval Tasks)",
            "item": f"{subj} Trials 1..10",
            "reason": "Calibration trials (N-pose and stick-guided calibration); kept separate from Tasks 11..20 functional evaluation benchmark",
        })

        for trial_num in range(11, 21):
            cls_id, task_name = OPTOTRAK_10CLASS_MAP[trial_num]
            included.append({
                "dataset": "Optotrak vs DOT",
                "trial_id": f"Optotrak_{subj.replace(' ', '')}_T{trial_num:02d}",
                "subject_id": subj,
                "trial_number_1idx": trial_num,
                "trial_index_0idx": trial_num - 1,
                "split_classification": split_cls,
                "split_angle_regression": split_reg,
                "eligible_for_angle_regression": bool(h_path.exists() and split_reg is not None),
                "task_name": task_name,
                "y_10class": cls_id,
                "y_shared_7class": OPTOTRAK_TO_SHARED_7CLASS.get(trial_num, None),
                "b_dot_path": os.path.relpath(b_path, raw_root),
                "h_angles_path": os.path.relpath(h_path, raw_root) if h_path.exists() else None,
                "native_hz": 120.0,
            })

    return included, excluded


def curate_ju_imu(raw_root: Path, config: dict) -> Tuple[List[dict], List[dict]]:
    """Curate JU-IMU ROM trials across 41 subjects (27 ND + 14 Stroke)."""
    ju_root = raw_root / "JU-IMU" / "ROM"
    split_cfg = config["subject_splits"]["JU_IMU"]
    verify_disjoint_subject_splits(split_cfg["train"], split_cfg["val"], split_cfg["test"], "JU_IMU")

    included: List[dict] = []
    excluded: List[dict] = []

    for fname in sorted(os.listdir(ju_root)):
        if not fname.endswith(".csv"):
            continue
        m = re.match(r"^ROM_([A-Za-z]+\d+)_([A-Za-z0-9]+)\.csv$", fname)
        if not m:
            excluded.append({
                "dataset": "JU-IMU",
                "item": fname,
                "reason": "Unrecognized filename format",
            })
            continue
        subj, act = m.group(1), m.group(2)
        split = assign_split(subj, split_cfg)
        if split is None or act not in JU_IMU_14CLASS_MAP:
            excluded.append({
                "dataset": "JU-IMU",
                "item": fname,
                "reason": f"Unmapped subject ({subj}) or activity ({act})",
            })
            continue
        cohort = 1 if subj.startswith("Stroke") else 0
        included.append({
            "dataset": "JU-IMU",
            "trial_id": f"JUIMU_{subj}_{act}",
            "subject_id": subj,
            "cohort": "Stroke" if cohort == 1 else "ND",
            "y_cohort": cohort,
            "split": split,
            "activity_name": act,
            "y_14class": JU_IMU_14CLASS_MAP[act],
            "y_shared_7class": JU_IMU_TO_SHARED_7CLASS.get(act, None),
            "csv_path": os.path.relpath(ju_root / fname, raw_root),
            "native_hz": 80.0,
        })

    return included, excluded


def curate_strokerehab(raw_root: Path, config: dict) -> Tuple[List[dict], List[dict]]:
    """Curate StrokeRehab CSV files, excluding Schema B/C and NaN RotX/Y/Z files."""
    sr_root = raw_root / "StrokeRehab" / "files" / "healthy controls"
    split_cfg = config["subject_splits"]["StrokeRehab"]
    verify_disjoint_subject_splits(split_cfg["train"], split_cfg["val"], split_cfg["test"], "StrokeRehab")

    included: List[dict] = []
    excluded: List[dict] = []

    known_nan_files = {
        "C5_RT_SB_mod_drinking 5.csv",
        "C5_SB_mod_drinking 5.csv",
    }

    for side in ("LT", "RT"):
        side_dir = sr_root / side
        if not side_dir.exists():
            continue
        for subj in sorted(os.listdir(side_dir)):
            subj_dir = side_dir / subj
            if not subj_dir.is_dir():
                continue
            split = assign_split(subj, split_cfg)
            for act_folder in sorted(os.listdir(subj_dir)):
                act_dir = subj_dir / act_folder
                if not act_dir.is_dir():
                    continue
                canonical_act = act_folder.lower().replace("deodrant", "deodorant")
                for fname in sorted(os.listdir(act_dir)):
                    if not fname.endswith(".csv"):
                        continue
                    fpath = act_dir / fname
                    rel_path = os.path.relpath(fpath, raw_root)
                    if fname in known_nan_files:
                        excluded.append({
                            "dataset": "StrokeRehab",
                            "item": rel_path,
                            "reason": "Massive NaNs (~50,000 cells) in RotX/Y/Z orientation channels",
                        })
                        continue
                    if fname.endswith("_labelled.csv") or (subj == "C26" and act_folder == "RTT left side" and side == "LT"):
                        excluded.append({
                            "dataset": "StrokeRehab",
                            "item": rel_path,
                            "reason": "Non-standard 80/84-column schema (omits 27 segment course/pitch/roll columns and/or LT UpperArm Accel)",
                        })
                        continue
                    included.append({
                        "dataset": "StrokeRehab",
                        "trial_id": f"StrokeRehab_{side}_{subj}_{fname[:-4]}",
                        "subject_id": subj,
                        "focus_side": side,
                        "activity": canonical_act,
                        "split": split,
                        "csv_path": rel_path,
                        "native_hz": 100.0,
                    })

    return included, excluded


def curate_vidimu(raw_root: Path, config: dict) -> Tuple[List[dict], List[dict]]:
    """Curate VIDIMU upper-limb IMU + IK trials (A05..A13), excluding mislabeled S48 files and lower-limb A01..A04."""
    vid_root = raw_root / "VIDIMU" / "dataset" / "videoandimus"
    split_cfg = config["subject_splits"]["VIDIMU"]
    verify_disjoint_subject_splits(split_cfg["train"], split_cfg["val"], split_cfg["test"], "VIDIMU")

    included: List[dict] = []
    excluded: List[dict] = []

    excluded.append({
        "dataset": "VIDIMU",
        "item": "VIDIMU/dataset/videonly (S01..S39, 508 CSV files)",
        "reason": "Video-only subjects lacking wearable IMU sensors",
    })

    for subj in sorted(os.listdir(vid_root)):
        subj_dir = vid_root / subj
        if not subj_dir.is_dir():
            continue
        split = assign_split(subj, split_cfg)
        for fname in sorted(os.listdir(subj_dir)):
            if re.match(r"^S48_A\d{2}_T\d{2}\.mot$", fname):
                excluded.append({
                    "dataset": "VIDIMU",
                    "item": os.path.relpath(subj_dir / fname, raw_root),
                    "reason": "Mislabeled 60 Hz .mot file in S48 folder whose internal header is 'ik_S53_A..' (cross-subject contamination from S53)",
                })
                continue
            m = re.match(r"^(S\d{2})_(A\d{2})_(T\d{2})\.sto$", fname)
            if not m:
                continue
            s_id, act, t_rep = m.group(1), m.group(2), m.group(3)
            act_num = int(act[1:])
            if act_num <= 4:
                excluded.append({
                    "dataset": "VIDIMU",
                    "item": os.path.relpath(subj_dir / fname, raw_root),
                    "reason": f"Lower-limb activity ({act}) with pelvis/femur/tibia IMU placement rather than upper-limb humerus/radius",
                })
                continue
            mot_file = subj_dir / f"ik_{s_id}_{act}_{t_rep}.mot"
            if not mot_file.exists():
                excluded.append({
                    "dataset": "VIDIMU",
                    "item": os.path.relpath(subj_dir / fname, raw_root),
                    "reason": "Missing corresponding ik_*.mot file",
                })
                continue
            included.append({
                "dataset": "VIDIMU",
                "trial_id": f"VIDIMU_{s_id}_{act}_{t_rep}",
                "subject_id": s_id,
                "activity": act,
                "y_9class_upper": act_num - 5,
                "split": split,
                "sto_path": os.path.relpath(subj_dir / fname, raw_root),
                "mot_path": os.path.relpath(mot_file, raw_root),
                "native_hz": 50.0,
            })

    return included, excluded


def curate_ultra_mocap(raw_root: Path, config: dict) -> Tuple[List[dict], List[dict]]:
    """Curate ULTra-MoCap trials across 13 subjects (P01..P13), matching IMU and Processed IK files."""
    ultra_root = raw_root / "ULTra-MoCap"
    split_cfg = config["subject_splits"]["ULTra_MoCap"]
    verify_disjoint_subject_splits(split_cfg["train"], split_cfg["val"], split_cfg["test"], "ULTra_MoCap")

    imu_map: Dict[str, Tuple[str, str]] = {}
    ik_map: Dict[str, str] = {}

    for part in ("ULTra-MoCap-raw-0", "ULTra-MoCap-raw-1"):
        part_dir = ultra_root / part / part
        if not part_dir.exists():
            continue
        for subj in sorted(os.listdir(part_dir)):
            subj_dir = part_dir / subj
            if not subj_dir.is_dir():
                continue
            for root, _, files in os.walk(subj_dir):
                folder_name = os.path.basename(root)
                for f in files:
                    if not f.endswith(".csv"):
                        continue
                    rel = os.path.relpath(os.path.join(root, f), raw_root)
                    if folder_name == "IMU":
                        imu_map[f.lower()] = (subj, rel)
                    elif folder_name == "Processed IK":
                        ik_map[f.lower()] = rel

    included: List[dict] = []
    excluded: List[dict] = []

    ex_name_map = {
        "elbowflexion": (0, "Elbow Flexion"),
        "shoulderrotation": (1, "Shoulder Rotation"),
        "crossbodyreach": (2, "Crossbody Reach"),
        "armswing": (3, "Arm Swing"),
        "overheadreach": (4, "Overhead Reach"),
    }

    for key_lower, (subj, imu_rel) in sorted(imu_map.items()):
        ik_rel = ik_map.get(key_lower, None)
        base = key_lower[:-4]  # strip .csv
        parts = base.split("_")
        ex_key = parts[1] if len(parts) >= 3 else "unknown"
        cond = parts[2] if len(parts) >= 3 else "unknown"
        cls_id, cls_name = ex_name_map.get(ex_key, (-1, ex_key))
        split = assign_split(subj, split_cfg)

        if ik_rel is None:
            excluded.append({
                "dataset": "ULTra-MoCap (Track 3C Angle Regression)",
                "item": imu_rel,
                "reason": "Missing corresponding OpenSim Inverse Kinematics (Processed IK / Raw IK) file for overheadreach_45 trial",
            })
            continue

        included.append({
            "dataset": "ULTra-MoCap",
            "trial_id": f"ULTra_{base}",
            "subject_id": subj,
            "split": split,
            "exercise_name": cls_name,
            "y_5class": cls_id,
            "condition": cond,
            "imu_path": imu_rel,
            "ik_path": ik_rel,
            "imu_hz": 2000.0,
            "ik_hz": 100.0,
        })

    return included, excluded


def curate_ulf_in_adl(raw_root: Path, config: dict) -> Tuple[List[dict], List[dict]]:
    """Curate ULF_in_ADL .mvnx trials across 25 subjects (20 Stroke P02..P30, 5 Healthy H01..H05)."""
    ulf_root = raw_root / "ULF_in_ADL"
    split_cfg = config["subject_splits"]["ULF_in_ADL"]
    verify_disjoint_subject_splits(split_cfg["train"], split_cfg["val"], split_cfg["test"], "ULF_in_ADL")

    # Verified Affected Upper Limb map from ParticipantCharacteristics.xlsx
    affected_limb_map = {
        "P02": "L", "P03": "R", "P04": "L", "P05": "R", "P06": "L",
        "P08": "R", "P09": "L", "P10": "L", "P12": "R", "P16": "R",
        "P17": "L", "P19": "L", "P20": "L", "P21": "R", "P22": "R",
        "P24": "L", "P25": "R", "P28": "R", "P29": "L", "P30": "R",
    }
    fmma_ue_map = {
        "P02": 54, "P03": 55, "P04": 55, "P05": 38, "P06": 41,
        "P08": 47, "P09": 40, "P10": 49, "P12": 43, "P16": 37,
        "P17": 53, "P19": 59, "P20": 37, "P21": 61, "P22": 42,
        "P24": 48, "P25": 60, "P28": 34, "P29": 46, "P30": 21,
    }

    included: List[dict] = []
    excluded: List[dict] = []

    for p in sorted(ulf_root.rglob("*.mvnx")):
        rel_path = os.path.relpath(p, raw_root)
        m = re.match(r"^([HP]\d{2})_(T\d{2})_([LR])(\d+)([A-Za-z]*)\.mvnx$", p.name, flags=re.IGNORECASE)
        if not m:
            excluded.append({
                "dataset": "ULF_in_ADL",
                "item": rel_path,
                "reason": "Unrecognized .mvnx filename pattern",
            })
            continue
        subj = m.group(1).upper()
        task = m.group(2).upper()
        side = m.group(3).upper()
        rep = int(m.group(4))
        suffix = m.group(5)
        if suffix:
            excluded.append({
                "dataset": "ULF_in_ADL",
                "item": rel_path,
                "reason": f"Error-flagged recording with '{suffix}' filename suffix",
            })
            continue
        split = assign_split(subj, split_cfg)
        if split is None:
            continue

        # 3-class limb impairment status: 0=Healthy, 1=Non-Paretic, 2=Paretic
        if subj.startswith("H"):
            limb_status = 0
            limb_status_name = "Healthy"
        else:
            aff_side = affected_limb_map[subj]
            if side == aff_side:
                limb_status = 2
                limb_status_name = "Paretic"
            else:
                limb_status = 1
                limb_status_name = "Non-Paretic"

        included.append({
            "dataset": "ULF_in_ADL",
            "trial_id": f"ULF_{subj}_{task}_{side}{rep}",
            "subject_id": subj,
            "task_code": task,
            "active_side": side,
            "repetition": rep,
            "y_limb_3class": limb_status,
            "limb_status_name": limb_status_name,
            "fmma_ue": fmma_ue_map.get(subj, 66),
            "split": split,
            "mvnx_path": rel_path,
            "native_hz": 60.0,
        })

    return included, excluded


def curate_rehab_anomalies(raw_root: Path) -> List[dict]:
    """Record the verified anomalies and exclusions in REHAB and the 5 excluded datasets."""
    return [
        {
            "dataset": "REHAB",
            "item": "REHAB/Rehab_exercise/d02_processed_data/014_1.npy",
            "reason": "Corrupted binary .npy file containing UTF-8 replacement bytes (\\xef\\xbf\\xbd) in header; raises UnpicklingError (raw counterpart d01_raw_data/014_1.npy is intact)",
        },
        {
            "dataset": "REHAB",
            "item": "REHAB/Rehab_assessment/d02_processed_data/195/2021-03-03 11-00-11 (81 .npy files)",
            "reason": "Duplicate nested subfolder inside assessment session 195",
        },
        {
            "dataset": "REHAB",
            "item": "REHAB/Rehab_assessment/*/015..025_*.npy (11 distal hand/wrist tasks per session)",
            "reason": "Distal data-glove tasks where upper-limb IMU streams (_1.npy and _2.npy) are all zeros (0.0)",
        },
        {
            "dataset": "EMG elbow dataset",
            "item": "Entire dataset (120 measurement .txt files)",
            "reason": "Contains only 2-channel sEMG + 1 scalar joint angle column; zero raw accelerometer or gyroscope IMU channels exist",
        },
        {
            "dataset": "mm-fit",
            "item": "Entire dataset (21 workouts w00..w20)",
            "reason": "No upper-arm IMU exists (wrists, earbud, pocket phones only), missing local workout-to-participant mapping, and 5,256,856 duplicate ms timestamps",
        },
        {
            "dataset": "MoCap and IMU Dataset for Upper Limb Joint Angle Estimation",
            "item": "Entire dataset (Take1..Take6)",
            "reason": "Only 1 IMU per arm (wrist + chest; no upper-arm IMU) and no subject identifiers across the 6 takes",
        },
        {
            "dataset": "Smartwatch IMU Dataset for Upper-Limb Movement Ana",
            "item": "Entire dataset (3 subjects u01, u04, u05)",
            "reason": "Wrist-only Apple Watch (no upper-arm IMU), only N=3 subjects, embedded duplicate header rows in all_users_valid.csv, and missing raw Gyro for u01",
        },
        {
            "dataset": "UCI Biceps Curl",
            "item": "Example_WearableComputing_weight_lifting_exercises_biceps_curl_variations.csv (4,024-row excerpt)",
            "reason": "Severe subject-class confounding in 4,024-row excerpt (adelmo, eurico, jeremy only have Class E; pedro only has A,B,C; carlitos only has A,D,E), preventing leak-free subject-independent splits",
        },
    ]
