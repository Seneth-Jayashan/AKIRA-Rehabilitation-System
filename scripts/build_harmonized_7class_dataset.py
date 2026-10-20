"""Build the harmonized 7-class rehabilitation exercise dataset.

Extracts 12-channel dual-IMU recordings at 100 Hz across 4 datasets:
1. PHYTMO
2. Optotrak vs DOT
3. JU-IMU
4. ULTra-MoCap

Classes:
0: Elbow Flexion-Extension
1: Forearm Pronation-Supination
2: Shoulder Flexion / Overhead Reach
3: Shoulder Abduction / Scaption
4: Shoulder Internal-External Rotation
5: Crossbody Reach / Horizontal Adduction
6: Functional Drinking / Reach-to-Grasp
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.loaders import (
    load_phytmo_trial_12ch,
    load_optotrak_trial_12ch,
    load_ju_imu_trial_12ch,
    load_ultra_mocap_trial_12ch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_PATH = PROJECT_ROOT / "configs" / "benchmark_tracks.json"

CLASSES = [
    "Elbow Flexion-Extension",
    "Forearm Pronation-Supination",
    "Shoulder Flexion / Overhead Reach",
    "Shoulder Abduction / Scaption",
    "Shoulder Internal-External Rotation",
    "Crossbody Reach / Horizontal Adduction",
    "Functional Drinking / Reach-to-Grasp",
]

WINDOW_SIZE = 200  # 2.0s at 100 Hz
STRIDE = 100       # 1.0s at 100 Hz (50% overlap)


def get_splits() -> Dict[str, Dict[str, List[str]]]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    s = cfg["subject_splits"]
    return {
        "phytmo": {
            "train": s["PHYTMO"]["train"],
            "val": s["PHYTMO"]["val"],
            "test": s["PHYTMO"]["test"],
        },
        "optotrak": {
            "train": s["Optotrak_vs_DOT_Classification"]["train"],
            "val": s["Optotrak_vs_DOT_Classification"]["val"],
            "test": s["Optotrak_vs_DOT_Classification"]["test"],
        },
        "ju_imu": {
            "train": s["JU_IMU"]["train"],
            "val": s["JU_IMU"]["val"],
            "test": s["JU_IMU"]["test"],
        },
        "ultra_mocap": {
            "train": ["P01", "P02", "P04", "P05", "P07", "P08", "P10", "P11"],
            "val": ["P03", "P09"],
            "test": ["P06", "P12", "P13"],
        },
    }


def extract_windows(signal_12ch: np.ndarray, class_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = len(signal_12ch)
    if n_samples < WINDOW_SIZE:
        pad_len = WINDOW_SIZE - n_samples
        padded = np.pad(signal_12ch, ((0, pad_len), (0, 0)), mode="edge")
        return padded[np.newaxis, :, :], np.array([class_idx], dtype=np.int64)

    windows = []
    labels = []
    n_win = (n_samples - WINDOW_SIZE) // STRIDE + 1
    for w in range(n_win):
        st = w * STRIDE
        windows.append(signal_12ch[st:st + WINDOW_SIZE])
        labels.append(class_idx)

    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64)


def collect_phytmo(splits: Dict[str, List[str]]) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]:
    data = {"train": ([], []), "val": ([], []), "test": ([], [])}
    phytmo_dir = RAW_DIR / "PHYTMO" / "inertial" / "upper"

    # Tasks: EFE0 -> 0, EAH0 -> 2
    task_map = {"EFE0": 0, "EAH0": 2}

    for cohort_dir in phytmo_dir.iterdir():
        if not cohort_dir.is_dir() or cohort_dir.name not in ["A", "B", "C", "D", "E"]:
            continue
        rarm_dir = cohort_dir / "Rarm"
        rfa_dir = cohort_dir / "Rforearm"
        if not rarm_dir.exists() or not rfa_dir.exists():
            continue

        for ua_csv in rarm_dir.glob("*.csv"):
            fname = ua_csv.name
            if "Calib" in fname:
                continue

            # Determine subject and task
            subj_id = fname[:3]  # e.g., A01
            split_name = None
            for sname, slist in splits.items():
                if subj_id in slist:
                    split_name = sname
                    break
            if not split_name:
                continue

            fa_csv = rfa_dir / fname
            if not fa_csv.exists():
                continue

            for t_prefix, c_idx in task_map.items():
                if t_prefix in fname:
                    try:
                        x_12ch, _ = load_phytmo_trial_12ch(ua_csv, fa_csv, target_hz=100.0)
                        wins, lbls = extract_windows(x_12ch, c_idx)
                        data[split_name][0].append(wins)
                        data[split_name][1].append(lbls)
                    except Exception as e:
                        print(f"Error loading PHYTMO {fname}: {e}")
                    break

    return data


def collect_optotrak(splits: Dict[str, List[str]]) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]:
    data = {"train": ([], []), "val": ([], []), "test": ([], [])}
    opt_dir = RAW_DIR / "Optotrak vs DOT" / "Optotrak vs DOT" / "Data"

    # Tasks: 11 (idx 10) -> 0, 12 (idx 11) -> 1, 13 (idx 12) -> 2, 14 (idx 13) -> 3, 15 (idx 14) -> 4, 18 (idx 17) -> 6
    task_map = {10: 0, 11: 1, 12: 2, 13: 3, 14: 4, 17: 6}

    for subj_dir in opt_dir.iterdir():
        if not subj_dir.is_dir() or not subj_dir.name.lower().startswith("subject"):
            continue
        subj_name = subj_dir.name  # e.g. "subject 1"
        if subj_name == "subject 2":
            continue  # Left-arm mounting

        split_name = None
        for sname, slist in splits.items():
            if subj_name in slist:
                split_name = sname
                break
        if not split_name:
            continue

        b_dot_path = subj_dir / "B_dotComplete"
        if not b_dot_path.exists():
            continue

        for trial_idx, c_idx in task_map.items():
            try:
                x_12ch, _ = load_optotrak_trial_12ch(b_dot_path, trial_idx, target_hz=100.0)
                wins, lbls = extract_windows(x_12ch, c_idx)
                data[split_name][0].append(wins)
                data[split_name][1].append(lbls)
            except Exception as e:
                print(f"Error loading Optotrak {subj_name} task {trial_idx}: {e}")

    return data


def collect_ju_imu(splits: Dict[str, List[str]]) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]:
    data = {"train": ([], []), "val": ([], []), "test": ([], [])}
    ju_dir = RAW_DIR / "JU-IMU" / "ROM"

    # Mapping
    task_map = {
        "ElbFlex": 0,
        "SupPro": 1,
        "FrontSupPro": 1,
        "ShFlex90": 2,
        "ShAbd90": 3,
        "Scaption90": 3,
        "ShRot": 4,
        "ShHorAdd": 5,
        "PickCup": 6,
    }

    for csv_path in ju_dir.glob("*.csv"):
        # Format: ROM_{Subject}_{Activity}.csv
        parts = csv_path.stem.split("_")
        if len(parts) < 3:
            continue
        subj_id = parts[1]
        act_name = parts[2]

        if act_name not in task_map:
            continue
        c_idx = task_map[act_name]

        split_name = None
        for sname, slist in splits.items():
            if subj_id in slist:
                split_name = sname
                break
        if not split_name:
            continue

        try:
            x_12ch = load_ju_imu_trial_12ch(csv_path, side="R", target_hz=100.0)
            wins, lbls = extract_windows(x_12ch, c_idx)
            data[split_name][0].append(wins)
            data[split_name][1].append(lbls)
        except Exception as e:
            print(f"Error loading JU-IMU {csv_path.name}: {e}")

    return data


def collect_ultra_mocap(splits: Dict[str, List[str]]) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]:
    data = {"train": ([], []), "val": ([], []), "test": ([], [])}

    task_map = {
        "Elbow Flexion": 0,
        "Overhead Reach": 2,
        "Shoulder Rotation": 4,
        "Crossbody Reach": 5,
    }

    raw_roots = [
        RAW_DIR / "ULTra-MoCap" / "ULTra-MoCap-raw-0" / "ULTra-MoCap-raw-0",
        RAW_DIR / "ULTra-MoCap" / "ULTra-MoCap-raw-1" / "ULTra-MoCap-raw-1",
    ]

    for root_dir in raw_roots:
        if not root_dir.exists():
            continue
        for subj_dir in root_dir.iterdir():
            if not subj_dir.is_dir() or not subj_dir.name.startswith("P"):
                continue
            subj_id = subj_dir.name

            split_name = None
            for sname, slist in splits.items():
                if subj_id in slist:
                    split_name = sname
                    break
            if not split_name:
                continue

            for task_folder, c_idx in task_map.items():
                t_dir = subj_dir / task_folder
                if not t_dir.exists():
                    continue
                imu_dir = t_dir / "IMU"
                ik_dir = t_dir / "Processed IK"
                if not imu_dir.exists() or not ik_dir.exists():
                    continue

                for imu_csv in imu_dir.glob("*.csv"):
                    ik_csv = ik_dir / imu_csv.name
                    if not ik_csv.exists():
                        continue
                    try:
                        x_12ch, _ = load_ultra_mocap_trial_12ch(imu_csv, ik_csv)
                        wins, lbls = extract_windows(x_12ch, c_idx)
                        data[split_name][0].append(wins)
                        data[split_name][1].append(lbls)
                    except Exception as e:
                        print(f"Error loading ULTra-MoCap {imu_csv.name}: {e}")

    return data


def main():
    print("=" * 70)
    print("Building Harmonized 7-Class Rehabilitation Exercise Dataset")
    print("=" * 70)

    all_splits = get_splits()

    print("\n[1/4] Collecting from PHYTMO...")
    phytmo_data = collect_phytmo(all_splits["phytmo"])

    print("\n[2/4] Collecting from Optotrak vs DOT...")
    optotrak_data = collect_optotrak(all_splits["optotrak"])

    print("\n[3/4] Collecting from JU-IMU...")
    ju_data = collect_ju_imu(all_splits["ju_imu"])

    print("\n[4/4] Collecting from ULTra-MoCap...")
    ultra_data = collect_ultra_mocap(all_splits["ultra_mocap"])

    # Combine all datasets per split
    final_splits = {}
    for s in ["train", "val", "test"]:
        X_list = (
            phytmo_data[s][0]
            + optotrak_data[s][0]
            + ju_data[s][0]
            + ultra_data[s][0]
        )
        y_list = (
            phytmo_data[s][1]
            + optotrak_data[s][1]
            + ju_data[s][1]
            + ultra_data[s][1]
        )

        X_all = np.concatenate(X_list, axis=0) if X_list else np.empty((0, WINDOW_SIZE, 12), dtype=np.float32)
        y_all = np.concatenate(y_list, axis=0) if y_list else np.empty((0,), dtype=np.int64)

        final_splits[s] = {"X": X_all, "y": y_all}
        print(f"Split {s.upper()}: {X_all.shape[0]} windows, shape = {X_all.shape}")

    # Compute frozen train-only normalizer
    X_train = final_splits["train"]["X"]
    # Shape: (N, 200, 12) -> compute mean and std across N and T
    mean_train = np.mean(X_train, axis=(0, 1)).astype(np.float32)
    std_train = np.maximum(np.std(X_train, axis=(0, 1)).astype(np.float32), 1e-8)

    print("\nFrozen Train Normalization:")
    print("  Mean:", np.round(mean_train, 3))
    print("  Std: ", np.round(std_train, 3))

    # Apply frozen normalizer to train, val, test
    for s in ["train", "val", "test"]:
        X_norm = (final_splits[s]["X"] - mean_train) / std_train
        y_arr = final_splits[s]["y"]

        out_path = PROCESSED_DIR / f"task3_7class_exercise_{s}.npz"
        np.savez_compressed(out_path, x=X_norm, y=y_arr)
        print(f"Saved: {out_path.name} -> {X_norm.shape[0]} windows")

        # Class distribution
        unique, counts = np.unique(y_arr, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"  Distribution ({s}): {dist}")

    # Save 7-class manifest
    manifest_7class = {
        "task_name": "Harmonized 7-Class Rehabilitation Exercise Classification",
        "num_classes": 7,
        "classes": CLASSES,
        "input_channels": 12,
        "sampling_rate_hz": 100.0,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "source_datasets": ["PHYTMO", "Optotrak vs DOT", "JU-IMU", "ULTra-MoCap"],
        "normalization": {
            "mean": mean_train.tolist(),
            "std": std_train.tolist(),
        },
        "subjects": all_splits,
        "window_counts": {
            s: int(final_splits[s]["X"].shape[0]) for s in ["train", "val", "test"]
        },
    }

    manifest_path = PROCESSED_DIR / "task3_7class_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_7class, f, indent=2)
    print(f"\nSaved manifest to {manifest_path.name}")
    print("=" * 70)
    print("Harmonized 7-class dataset successfully created!")
    print("=" * 70)


if __name__ == "__main__":
    main()
