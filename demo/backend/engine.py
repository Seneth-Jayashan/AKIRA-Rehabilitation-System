"""Multi-task inference engine for AI Elbow Rehabilitation.

Supports single-model and 4-model comparative analysis across:
1. Task 1: Continuous Elbow Flexion/Extension Angle Regression & ROM Analysis (deg)
2. Task 2: Continuous Forearm Pronation/Supination Angle Regression & ROM Analysis (deg)
3. Task 3: Rehabilitation Exercise Classification (3-Class)
4. Task 4: Movement-Quality Classification (Binary: Correct vs. Incorrect/Compensatory)
5. Three-Tier ROM Metrics (Window ROM, Full-Trial ROM, Repetition ROM, Cycle Detection)
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.data.schemas import CANONICAL_12_CHANNELS
from src.evaluation.rom_segmentation import (
    compute_angular_velocity,
    compute_sparc_smoothness,
    segment_repetition_cycles_rom,
)
from src.models.architectures import build_four_models

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"
METRICS_PATH = PROJECT_ROOT / "experiments" / "benchmark_summary.json"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"

MODEL_ORDER = ("TCN", "1D_CNN", "LSTM", "GRU")

CANONICAL_USER_COLS = (
    "RUA_Ax", "RUA_Ay", "RUA_Az", "RUA_Gx", "RUA_Gy", "RUA_Gz",
    "RLA_Ax", "RLA_Ay", "RLA_Az", "RLA_Gx", "RLA_Gy", "RLA_Gz",
)

ALT_COL_MAP = {
    "rua_ax": "RUA_Ax", "rua_ay": "RUA_Ay", "rua_az": "RUA_Az",
    "rua_gx": "RUA_Gx", "rua_gy": "RUA_Gy", "rua_gz": "RUA_Gz",
    "rla_ax": "RLA_Ax", "rla_ay": "RLA_Ay", "rla_az": "RLA_Az",
    "rla_gx": "RLA_Gx", "rla_gy": "RLA_Gy", "rla_gz": "RLA_Gz",
    "ua_acc_x_ms2": "RUA_Ax", "ua_acc_y_ms2": "RUA_Ay", "ua_acc_z_ms2": "RUA_Az",
    "ua_gyr_x_rads": "RUA_Gx", "ua_gyr_y_rads": "RUA_Gy", "ua_gyr_z_rads": "RUA_Gz",
    "fa_acc_x_ms2": "RLA_Ax", "fa_acc_y_ms2": "RLA_Ay", "fa_acc_z_ms2": "RLA_Az",
    "fa_gyr_x_rads": "RLA_Gx", "fa_gyr_y_rads": "RLA_Gy", "fa_gyr_z_rads": "RLA_Gz",
    "ua_acc_x": "RUA_Ax", "ua_acc_y": "RUA_Ay", "ua_acc_z": "RUA_Az",
    "ua_gyr_x": "RUA_Gx", "ua_gyr_y": "RUA_Gy", "ua_gyr_z": "RUA_Gz",
    "fa_acc_x": "RLA_Ax", "fa_acc_y": "RLA_Ay", "fa_acc_z": "RLA_Az",
    "fa_gyr_x": "RLA_Gx", "fa_gyr_y": "RLA_Gy", "fa_gyr_z": "RLA_Gz",
}

TASK_FILE_PREFIXES = {
    "task1_elbow_flexion": "task1_elbow_flexion",
    "task_1_elbow_flexion": "task1_elbow_flexion",
    "task2_forearm_prosup": "task2_forearm_prosup",
    "task_2_forearm_prosup": "task2_forearm_prosup",
    "task3_exercise_classification": "task3_exercise_classification",
    "task_3_exercise_classification": "task3_exercise_classification",
    "task3_7class_exercise": "task3_7class_exercise",
    "task_3_7class_exercise": "task3_7class_exercise",
    "task4_quality_classification": "task4_quality_classification",
    "task_4_quality_classification": "task4_quality_classification",
}


class RehabilitationEngine:
    """Multi-task inference engine executing TCN, 1D-CNN, LSTM, and GRU."""

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.processed_dir = self.project_root / "data" / "processed"
        self.checkpoints_dir = self.project_root / "models" / "checkpoints"
        self.metrics_path = self.project_root / "experiments" / "benchmark_summary.json"
        self.samples_dir = self.project_root / "data" / "samples"

        self._checkpoint_cache: Dict[Tuple[str, str], Any] = {}
        self._manifest_cache: Optional[Dict[str, Any]] = None

    def get_split_manifest(self) -> Dict[str, Any]:
        if self._manifest_cache is None:
            manifest_p = self.processed_dir / "split_manifest.json"
            if manifest_p.exists():
                with open(manifest_p, "r", encoding="utf-8") as f:
                    self._manifest_cache = json.load(f)
            else:
                self._manifest_cache = {"tasks": {}}
        return self._manifest_cache

    def get_benchmark_summary(self) -> Dict[str, Any]:
        if self.metrics_path.exists():
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        alt_path = self.project_root / "experiments" / "metrics" / "benchmark_summary.json"
        if alt_path.exists():
            with open(alt_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"tasks": {}}

    def list_samples(self) -> List[Dict[str, Any]]:
        manifest_p = self.samples_dir / "manifest.json"
        if manifest_p.exists():
            with open(manifest_p, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("samples", [])
        return []

    def load_checkpoint(self, task_key: str, model_name: str) -> Any:
        cache_key = (task_key, model_name)
        if cache_key in self._checkpoint_cache:
            return self._checkpoint_cache[cache_key]

        prefix = TASK_FILE_PREFIXES.get(task_key, task_key)
        ckpt_path = self.checkpoints_dir / f"{prefix}_{model_name.lower()}.pt"
        if not ckpt_path.exists():
            # Try alternate naming
            alt = self.checkpoints_dir / f"unified_v2_{task_key}_{model_name}_best.pt"
            if alt.exists():
                ckpt_path = alt
            else:
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.name}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        task_mode = ckpt.get("task_mode", "regression")
        in_channels = ckpt.get("in_channels", 12)

        if task_mode == "regression":
            models_dict = build_four_models(in_channels=in_channels, task_mode="regression", num_reg_targets=1)
        else:
            num_classes = ckpt.get("num_classes", len(ckpt.get("classes", [2, 2])))
            models_dict = build_four_models(in_channels=in_channels, task_mode="classification", num_classes=num_classes)

        model = models_dict[model_name]
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        bundle = {
            "model": model,
            "ckpt": ckpt,
            "task_mode": task_mode,
        }
        self._checkpoint_cache[cache_key] = bundle
        return bundle

    def parse_csv(self, csv_bytes: bytes, filename: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate and parse an uploaded or sample CSV into a standardized DataFrame."""
        try:
            df = pd.read_csv(io.BytesIO(csv_bytes))
        except Exception as exc:
            raise ValueError(f"Failed to parse CSV {filename!r}: {exc}") from exc

        col_rename = {}
        for c in df.columns:
            low_c = c.strip().lower()
            if low_c in ALT_COL_MAP:
                col_rename[c] = ALT_COL_MAP[low_c]

        df = df.rename(columns=col_rename)

        missing = [c for c in CANONICAL_USER_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV is missing required canonical IMU columns: {missing}. Expected columns: {list(CANONICAL_USER_COLS)}"
            )

        has_flex_gt = "gt_elbow_flex_ext_deg" in df.columns
        has_prosup_gt = "gt_forearm_pro_sup_deg" in df.columns
        has_ex_gt = "gt_exercise_class_idx" in df.columns or "gt_exercise_name" in df.columns
        has_qual_gt = "gt_quality_class_idx" in df.columns or "gt_quality_name" in df.columns

        gt_meta = {
            "has_ground_truth": bool(has_flex_gt or has_prosup_gt or has_ex_gt or has_qual_gt),
            "has_elbow_flex_gt": has_flex_gt,
            "has_forearm_prosup_gt": has_prosup_gt,
            "has_exercise_class_gt": has_ex_gt,
            "has_quality_class_gt": has_qual_gt,
        }

        if has_flex_gt:
            gt_meta["true_elbow_flex_traj"] = df["gt_elbow_flex_ext_deg"].to_numpy(dtype=np.float64)
        if has_prosup_gt:
            gt_meta["true_forearm_prosup_traj"] = df["gt_forearm_pro_sup_deg"].to_numpy(dtype=np.float64)
        if has_ex_gt:
            gt_meta["true_exercise_class_idx"] = int(df["gt_exercise_class_idx"].iloc[0]) if "gt_exercise_class_idx" in df.columns else None
            gt_meta["true_exercise_name"] = str(df["gt_exercise_name"].iloc[0]) if "gt_exercise_name" in df.columns else None
        if has_qual_gt:
            gt_meta["true_quality_class_idx"] = int(df["gt_quality_class_idx"].iloc[0]) if "gt_quality_class_idx" in df.columns else None
            gt_meta["true_quality_name"] = str(df["gt_quality_name"].iloc[0]) if "gt_quality_name" in df.columns else None

        return df, gt_meta

    def run_complete_analysis(
        self,
        csv_bytes: bytes,
        filename: str,
        selected_model: str = "COMPARE_ALL",
    ) -> Dict[str, Any]:
        """Execute complete multi-task analysis across all supported tasks."""
        df, gt_meta = self.parse_csv(csv_bytes, filename)
        raw_signals = df[list(CANONICAL_USER_COLS)].to_numpy(dtype=np.float32)
        n_samples = len(raw_signals)
        time_axis = df["time_sec"].tolist() if "time_sec" in df.columns else [round(i * 0.01, 3) for i in range(n_samples)]

        manifest = self.get_split_manifest().get("tasks", {})
        compare_all = selected_model in ("COMPARE_ALL", "ALL")
        models_to_run = list(MODEL_ORDER) if compare_all else [selected_model]

        # Scalers
        norm_t1 = manifest.get("task_1_elbow_flexion", {}).get("normalization", {})
        u_mean = np.array(norm_t1.get("x_scaler", {}).get("mean", [0.0] * 12), dtype=np.float32)
        u_std = np.maximum(np.array(norm_t1.get("x_scaler", {}).get("std", [1.0] * 12), dtype=np.float32), 1e-8)
        y_flex_mean = float(norm_t1.get("y_scaler", {}).get("mean", [65.48])[0])
        y_flex_std = float(norm_t1.get("y_scaler", {}).get("std", [37.62])[0])

        norm_t2 = manifest.get("task_2_forearm_prosup", {}).get("normalization", {})
        y_prosup_mean = float(norm_t2.get("y_scaler", {}).get("mean", [115.34])[0])
        y_prosup_std = float(norm_t2.get("y_scaler", {}).get("std", [30.12])[0])

        norm_t3 = manifest.get("task_3_exercise_classification", {}).get("normalization", {})
        p_mean = np.array(norm_t3.get("scaler", {}).get("mean", [0.0] * 12), dtype=np.float32)
        p_std = np.maximum(np.array(norm_t3.get("scaler", {}).get("std", [1.0] * 12), dtype=np.float32), 1e-8)

        # 7-Class Harmonized Scaler
        norm_7class_path = self.processed_dir / "task3_7class_manifest.json"
        if norm_7class_path.exists():
            with open(norm_7class_path, "r", encoding="utf-8") as f:
                m7 = json.load(f)
            p7_mean = np.array(m7.get("normalization", {}).get("mean", p_mean), dtype=np.float32)
            p7_std = np.maximum(np.array(m7.get("normalization", {}).get("std", p_std), dtype=np.float32), 1e-8)
        else:
            p7_mean, p7_std = p_mean, p_std

        # Sliding windows (200 frames = 2.0s, stride = 100 frames = 1.0s)
        window_size = 200
        stride = 100
        n_windows = max(1, (n_samples - window_size) // stride + 1)
        windows_u = []
        windows_p = []
        windows_p7 = []
        window_starts = []

        if n_samples < window_size:
            pad_len = window_size - n_samples
            padded = np.pad(raw_signals, ((0, pad_len), (0, 0)), mode="edge")
            windows_u.append((padded - u_mean) / u_std)
            windows_p.append((padded - p_mean) / p_std)
            windows_p7.append((padded - p7_mean) / p7_std)
            window_starts.append(0)
        else:
            for w in range(n_windows):
                start = w * stride
                end = start + window_size
                seg = raw_signals[start:end]
                windows_u.append((seg - u_mean) / u_std)
                windows_p.append((seg - p_mean) / p_std)
                windows_p7.append((seg - p7_mean) / p7_std)
                window_starts.append(start)

        x_tensor_u = torch.from_numpy(np.stack(windows_u, axis=0)).float()
        x_tensor_p = torch.from_numpy(np.stack(windows_p, axis=0)).float()
        x_tensor_p7 = torch.from_numpy(np.stack(windows_p7, axis=0)).float()

        summary = self.get_benchmark_summary()

        model_results = {}
        for m_name in models_to_run:
            m_res = {"model_name": m_name}
            t_start_model = time.perf_counter()

            # Task 1: Elbow Flexion
            b1 = self.load_checkpoint("task1_elbow_flexion", m_name)
            with torch.no_grad():
                out1 = b1["model"](x_tensor_u).numpy()
            pred_flex_deg = out1[:, :, 0] * y_flex_std + y_flex_mean

            recon_flex = np.zeros(n_samples, dtype=np.float64)
            count_flex = np.zeros(n_samples, dtype=np.float64)
            for w_idx, st in enumerate(window_starts):
                end_pos = min(st + window_size, n_samples)
                w_len = end_pos - st
                recon_flex[st:end_pos] += pred_flex_deg[w_idx, :w_len]
                count_flex[st:end_pos] += 1.0
            recon_flex /= np.maximum(count_flex, 1.0)

            win_rom_flex = float(np.mean(np.ptp(pred_flex_deg, axis=1)))
            trial_rom_flex = float(recon_flex.max() - recon_flex.min())
            rom_cycles_flex = segment_repetition_cycles_rom(recon_flex, sampling_rate_hz=100.0)
            rep_rom_flex = rom_cycles_flex.get("mean_repetition_rom_deg", trial_rom_flex)
            n_reps_flex = rom_cycles_flex.get("num_repetitions_detected", len(rom_cycles_flex.get("cycles", [])))

            # Angular Velocity & Movement Smoothness (SPARC)
            vel_flex = compute_angular_velocity(recon_flex, sampling_rate_hz=100.0)
            sparc_flex = compute_sparc_smoothness(vel_flex, sampling_rate_hz=100.0)

            task1_eval = {
                "current_angle_deg": round(float(recon_flex[-1]), 2),
                "predicted_trajectory": [round(float(v), 2) for v in recon_flex],
                "angular_velocity_deg_s": [round(float(v), 2) for v in vel_flex],
                "min_deg": round(float(recon_flex.min()), 2),
                "max_deg": round(float(recon_flex.max()), 2),
                "mean_deg": round(float(recon_flex.mean()), 2),
                "window_rom_deg": round(win_rom_flex, 2),
                "full_trial_rom_deg": round(trial_rom_flex, 2),
                "repetition_mean_rom_deg": round(float(rep_rom_flex), 2) if rep_rom_flex else None,
                "num_repetitions_detected": int(n_reps_flex),
                "repetition_cycles": rom_cycles_flex.get("cycles", []),
                "peak_flexion_velocity_deg_s": round(float(np.max(vel_flex)), 2),
                "peak_extension_velocity_deg_s": round(float(np.min(vel_flex)), 2),
                "mean_abs_velocity_deg_s": round(float(np.mean(np.abs(vel_flex))), 2),
                "movement_smoothness_sparc": round(float(sparc_flex), 3),
            }

            if gt_meta["has_elbow_flex_gt"]:
                gt_flex = gt_meta["true_elbow_flex_traj"]
                mae_flex = float(np.mean(np.abs(recon_flex - gt_flex)))
                rmse_flex = float(np.sqrt(np.mean((recon_flex - gt_flex) ** 2)))
                ss_res_flex = float(np.sum((recon_flex - gt_flex) ** 2))
                ss_tot_flex = float(np.sum((gt_flex - np.mean(gt_flex)) ** 2))
                r2_flex = float(1.0 - (ss_res_flex / max(ss_tot_flex, 1e-8)))
                gt_rom_flex = float(gt_flex.max() - gt_flex.min())
                gt_vel_flex = compute_angular_velocity(gt_flex, sampling_rate_hz=100.0)
                gt_sparc_flex = compute_sparc_smoothness(gt_vel_flex, sampling_rate_hz=100.0)

                task1_eval.update({
                    "sample_mae_deg": round(mae_flex, 2),
                    "sample_rmse_deg": round(rmse_flex, 2),
                    "sample_r2": round(r2_flex, 3),
                    "true_trial_rom_deg": round(gt_rom_flex, 2),
                    "rom_error_deg": round(abs(trial_rom_flex - gt_rom_flex), 2),
                    "gt_peak_velocity_deg_s": round(float(np.max(np.abs(gt_vel_flex))), 2),
                    "gt_movement_smoothness_sparc": round(float(gt_sparc_flex), 3),
                })
            m_res["task1_elbow_flexion"] = task1_eval

            # Task 2: Forearm Pronation/Supination
            b2 = self.load_checkpoint("task2_forearm_prosup", m_name)
            with torch.no_grad():
                out2 = b2["model"](x_tensor_u).numpy()
            pred_prosup_deg = out2[:, :, 0] * y_prosup_std + y_prosup_mean

            recon_prosup = np.zeros(n_samples, dtype=np.float64)
            count_prosup = np.zeros(n_samples, dtype=np.float64)
            for w_idx, st in enumerate(window_starts):
                end_pos = min(st + window_size, n_samples)
                w_len = end_pos - st
                recon_prosup[st:end_pos] += pred_prosup_deg[w_idx, :w_len]
                count_prosup[st:end_pos] += 1.0
            recon_prosup /= np.maximum(count_prosup, 1.0)

            win_rom_prosup = float(np.mean(np.ptp(pred_prosup_deg, axis=1)))
            trial_rom_prosup = float(recon_prosup.max() - recon_prosup.min())
            rom_cycles_prosup = segment_repetition_cycles_rom(recon_prosup, sampling_rate_hz=100.0)
            rep_rom_prosup = rom_cycles_prosup.get("mean_repetition_rom_deg", trial_rom_prosup)

            # Angular Velocity & Smoothness for Forearm Pro/Sup
            vel_prosup = compute_angular_velocity(recon_prosup, sampling_rate_hz=100.0)
            sparc_prosup = compute_sparc_smoothness(vel_prosup, sampling_rate_hz=100.0)

            task2_eval = {
                "current_angle_deg": round(float(recon_prosup[-1]), 2),
                "predicted_trajectory": [round(float(v), 2) for v in recon_prosup],
                "angular_velocity_deg_s": [round(float(v), 2) for v in vel_prosup],
                "min_deg": round(float(recon_prosup.min()), 2),
                "max_deg": round(float(recon_prosup.max()), 2),
                "mean_deg": round(float(recon_prosup.mean()), 2),
                "window_rom_deg": round(win_rom_prosup, 2),
                "full_trial_rom_deg": round(trial_rom_prosup, 2),
                "repetition_mean_rom_deg": round(float(rep_rom_prosup), 2) if rep_rom_prosup else None,
                "num_repetitions_detected": int(rom_cycles_prosup.get("num_repetitions_detected", 0)),
                "repetition_cycles": rom_cycles_prosup.get("cycles", []),
                "peak_velocity_deg_s": round(float(np.max(np.abs(vel_prosup))), 2),
                "mean_abs_velocity_deg_s": round(float(np.mean(np.abs(vel_prosup))), 2),
                "movement_smoothness_sparc": round(float(sparc_prosup), 3),
            }

            if gt_meta["has_forearm_prosup_gt"]:
                gt_prosup = gt_meta["true_forearm_prosup_traj"]
                mae_prosup = float(np.mean(np.abs(recon_prosup - gt_prosup)))
                rmse_prosup = float(np.sqrt(np.mean((recon_prosup - gt_prosup) ** 2)))
                ss_res_prosup = float(np.sum((recon_prosup - gt_prosup) ** 2))
                ss_tot_prosup = float(np.sum((gt_prosup - np.mean(gt_prosup)) ** 2))
                r2_prosup = float(1.0 - (ss_res_prosup / max(ss_tot_prosup, 1e-8)))
                gt_rom_prosup = float(gt_prosup.max() - gt_prosup.min())
                gt_vel_prosup = compute_angular_velocity(gt_prosup, sampling_rate_hz=100.0)
                gt_sparc_prosup = compute_sparc_smoothness(gt_vel_prosup, sampling_rate_hz=100.0)

                task2_eval.update({
                    "sample_mae_deg": round(mae_prosup, 2),
                    "sample_rmse_deg": round(rmse_prosup, 2),
                    "sample_r2": round(r2_prosup, 3),
                    "true_trial_rom_deg": round(gt_rom_prosup, 2),
                    "rom_error_deg": round(abs(trial_rom_prosup - gt_rom_prosup), 2),
                    "gt_peak_velocity_deg_s": round(float(np.max(np.abs(gt_vel_prosup))), 2),
                    "gt_movement_smoothness_sparc": round(float(gt_sparc_prosup), 3),
                })
            m_res["task2_forearm_prosup"] = task2_eval

            # Task 3: Exercise Classification (Supports both 7-Class Harmonized and 3-Class Legacy)
            ckpt_7class_path = self.checkpoints_dir / f"task3_7class_exercise_{m_name.lower()}.pt"
            if ckpt_7class_path.exists():
                b3 = self.load_checkpoint("task3_7class_exercise", m_name)
                x_input_3 = x_tensor_p7
            else:
                b3 = self.load_checkpoint("task3_exercise_classification", m_name)
                x_input_3 = x_tensor_p

            with torch.no_grad():
                logits3 = b3["model"](x_input_3).numpy()
            probs3 = torch.softmax(torch.from_numpy(logits3), dim=-1).numpy()
            mean_probs3 = probs3.mean(axis=0)
            pred_cls3 = int(np.argmax(mean_probs3))

            classes3 = b3["ckpt"].get("classes")
            if not classes3:
                if b3["ckpt"].get("num_classes", 3) == 7:
                    classes3 = [
                        "Elbow Flexion-Extension",
                        "Forearm Pronation-Supination",
                        "Shoulder Flexion / Overhead Reach",
                        "Shoulder Abduction / Scaption",
                        "Shoulder Internal-External Rotation",
                        "Crossbody Reach / Horizontal Adduction",
                        "Functional Drinking / Reach-to-Grasp",
                    ]
                else:
                    classes3 = [
                        "Elbow Flexion-Extension (EFE)",
                        "Extension of Arms Overhead (EAH)",
                        "Squeezing (SQZ)",
                    ]
            pred_name3 = classes3[pred_cls3] if pred_cls3 < len(classes3) else f"Class {pred_cls3}"

            task3_eval = {
                "predicted_class_idx": pred_cls3,
                "predicted_class_name": pred_name3,
                "confidence_pct": round(float(mean_probs3[pred_cls3]) * 100.0, 1),
                "class_probabilities": [round(float(p), 4) for p in mean_probs3],
                "classes": classes3,
                "num_classes": len(classes3),
            }
            if gt_meta["has_exercise_class_gt"]:
                gt_cls3 = gt_meta.get("true_exercise_class_idx")
                gt_name3 = gt_meta.get("true_exercise_name")
                is_correct = None
                if gt_cls3 is not None and len(classes3) == 7:
                    if gt_cls3 == pred_cls3:
                        is_correct = True
                    elif gt_name3 and (gt_name3.lower() in pred_name3.lower() or pred_name3.lower() in gt_name3.lower()):
                        is_correct = True
                    else:
                        is_correct = False
                elif gt_cls3 is not None:
                    is_correct = bool(pred_cls3 == gt_cls3)

                task3_eval.update({
                    "true_class_idx": gt_cls3,
                    "true_class_name": gt_name3,
                    "is_correct": is_correct,
                })
            m_res["task3_exercise_classification"] = task3_eval

            # Task 4: Movement Quality Classification
            b4 = self.load_checkpoint("task4_quality_classification", m_name)
            with torch.no_grad():
                logits4 = b4["model"](x_tensor_p).numpy()
            probs4 = torch.softmax(torch.from_numpy(logits4), dim=-1).numpy()
            mean_probs4 = probs4.mean(axis=0)
            pred_cls4 = int(np.argmax(mean_probs4))
            classes4 = ["Correct Execution", "Incorrect / Compensatory Execution"]
            pred_name4 = classes4[pred_cls4]

            task4_eval = {
                "predicted_quality_idx": pred_cls4,
                "predicted_quality_name": pred_name4,
                "confidence_pct": round(float(mean_probs4[pred_cls4]) * 100.0, 1),
                "quality_probabilities": [round(float(p), 4) for p in mean_probs4],
                "classes": classes4,
            }
            if gt_meta["has_quality_class_gt"]:
                gt_cls4 = gt_meta.get("true_quality_class_idx")
                task4_eval.update({
                    "true_quality_idx": gt_cls4,
                    "true_quality_name": gt_meta.get("true_quality_name"),
                    "is_correct": bool(pred_cls4 == gt_cls4) if gt_cls4 is not None else None,
                })
            m_res["task4_quality_classification"] = task4_eval

            m_res["total_inference_latency_ms"] = round((time.perf_counter() - t_start_model) * 1000.0, 2)
            model_results[m_name] = m_res

        sensor_signals = {}
        for idx, col in enumerate(CANONICAL_USER_COLS):
            sensor_signals[col] = [round(float(v), 4) for v in raw_signals[:, idx]]

        comparison_tables = self._build_comparison_tables(model_results, gt_meta, summary)
        final_assessment = self._generate_clinical_assessment(model_results, gt_meta, filename, n_samples)

        return {
            "status": "ok",
            "filename": filename,
            "duration_sec": round(n_samples * 0.01, 2),
            "sampling_rate_hz": 100.0,
            "num_timesteps": n_samples,
            "selected_model": selected_model,
            "compare_all": compare_all,
            "time_axis": time_axis,
            "ground_truth_metadata": {
                "has_ground_truth": gt_meta["has_ground_truth"],
                "has_elbow_flex_gt": gt_meta["has_elbow_flex_gt"],
                "has_forearm_prosup_gt": gt_meta["has_forearm_prosup_gt"],
                "has_exercise_class_gt": gt_meta["has_exercise_class_gt"],
                "has_quality_class_gt": gt_meta["has_quality_class_gt"],
                "true_elbow_flex_traj": [round(float(v), 2) for v in gt_meta["true_elbow_flex_traj"]] if gt_meta["has_elbow_flex_gt"] else None,
                "true_elbow_velocity_deg_s": [round(float(v), 2) for v in compute_angular_velocity(gt_meta["true_elbow_flex_traj"], 100.0)] if gt_meta["has_elbow_flex_gt"] else None,
                "true_forearm_prosup_traj": [round(float(v), 2) for v in gt_meta["true_forearm_prosup_traj"]] if gt_meta["has_forearm_prosup_gt"] else None,
                "true_forearm_prosup_velocity_deg_s": [round(float(v), 2) for v in compute_angular_velocity(gt_meta["true_forearm_prosup_traj"], 100.0)] if gt_meta["has_forearm_prosup_gt"] else None,
                "true_exercise_name": gt_meta.get("true_exercise_name"),
                "true_quality_name": gt_meta.get("true_quality_name"),
            },
            "sensor_signals": sensor_signals,
            "model_predictions": model_results,
            "comparison_tables": comparison_tables,
            "final_clinical_assessment": final_assessment,
        }

    def _generate_clinical_assessment(
        self,
        model_results: Dict[str, Any],
        gt_meta: Dict[str, Any],
        filename: str,
        n_samples: int,
    ) -> Dict[str, Any]:
        """Synthesize an evidence-grounded clinical rehabilitation assessment summary."""
        # Use primary benchmark model (TCN or first available)
        primary_name = "TCN" if "TCN" in model_results else list(model_results.keys())[0]
        m = model_results[primary_name]

        t1 = m.get("task1_elbow_flexion", {})
        t2 = m.get("task2_forearm_prosup", {})
        t3 = m.get("task3_exercise_classification", {})
        t4 = m.get("task4_quality_classification", {})

        elbow_rom = t1.get("full_trial_rom_deg", 0.0)
        forearm_rom = t2.get("full_trial_rom_deg", None)
        reps = t1.get("num_repetitions_detected", 0)
        sparc = t1.get("movement_smoothness_sparc", None)
        quality_class = t4.get("predicted_quality_name", "Not Evaluated")
        quality_conf = t4.get("confidence_pct", 0.0)
        exercise_name = t3.get("predicted_class_name", "Rehabilitation Drill")

        # 1. ROM Adequacy Evaluation (Target: 130 deg functional elbow flexion)
        if elbow_rom >= 130.0:
            rom_status = "Full Functional Range (Normal ADL Target Reached)"
            rom_badge = "Excellent"
        elif elbow_rom >= 110.0:
            rom_status = "Moderate Functional Range (Sufficient for Most ADLs)"
            rom_badge = "Good"
        elif elbow_rom >= 90.0:
            rom_status = "Basic Functional Arc (Mild End-Range Restriction)"
            rom_badge = "Fair"
        else:
            rom_status = "Severe Range-of-Motion Deficit (Guarding / Contracture)"
            rom_badge = "Restricted"

        # 2. Movement Smoothness & Motor Control (SPARC)
        if sparc is not None:
            if sparc >= -2.5:
                smooth_status = "Coordinated & Fluid Motor Execution"
                smooth_badge = "Smooth"
            elif sparc >= -3.5:
                smooth_status = "Moderate Sub-movements / Tremor Adjustments"
                smooth_badge = "Moderate"
            else:
                smooth_status = "Arrested / Segmented Execution (Significant Hesitation)"
                smooth_badge = "Jerky"
        else:
            smooth_status = "Insufficient Velocity Data"
            smooth_badge = "N/A"

        # 3. Compensatory Execution Risk
        if quality_class == "Correct Execution":
            comp_risk = "Low Risk — Normal Synergistic Movement Control"
            comp_badge = "Compliant"
        else:
            comp_risk = "Elevated Risk — Compensatory Trunk/Shoulder Substitution Detected"
            comp_badge = "Compensatory"

        # 4. Actionable Evidence-Based Recommendations
        recs: List[str] = []
        if elbow_rom < 110.0:
            recs.append(f"Prioritize active-assisted end-range flexion exercises to expand functional arc toward 130° (current: {elbow_rom}°).")
        if sparc is not None and sparc < -2.8:
            recs.append("Incorporate rhythmic pacing or metronome-guided drills to reduce corrective sub-movements and tremor.")
        if quality_class != "Correct Execution":
            recs.append(f"Patient demonstrated compensatory movement ({quality_conf}% confidence). Reinforce posture and stabilize shoulder girdle.")
        if reps < 3:
            recs.append("Recording duration contained fewer than 3 full movement cycles. Extend trial window to assess muscular endurance.")
        if not recs:
            recs.append("Movement execution meets clinical milestones. Continue maintenance program and progress load gradually.")

        return {
            "evaluating_model": primary_name,
            "exercise_identified": exercise_name,
            "measured_elbow_rom_deg": elbow_rom,
            "target_elbow_rom_deg": 130.0,
            "rom_achievement_pct": round(min(100.0, (elbow_rom / 130.0) * 100.0), 1),
            "rom_status": rom_status,
            "rom_badge": rom_badge,
            "measured_forearm_rom_deg": forearm_rom,
            "repetition_cycles_detected": reps,
            "smoothness_sparc": sparc,
            "smoothness_status": smooth_status,
            "smoothness_badge": smooth_badge,
            "movement_quality_class": quality_class,
            "movement_quality_confidence_pct": quality_conf,
            "compensatory_risk": comp_risk,
            "compensatory_badge": comp_badge,
            "actionable_recommendations": recs,
            "evidence_basis": {
                "sampling_rate_hz": 100.0,
                "duration_sec": round(n_samples * 0.01, 2),
                "total_frames": n_samples,
                "has_ground_truth": gt_meta["has_ground_truth"],
                "source_file": filename,
            },
            "clinical_disclaimer": (
                "This assessment is derived strictly from kinematic sensor data and validated AI model inferences. "
                "It represents an objective biomechanical summary and is NOT a substitute for clinical diagnosis or physical examination by a licensed medical practitioner."
            ),
        }

    def _build_comparison_tables(
        self,
        model_results: Dict[str, Any],
        gt_meta: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Format 4-model comparison tables for each of the 4 tasks."""
        tables = {
            "task1_elbow_flexion": [],
            "task2_forearm_prosup": [],
            "task3_exercise_classification": [],
            "task4_quality_classification": [],
        }

        tasks_summary = summary.get("tasks", summary)

        for m_name in MODEL_ORDER:
            if m_name not in model_results:
                continue
            m = model_results[m_name]

            # Task 1
            t1 = m["task1_elbow_flexion"]
            s1 = tasks_summary.get("task_1_elbow_flexion", {}).get("models", {}).get(m_name, {})
            tables["task1_elbow_flexion"].append({
                "model_name": m_name,
                "current_angle_deg": t1.get("current_angle_deg"),
                "sample_mae_deg": t1.get("sample_mae_deg"),
                "sample_rmse_deg": t1.get("sample_rmse_deg"),
                "sample_r2": t1.get("sample_r2"),
                "predicted_rom_deg": t1["full_trial_rom_deg"],
                "repetition_rom_deg": t1.get("repetition_mean_rom_deg"),
                "rom_error_deg": t1.get("rom_error_deg"),
                "num_repetitions": t1["num_repetitions_detected"],
                "movement_smoothness_sparc": t1.get("movement_smoothness_sparc"),
                "peak_flexion_velocity_deg_s": t1.get("peak_flexion_velocity_deg_s"),
                "test_overall_mae_deg": s1.get("held_out_test_metrics", {}).get("mae_deg"),
                "test_overall_rmse_deg": s1.get("held_out_test_metrics", {}).get("rmse_deg"),
                "test_overall_r2": s1.get("held_out_test_metrics", {}).get("r2"),
                "test_rom_mae_deg": s1.get("held_out_test_metrics", {}).get("rom_mae_deg"),
                "val_mae_deg": s1.get("validation_metrics", {}).get("mae_deg"),
                "latency_ms": s1.get("efficiency", {}).get("inference_latency_ms_p50", m["total_inference_latency_ms"]),
                "parameters": s1.get("efficiency", {}).get("num_parameters"),
            })

            # Task 2
            t2 = m["task2_forearm_prosup"]
            s2 = tasks_summary.get("task_2_forearm_prosup", {}).get("models", {}).get(m_name, {})
            tables["task2_forearm_prosup"].append({
                "model_name": m_name,
                "current_angle_deg": t2.get("current_angle_deg"),
                "sample_mae_deg": t2.get("sample_mae_deg"),
                "sample_rmse_deg": t2.get("sample_rmse_deg"),
                "sample_r2": t2.get("sample_r2"),
                "predicted_rom_deg": t2["full_trial_rom_deg"],
                "repetition_rom_deg": t2.get("repetition_mean_rom_deg"),
                "rom_error_deg": t2.get("rom_error_deg"),
                "movement_smoothness_sparc": t2.get("movement_smoothness_sparc"),
                "peak_velocity_deg_s": t2.get("peak_velocity_deg_s"),
                "test_overall_mae_deg": s2.get("held_out_test_metrics", {}).get("mae_deg"),
                "test_overall_rmse_deg": s2.get("held_out_test_metrics", {}).get("rmse_deg"),
                "test_overall_r2": s2.get("held_out_test_metrics", {}).get("r2"),
                "val_mae_deg": s2.get("validation_metrics", {}).get("mae_deg"),
                "latency_ms": s2.get("efficiency", {}).get("inference_latency_ms_p50", m["total_inference_latency_ms"]),
                "parameters": s2.get("efficiency", {}).get("num_parameters"),
            })

            # Task 3
            t3 = m["task3_exercise_classification"]
            s3 = tasks_summary.get("task_3_7class_exercise", {}).get("models", {}).get(m_name, {})
            if not s3:
                s3 = tasks_summary.get("task_3_exercise_classification", {}).get("models", {}).get(m_name, {})
            tables["task3_exercise_classification"].append({
                "model_name": m_name,
                "predicted_class": t3["predicted_class_name"],
                "confidence_pct": t3["confidence_pct"],
                "is_correct": t3.get("is_correct"),
                "test_macro_f1": s3.get("held_out_test_metrics", {}).get("macro_f1"),
                "test_accuracy": s3.get("held_out_test_metrics", {}).get("overall_accuracy"),
                "val_macro_f1": s3.get("validation_metrics", {}).get("macro_f1"),
                "latency_ms": s3.get("efficiency", {}).get("inference_latency_ms_p50", m["total_inference_latency_ms"]),
                "parameters": s3.get("efficiency", {}).get("num_parameters"),
            })

            # Task 4
            t4 = m["task4_quality_classification"]
            s4 = tasks_summary.get("task_4_quality_classification", {}).get("models", {}).get(m_name, {})
            tables["task4_quality_classification"].append({
                "model_name": m_name,
                "predicted_quality": t4["predicted_quality_name"],
                "confidence_pct": t4["confidence_pct"],
                "is_correct": t4.get("is_correct"),
                "test_macro_f1": s4.get("held_out_test_metrics", {}).get("macro_f1"),
                "test_accuracy": s4.get("held_out_test_metrics", {}).get("overall_accuracy"),
                "val_macro_f1": s4.get("validation_metrics", {}).get("macro_f1"),
                "latency_ms": s4.get("efficiency", {}).get("inference_latency_ms_p50", m["total_inference_latency_ms"]),
                "parameters": s4.get("efficiency", {}).get("num_parameters"),
            })

        return tables
