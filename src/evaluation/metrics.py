"""Evaluation metrics for classification, continuous joint-angle regression, ROM estimation, and model profiling."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    r2_score,
)

from src.features.windowing import compute_rom_deg


def evaluate_classification_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: Optional[Sequence[str]] = None,
    subject_ids: Optional[Sequence[str]] = None,
    group_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute Macro-F1, Balanced Accuracy, Overall Accuracy, Per-Class P/R/F1, Confusion Matrix, and Subject/Group metrics."""
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    labels = list(range(num_classes))

    macro_f1 = float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(yt, yp))
    acc = float(accuracy_score(yt, yp))

    prec, rec, f1s, sup = precision_recall_fscore_support(
        yt, yp, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(yt, yp, labels=labels)

    names = list(class_names) if class_names is not None else [f"class_{i}" for i in labels]
    per_class = {}
    for idx, name in enumerate(names):
        per_class[name] = {
            "class_id": int(idx),
            "precision": round(float(prec[idx]), 4),
            "recall": round(float(rec[idx]), 4),
            "f1": round(float(f1s[idx]), 4),
            "support": int(sup[idx]),
        }

    result: Dict[str, Any] = {
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "overall_accuracy": round(acc, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }

    if subject_ids is not None and len(subject_ids) == len(yt):
        subj_arr = np.asarray(subject_ids)
        by_subject = {}
        for s in sorted(set(subj_arr.tolist())):
            mask = subj_arr == s
            s_acc = float(accuracy_score(yt[mask], yp[mask]))
            s_f1 = float(f1_score(yt[mask], yp[mask], average="macro", zero_division=0))
            by_subject[str(s)] = {
                "accuracy": round(s_acc, 4),
                "macro_f1": round(s_f1, 4),
                "n_windows": int(mask.sum()),
            }
        result["by_subject"] = by_subject

    if group_labels is not None and len(group_labels) == len(yt):
        grp_arr = np.asarray(group_labels)
        by_group = {}
        for g in sorted(set(grp_arr.tolist())):
            mask = grp_arr == g
            g_acc = float(accuracy_score(yt[mask], yp[mask]))
            g_f1 = float(f1_score(yt[mask], yp[mask], average="macro", zero_division=0))
            by_group[str(g)] = {
                "accuracy": round(g_acc, 4),
                "macro_f1": round(g_f1, 4),
                "n_windows": int(mask.sum()),
            }
        result["by_group"] = by_group

    return result


def _safe_pearson_r(y_true_1d: np.ndarray, y_pred_1d: np.ndarray) -> float:
    if np.std(y_true_1d) < 1e-8 or np.std(y_pred_1d) < 1e-8:
        return 0.0
    r = np.corrcoef(y_true_1d, y_pred_1d)[0, 1]
    return 0.0 if np.isnan(r) else float(r)


def evaluate_regression_predictions(
    y_true_nwd: np.ndarray,
    y_pred_nwd: np.ndarray,
    target_names: Sequence[str] = ("elbow_flex_ext_deg", "forearm_pro_sup_deg"),
    subject_ids: Optional[Sequence[str]] = None,
    task_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute MAE (deg), RMSE (deg), R^2, Pearson r, and Window ROM MAE (deg) overall, per target, per subject, and per task."""
    yt = np.asarray(y_true_nwd, dtype=np.float64)
    yp = np.asarray(y_pred_nwd, dtype=np.float64)
    if yt.shape != yp.shape or yt.ndim != 3:
        raise ValueError(f"Expected matching 3D (N, W, D) arrays, got {yt.shape} vs {yp.shape}")

    err = yp - yt
    abs_err = np.abs(err)
    mae_overall = float(np.mean(abs_err))
    rmse_overall = float(np.sqrt(np.mean(err ** 2)))

    per_target = {}
    r2_list = []
    r_list = []
    rom_mae_list = []

    for d_idx, t_name in enumerate(target_names):
        yt_d = yt[:, :, d_idx]
        yp_d = yp[:, :, d_idx]
        yt_flat = yt_d.reshape(-1)
        yp_flat = yp_d.reshape(-1)

        mae_d = float(np.mean(np.abs(yp_flat - yt_flat)))
        rmse_d = float(np.sqrt(np.mean((yp_flat - yt_flat) ** 2)))
        r2_d = float(r2_score(yt_flat, yp_flat)) if np.var(yt_flat) > 1e-6 else 0.0
        r_d = _safe_pearson_r(yt_flat, yp_flat)

        # Window-level Range of Motion (max - min within each window) error in degrees
        rom_true = compute_rom_deg(yt_d, axis=1)
        rom_pred = compute_rom_deg(yp_d, axis=1)
        rom_mae_d = float(np.mean(np.abs(rom_pred - rom_true)))

        r2_list.append(r2_d)
        r_list.append(r_d)
        rom_mae_list.append(rom_mae_d)

        per_target[t_name] = {
            "mae_deg": round(mae_d, 3),
            "rmse_deg": round(rmse_d, 3),
            "r2": round(r2_d, 4),
            "pearson_r": round(r_d, 4),
            "rom_mae_deg": round(rom_mae_d, 3),
        }

    result: Dict[str, Any] = {
        "mae_deg_overall": round(mae_overall, 3),
        "rmse_deg_overall": round(rmse_overall, 3),
        "r2_overall": round(float(np.mean(r2_list)), 4),
        "pearson_r_overall": round(float(np.mean(r_list)), 4),
        "rom_mae_deg_overall": round(float(np.mean(rom_mae_list)), 3),
        "per_target": per_target,
    }

    if subject_ids is not None and len(subject_ids) == yt.shape[0]:
        subj_arr = np.asarray(subject_ids)
        by_subject = {}
        for s in sorted(set(subj_arr.tolist())):
            mask = subj_arr == s
            by_subject[str(s)] = {
                "mae_deg": round(float(np.mean(abs_err[mask])), 3),
                "rmse_deg": round(float(np.sqrt(np.mean(err[mask] ** 2))), 3),
                "elbow_flex_mae_deg": round(float(np.mean(abs_err[mask, :, 0])), 3),
                "pro_sup_mae_deg": round(float(np.mean(abs_err[mask, :, 1])), 3) if yt.shape[2] > 1 else None,
                "n_windows": int(mask.sum()),
            }
        result["by_subject"] = by_subject

    if task_names is not None and len(task_names) == yt.shape[0]:
        task_arr = np.asarray(task_names)
        by_task = {}
        for t in sorted(set(task_arr.tolist())):
            mask = task_arr == t
            by_task[str(t)] = {
                "mae_deg": round(float(np.mean(abs_err[mask])), 3),
                "rmse_deg": round(float(np.sqrt(np.mean(err[mask] ** 2))), 3),
                "elbow_flex_mae_deg": round(float(np.mean(abs_err[mask, :, 0])), 3),
                "pro_sup_mae_deg": round(float(np.mean(abs_err[mask, :, 1])), 3) if yt.shape[2] > 1 else None,
                "n_windows": int(mask.sum()),
            }
        result["by_task"] = by_task

    return result


def profile_model_efficiency(
    model: nn.Module,
    input_shape: Sequence[int] = (1, 200, 12),
    checkpoint_path: Optional[Path] = None,
    n_warmup: int = 5,
    n_runs: int = 25,
) -> Dict[str, Any]:
    """Measure trainable parameter count, saved checkpoint size (KB), and single-window CPU latency (ms)."""
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy = torch.randn(*input_shape, dtype=torch.float32)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
        latencies_ms = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    lat_arr = np.asarray(latencies_ms, dtype=np.float64)
    ckpt_kb = round(checkpoint_path.stat().st_size / 1024.0, 2) if checkpoint_path and checkpoint_path.exists() else 0.0

    return {
        "num_parameters": int(n_params),
        "checkpoint_size_kb": ckpt_kb,
        "inference_latency_ms_p50": round(float(np.percentile(lat_arr, 50)), 3),
        "inference_latency_ms_p95": round(float(np.percentile(lat_arr, 95)), 3),
    }
