"""Comprehensive Training and Benchmarking Runner for Unified Dataset v2.

Trains and evaluates 4 neural sequence architectures:
- TCN (Temporal Convolutional Network)
- 1D-CNN (1D Convolutional Neural Network)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

Across 4 clinical tasks:
1. Task 1: Continuous Elbow Flexion/Extension Angle Regression (deg) [ULTra-MoCap]
2. Task 2: Continuous Forearm Pronation/Supination Angle Regression (deg) [ULTra-MoCap]
3. Task 3: Upper-Limb Rehabilitation Exercise Classification (3-Class) [PHYTMO]
4. Task 4: Exercise Movement-Quality Classification (Binary) [PHYTMO]

Strict subject-independent 80/20 partitions with zero leakage and frozen train-only normalization.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from src.models.architectures import build_four_models

UNIFIED_V2_DIR = PROJECT_ROOT / "data" / "processed" / "rehab"
OUTPUT_MODELS_DIR = PROJECT_ROOT / "models" / "rehab" / "checkpoints"
OUTPUT_METRICS_DIR = PROJECT_ROOT / "experiments" / "rehab" / "metrics"
OUTPUT_LOGS_DIR = PROJECT_ROOT / "experiments" / "rehab" / "logs"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_regression_metrics(y_true_deg: np.ndarray, y_pred_deg: np.ndarray) -> Dict[str, Any]:
    mae = float(np.mean(np.abs(y_pred_deg - y_true_deg)))
    rmse = float(np.sqrt(np.mean((y_pred_deg - y_true_deg) ** 2)))
    ss_res = float(np.sum((y_true_deg - y_pred_deg) ** 2))
    ss_tot = float(np.sum((y_true_deg - np.mean(y_true_deg)) ** 2))
    r2 = 1.0 - (ss_res / max(ss_tot, 1e-8))

    flat_true = y_true_deg.reshape(-1)
    flat_pred = y_pred_deg.reshape(-1)
    if np.std(flat_true) > 1e-6 and np.std(flat_pred) > 1e-6:
        pearson_r = float(np.corrcoef(flat_true, flat_pred)[0, 1])
    else:
        pearson_r = 0.0

    # Window-level ROM
    true_rom = np.ptp(y_true_deg, axis=1)  # (N, D)
    pred_rom = np.ptp(y_pred_deg, axis=1)
    rom_mae = float(np.mean(np.abs(pred_rom - true_rom)))

    return {
        "mae_deg": round(mae, 3),
        "rmse_deg": round(rmse, 3),
        "r2": round(r2, 4),
        "pearson_r": round(pearson_r, 4),
        "rom_mae_deg": round(rom_mae, 3),
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist()

    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )

    per_class = {}
    for idx, cname in enumerate(class_names):
        per_class[cname] = {
            "precision": round(float(prec[idx]), 4),
            "recall": round(float(rec[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(supp[idx]),
        }

    return {
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "overall_accuracy": round(acc, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def measure_inference_efficiency(
    model: nn.Module,
    sample_shape: Tuple[int, int, int] = (1, 200, 12),
    device: str = "cpu",
    n_runs: int = 50,
) -> Dict[str, float]:
    model.eval()
    x = torch.randn(*sample_shape, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    return {
        "num_parameters": num_params,
        "inference_latency_ms_p50": round(p50, 3),
        "inference_latency_ms_p95": round(p95, 3),
    }


def train_regression_task(
    task_key: str,
    task_name: str,
    npz_prefix: str,
    target_name: str,
    max_epochs: int = 15,
    patience: int = 6,
    batch_size: int = 64,
    lr: float = 1.5e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"TRAINING {task_name.upper()} ({task_key})")
    print("=" * 80)

    train_npz = np.load(UNIFIED_V2_DIR / f"{npz_prefix}_train.npz", allow_pickle=True)
    val_npz = np.load(UNIFIED_V2_DIR / f"{npz_prefix}_val.npz", allow_pickle=True)
    test_npz = np.load(UNIFIED_V2_DIR / f"{npz_prefix}_test.npz", allow_pickle=True)

    x_train_raw, y_train_deg = train_npz["x"], train_npz["y"]
    x_val_raw, y_val_deg = val_npz["x"], val_npz["y"]
    x_test_raw, y_test_deg = test_npz["x"], test_npz["y"]

    with open(UNIFIED_V2_DIR / "split_manifest.json", "r", encoding="utf-8") as f:
        split_manifest = json.load(f)["tasks"][task_key]

    norm_cfg = split_manifest["normalization"]
    x_mean = np.array(norm_cfg["x_scaler"]["mean"], dtype=np.float32)
    x_std = np.maximum(np.array(norm_cfg["x_scaler"]["std"], dtype=np.float32), 1e-8)
    y_mean = np.array(norm_cfg["y_scaler"]["mean"], dtype=np.float32)
    y_std = np.maximum(np.array(norm_cfg["y_scaler"]["std"], dtype=np.float32), 1e-8)

    x_train = (x_train_raw - x_mean) / x_std
    y_train_norm = (y_train_deg - y_mean) / y_std
    x_val = (x_val_raw - x_mean) / x_std
    x_test = (x_test_raw - x_mean) / x_std

    window_len = x_train.shape[1]
    in_channels = x_train.shape[2]
    num_targets = y_train_deg.shape[2]

    results = {
        "task_name": task_name,
        "task_mode": "regression",
        "dataset": split_manifest["source_dataset"],
        "target_names": [target_name],
        "subject_splits": split_manifest["subjects"],
        "window_counts": split_manifest["window_counts"],
        "models": {},
    }

    models = build_four_models(in_channels=in_channels, task_mode="regression", num_reg_targets=num_targets)

    for model_name in ("TCN", "1D_CNN", "LSTM", "GRU"):
        set_seed(seed)
        model = models[model_name]

        train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_norm))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
        criterion = nn.SmoothL1Loss()

        best_val_mae = float("inf")
        best_epoch = 0
        best_state = None
        history = []
        epochs_no_improve = 0

        t0 = time.perf_counter()
        for epoch in range(1, max_epochs + 1):
            model.train()
            running_loss = 0.0
            n_seen = 0

            for bx, by in train_loader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

                running_loss += float(loss.item()) * len(bx)
                n_seen += len(bx)

            scheduler.step()
            train_loss = running_loss / max(n_seen, 1)

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred_norm = model(torch.from_numpy(x_val)).numpy()
                val_pred_deg = val_pred_norm * y_std + y_mean
                val_metrics = compute_regression_metrics(y_val_deg, val_pred_deg)

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_mae_deg": val_metrics["mae_deg"],
                "val_rmse_deg": val_metrics["rmse_deg"],
                "val_r2": val_metrics["r2"],
            })

            print(f"[{model_name:6s}] Epoch {epoch:2d}/{max_epochs} | Train Loss: {train_loss:.5f} | Val MAE: {val_metrics['mae_deg']:.2f}° | Val R²: {val_metrics['r2']:.4f}")

            if val_metrics["mae_deg"] < best_val_mae:
                best_val_mae = val_metrics["mae_deg"]
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, best val MAE: {best_val_mae:.2f}°)")
                    break

        train_time = round(time.perf_counter() - t0, 2)

        # Load best state dict
        model.load_state_dict(best_state)
        model.eval()

        # Checkpoint save
        ckpt_filename = f"rehab_{task_key}_{model_name}_best.pt"
        ckpt_path = OUTPUT_MODELS_DIR / ckpt_filename
        torch.save({
            "state_dict": best_state,
            "architecture": model_name,
            "task_mode": "regression",
            "task_key": task_key,
            "target_names": [target_name],
            "in_channels": in_channels,
            "num_outputs": num_targets,
            "window_len": window_len,
            "best_epoch": best_epoch,
            "best_val_mae_deg": best_val_mae,
        }, ckpt_path)

        ckpt_size_kb = round(ckpt_path.stat().st_size / 1024.0, 2)

        # Final Validation evaluation
        with torch.no_grad():
            final_val_pred_deg = model(torch.from_numpy(x_val)).numpy() * y_std + y_mean
            val_final_metrics = compute_regression_metrics(y_val_deg, final_val_pred_deg)

            # Held-out Test evaluation
            test_pred_deg = model(torch.from_numpy(x_test)).numpy() * y_std + y_mean
            test_metrics = compute_regression_metrics(y_test_deg, test_pred_deg)

        efficiency = measure_inference_efficiency(model, sample_shape=(1, window_len, in_channels))
        efficiency["checkpoint_size_kb"] = ckpt_size_kb
        efficiency["training_duration_sec"] = train_time
        efficiency["best_epoch"] = best_epoch

        results["models"][model_name] = {
            "checkpoint_path": str(ckpt_path.relative_to(PROJECT_ROOT)),
            "efficiency": efficiency,
            "validation_metrics": val_final_metrics,
            "held_out_test_metrics": test_metrics,
            "history": history,
        }

        print(f"--> {model_name} Finished: Best Val MAE = {val_final_metrics['mae_deg']:.2f}° | Test MAE = {test_metrics['mae_deg']:.2f}° | Test R² = {test_metrics['r2']:.4f} | ROM MAE = {test_metrics['rom_mae_deg']:.2f}°")

    task_json_path = OUTPUT_METRICS_DIR / f"{task_key}_comparison.json"
    with open(task_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def train_classification_task(
    task_key: str,
    task_name: str,
    npz_prefix: str,
    class_names: List[str],
    max_epochs: int = 15,
    patience: int = 6,
    batch_size: int = 64,
    lr: float = 1.5e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"TRAINING {task_name.upper()} ({task_key})")
    print("=" * 80)

    train_npz = np.load(UNIFIED_V2_DIR / f"{npz_prefix}_train.npz", allow_pickle=True)
    val_npz = np.load(UNIFIED_V2_DIR / f"{npz_prefix}_val.npz", allow_pickle=True)
    test_npz = np.load(UNIFIED_V2_DIR / f"{npz_prefix}_test.npz", allow_pickle=True)

    x_train_raw, y_train = train_npz["x"], train_npz["y"]
    x_val_raw, y_val = val_npz["x"], val_npz["y"]
    x_test_raw, y_test = test_npz["x"], test_npz["y"]

    with open(UNIFIED_V2_DIR / "split_manifest.json", "r", encoding="utf-8") as f:
        split_manifest = json.load(f)["tasks"][task_key]

    norm_cfg = split_manifest["normalization"]
    x_mean = np.array(norm_cfg["scaler"]["mean"], dtype=np.float32)
    x_std = np.maximum(np.array(norm_cfg["scaler"]["std"], dtype=np.float32), 1e-8)

    x_train = (x_train_raw - x_mean) / x_std
    x_val = (x_val_raw - x_mean) / x_std
    x_test = (x_test_raw - x_mean) / x_std

    window_len = x_train.shape[1]
    in_channels = x_train.shape[2]
    num_classes = len(class_names)

    # Dampened class weights strictly from y_train
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    cls_weights = torch.tensor(inv_sqrt / inv_sqrt.mean(), dtype=torch.float32)

    results = {
        "task_name": task_name,
        "task_mode": "classification",
        "dataset": split_manifest["source_dataset"],
        "classes": class_names,
        "subject_splits": split_manifest["subjects"],
        "window_counts": split_manifest["window_counts"],
        "models": {},
    }

    models = build_four_models(in_channels=in_channels, task_mode="classification", num_classes=num_classes)

    for model_name in ("TCN", "1D_CNN", "LSTM", "GRU"):
        set_seed(seed)
        model = models[model_name]

        train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss(weight=cls_weights)

        best_val_f1 = -1.0
        best_epoch = 0
        best_state = None
        history = []
        epochs_no_improve = 0

        t0 = time.perf_counter()
        for epoch in range(1, max_epochs + 1):
            model.train()
            running_loss = 0.0
            n_seen = 0

            for bx, by in train_loader:
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

                running_loss += float(loss.item()) * len(bx)
                n_seen += len(bx)

            scheduler.step()
            train_loss = running_loss / max(n_seen, 1)

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.from_numpy(x_val)).numpy()
                val_preds = np.argmax(val_logits, axis=1)
                val_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
                val_acc = float(accuracy_score(y_val, val_preds))

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_macro_f1": round(val_f1, 4),
                "val_accuracy": round(val_acc, 4),
            })

            print(f"[{model_name:6s}] Epoch {epoch:2d}/{max_epochs} | Train Loss: {train_loss:.5f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc*100:.2f}%")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, best val F1: {best_val_f1:.4f})")
                    break

        train_time = round(time.perf_counter() - t0, 2)

        model.load_state_dict(best_state)
        model.eval()

        ckpt_filename = f"rehab_{task_key}_{model_name}_best.pt"
        ckpt_path = OUTPUT_MODELS_DIR / ckpt_filename
        torch.save({
            "state_dict": best_state,
            "architecture": model_name,
            "task_mode": "classification",
            "task_key": task_key,
            "classes": class_names,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "window_len": window_len,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
        }, ckpt_path)

        ckpt_size_kb = round(ckpt_path.stat().st_size / 1024.0, 2)

        # Final Evaluations
        with torch.no_grad():
            final_val_preds = np.argmax(model(torch.from_numpy(x_val)).numpy(), axis=1)
            val_final_metrics = compute_classification_metrics(y_val, final_val_preds, class_names)

            test_preds = np.argmax(model(torch.from_numpy(x_test)).numpy(), axis=1)
            test_metrics = compute_classification_metrics(y_test, test_preds, class_names)

        efficiency = measure_inference_efficiency(model, sample_shape=(1, window_len, in_channels))
        efficiency["checkpoint_size_kb"] = ckpt_size_kb
        efficiency["training_duration_sec"] = train_time
        efficiency["best_epoch"] = best_epoch

        results["models"][model_name] = {
            "checkpoint_path": str(ckpt_path.relative_to(PROJECT_ROOT)),
            "efficiency": efficiency,
            "validation_metrics": val_final_metrics,
            "held_out_test_metrics": test_metrics,
            "history": history,
        }

        print(f"--> {model_name} Finished: Best Val F1 = {val_final_metrics['macro_f1']:.4f} | Test F1 = {test_metrics['macro_f1']:.4f} | Test Acc = {test_metrics['overall_accuracy']*100:.2f}%")

    task_json_path = OUTPUT_METRICS_DIR / f"{task_key}_comparison.json"
    with open(task_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def main() -> None:
    OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 80)
    print("# STAGE 4 UNIFIED MULTI-TASK TRAINING (TCN, 1D-CNN, LSTM, GRU)")
    print("#" * 80)

    # 1. Task 1: Continuous Elbow Flexion
    t1_res = train_regression_task(
        task_key="task_1_elbow_flexion",
        task_name="Continuous Elbow Flexion Angle Regression (deg)",
        npz_prefix="task1_elbow_flexion",
        target_name="elbow_flex_ext_deg",
        max_epochs=12,
        patience=5,
    )

    # 2. Task 2: Continuous Forearm Pronation/Supination
    t2_res = train_regression_task(
        task_key="task_2_forearm_prosup",
        task_name="Continuous Forearm Pronation/Supination Angle Regression (deg)",
        npz_prefix="task2_forearm_prosup",
        target_name="forearm_pro_sup_deg",
        max_epochs=12,
        patience=5,
    )

    # 3. Task 3: Upper-Limb Rehabilitation Exercise Classification (3-Class)
    t3_res = train_classification_task(
        task_key="task_3_exercise_classification",
        task_name="Rehabilitation Exercise Classification",
        npz_prefix="task3_exercise_classification",
        class_names=[
            "Elbow Flexion-Extension (EFE)",
            "Extension of Arms Overhead (EAH)",
            "Squeezing (SQZ)",
        ],
        max_epochs=12,
        patience=5,
    )

    # 4. Task 4: Exercise Movement-Quality Classification (Binary)
    t4_res = train_classification_task(
        task_key="task_4_quality_classification",
        task_name="Exercise Movement-Quality Classification",
        npz_prefix="task4_quality_classification",
        class_names=[
            "Correct Execution",
            "Incorrect / Compensatory Execution",
        ],
        max_epochs=12,
        patience=5,
    )

    # Master summary
    master_summary = {
        "dataset_version": "rehab",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task_1_elbow_flexion": t1_res,
        "task_2_forearm_prosup": t2_res,
        "task_3_exercise_classification": t3_res,
        "task_4_quality_classification": t4_res,
    }

    master_path = OUTPUT_METRICS_DIR / "rehab_master_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL 16 MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print(f"Master summary saved to: {master_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
