"""Unified training, validation-based early stopping, checkpointing, and evaluation engine for the 4 models."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import (
    evaluate_classification_predictions,
    evaluate_regression_predictions,
    profile_model_efficiency,
)
from src.features.windowing import ChannelStandardScaler
from src.models.architectures import build_four_models


def set_reproducible_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _predict_in_batches(
    model: nn.Module,
    x_array: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Run batched forward pass on CPU without building autograd graphs."""
    model.eval()
    preds = []
    n = x_array.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = torch.from_numpy(x_array[start : start + batch_size])
            out = model(xb)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_and_evaluate_classification_track(
    track_id: str,
    track_title: str,
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_val_raw: np.ndarray,
    y_val: np.ndarray,
    x_test_raw: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    class_names: Sequence[str],
    project_root: Path,
    test_subject_ids: Optional[Sequence[str]] = None,
    test_group_labels: Optional[Sequence[str]] = None,
    max_epochs: int = 28,
    patience: int = 7,
    batch_size: int = 64,
    lr: float = 1.5e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and compare TCN, 1D_CNN, LSTM, and GRU on a classification benchmark track."""
    # 1. Fit channel-wise z-score scaler strictly on training windows
    scaler = ChannelStandardScaler()
    scaler.fit(x_train_raw, split_name="train")
    x_train = scaler.transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)
    x_test = scaler.transform(x_test_raw)

    in_channels = x_train.shape[2]
    window_len = x_train.shape[1]

    # Compute dampened inverse-frequency class weights strictly from y_train
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    cls_weights = torch.tensor(inv_sqrt / inv_sqrt.mean(), dtype=torch.float32)

    ckpt_dir = project_root / "models" / "checkpoints"
    logs_dir = project_root / "experiments" / "logs"
    metrics_dir = project_root / "experiments" / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    set_reproducible_seed(seed)
    models = build_four_models(
        in_channels=in_channels,
        task_mode="classification",
        num_classes=num_classes,
    )

    track_logs: Dict[str, Any] = {
        "track_id": track_id,
        "track_title": track_title,
        "task_mode": "classification",
        "num_classes": num_classes,
        "class_names": list(class_names),
        "window_shape": [int(window_len), int(in_channels)],
        "split_window_counts": {
            "train": int(len(x_train)),
            "val": int(len(x_val)),
            "test": int(len(x_test)),
        },
        "hyperparameters": {
            "max_epochs": max_epochs,
            "patience": patience,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
        },
        "models": {},
    }
    track_metrics: Dict[str, Any] = copy.deepcopy(track_logs)

    for model_name in ("TCN", "1D_CNN", "LSTM", "GRU"):
        set_reproducible_seed(seed)
        model = models[model_name]
        train_ds = TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train.astype(np.int64)),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=0.02)

        best_val_f1 = -1.0
        best_val_loss = float("inf")
        best_epoch = 0
        best_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        history: List[Dict[str, float]] = []

        t_start = time.perf_counter()
        for epoch in range(1, max_epochs + 1):
            model.train()
            running_loss = 0.0
            n_seen = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                running_loss += float(loss.item()) * len(xb)
                n_seen += len(xb)
            scheduler.step()
            train_loss = running_loss / max(n_seen, 1)

            # Evaluate on validation set
            val_logits = _predict_in_batches(model, x_val, batch_size=256)
            val_loss = float(
                F.cross_entropy(
                    torch.from_numpy(val_logits),
                    torch.from_numpy(y_val.astype(np.int64)),
                ).item()
            )
            val_preds = np.argmax(val_logits, axis=1)
            val_eval = evaluate_classification_predictions(
                y_val, val_preds, num_classes=num_classes, class_names=class_names
            )
            val_f1 = val_eval["macro_f1"]

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_macro_f1": round(val_f1, 4),
                "val_balanced_accuracy": round(val_eval["balanced_accuracy"], 4),
            })

            # Model selection strictly on validation Macro-F1 (tie-break on val_loss)
            improved = (val_f1 > best_val_f1 + 1e-4) or (
                abs(val_f1 - best_val_f1) <= 1e-4 and val_loss < best_val_loss
            )
            if improved:
                best_val_f1 = val_f1
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        train_duration_sec = round(time.perf_counter() - t_start, 2)

        # Restore best validation checkpoint and save to disk
        model.load_state_dict(best_state)
        ckpt_path = ckpt_dir / f"{track_id}_{model_name}_best.pt"
        torch.save(
            {
                "track_id": track_id,
                "model_name": model_name,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_f1,
                "state_dict": best_state,
                "scaler": scaler.to_dict(),
                "class_names": list(class_names),
            },
            ckpt_path,
        )

        # Evaluate best validation checkpoint on Val and held-out Test sets
        val_preds_best = np.argmax(_predict_in_batches(model, x_val, batch_size=256), axis=1)
        test_logits_best = _predict_in_batches(model, x_test, batch_size=256)
        test_preds_best = np.argmax(test_logits_best, axis=1)

        val_metrics = evaluate_classification_predictions(
            y_val, val_preds_best, num_classes=num_classes, class_names=class_names
        )
        test_metrics = evaluate_classification_predictions(
            y_test,
            test_preds_best,
            num_classes=num_classes,
            class_names=class_names,
            subject_ids=test_subject_ids,
            group_labels=test_group_labels,
        )
        eff_metrics = profile_model_efficiency(
            model,
            input_shape=(1, window_len, in_channels),
            checkpoint_path=ckpt_path,
        )
        eff_metrics["training_duration_sec"] = train_duration_sec
        eff_metrics["best_epoch"] = best_epoch
        eff_metrics["epochs_trained"] = len(history)

        track_logs["models"][model_name] = {
            "efficiency": eff_metrics,
            "history": history,
        }
        track_metrics["models"][model_name] = {
            "efficiency": eff_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        print(
            f"[{track_id}] {model_name:6s} | BestEp={best_epoch:02d} ({train_duration_sec:5.1f}s) | "
            f"Val Macro-F1={val_metrics['macro_f1']:.4f} | "
            f"Test Macro-F1={test_metrics['macro_f1']:.4f} | "
            f"Test BalAcc={test_metrics['balanced_accuracy']:.4f}"
        )

    with open(logs_dir / f"{track_id}_training_log.json", "w", encoding="utf-8") as f:
        json.dump(track_logs, f, indent=2)
    with open(metrics_dir / f"{track_id}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(track_metrics, f, indent=2)

    return track_metrics


def train_and_evaluate_regression_track(
    track_id: str,
    track_title: str,
    x_train_raw: np.ndarray,
    y_train_deg: np.ndarray,
    x_val_raw: np.ndarray,
    y_val_deg: np.ndarray,
    x_test_raw: np.ndarray,
    y_test_deg: np.ndarray,
    target_names: Sequence[str] = ("elbow_flex_ext_deg", "forearm_pro_sup_deg"),
    project_root: Path = Path("."),
    test_subject_ids: Optional[Sequence[str]] = None,
    test_task_names: Optional[Sequence[str]] = None,
    max_epochs: int = 30,
    patience: int = 7,
    batch_size: int = 64,
    lr: float = 1.5e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Train and compare TCN, 1D_CNN, LSTM, and GRU on a continuous angle regression track."""
    # 1. Fit input channel scaler strictly on x_train_raw
    x_scaler = ChannelStandardScaler()
    x_scaler.fit(x_train_raw, split_name="train")
    x_train = x_scaler.transform(x_train_raw)
    x_val = x_scaler.transform(x_val_raw)
    x_test = x_scaler.transform(x_test_raw)

    # 2. Fit target angle scaler strictly on y_train_deg so linear output heads start centered
    y_scaler = ChannelStandardScaler()
    y_scaler.fit(y_train_deg, split_name="train")
    y_train_norm = y_scaler.transform(y_train_deg)
    y_mean = y_scaler.mean_.astype(np.float32)
    y_std = y_scaler.std_.astype(np.float32)

    in_channels = x_train.shape[2]
    window_len = x_train.shape[1]
    num_targets = y_train_deg.shape[2]

    ckpt_dir = project_root / "models" / "checkpoints"
    logs_dir = project_root / "experiments" / "logs"
    metrics_dir = project_root / "experiments" / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    set_reproducible_seed(seed)
    models = build_four_models(
        in_channels=in_channels,
        task_mode="regression",
        num_reg_targets=num_targets,
    )

    track_logs: Dict[str, Any] = {
        "track_id": track_id,
        "track_title": track_title,
        "task_mode": "regression",
        "target_names": list(target_names),
        "window_shape": [int(window_len), int(in_channels)],
        "split_window_counts": {
            "train": int(len(x_train)),
            "val": int(len(x_val)),
            "test": int(len(x_test)),
        },
        "hyperparameters": {
            "max_epochs": max_epochs,
            "patience": patience,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
        },
        "models": {},
    }
    track_metrics: Dict[str, Any] = copy.deepcopy(track_logs)
    test_predictions_deg: Dict[str, np.ndarray] = {"y_true_deg": y_test_deg}

    for model_name in ("TCN", "1D_CNN", "LSTM", "GRU"):
        set_reproducible_seed(seed)
        model = models[model_name]
        train_ds = TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train_norm),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

        best_val_mae = float("inf")
        best_epoch = 0
        best_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        history: List[Dict[str, float]] = []

        t_start = time.perf_counter()
        for epoch in range(1, max_epochs + 1):
            model.train()
            running_loss = 0.0
            n_seen = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred_norm = model(xb)
                loss_pos = F.smooth_l1_loss(pred_norm, yb, beta=0.5)
                # Kinematic first-difference smoothness consistency
                diff_pred = pred_norm[:, 1:, :] - pred_norm[:, :-1, :]
                diff_true = yb[:, 1:, :] - yb[:, :-1, :]
                loss_vel = F.smooth_l1_loss(diff_pred, diff_true, beta=0.2)
                loss = loss_pos + 0.15 * loss_vel
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                running_loss += float(loss.item()) * len(xb)
                n_seen += len(xb)
            scheduler.step()
            train_loss = running_loss / max(n_seen, 1)

            # Evaluate on validation set in physical degrees
            val_pred_norm = _predict_in_batches(model, x_val, batch_size=256)
            val_pred_deg = val_pred_norm * y_std + y_mean
            val_eval = evaluate_regression_predictions(y_val_deg, val_pred_deg, target_names=target_names)
            val_mae = val_eval["mae_deg_overall"]

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_mae_deg": round(val_mae, 3),
                "val_rmse_deg": round(val_eval["rmse_deg_overall"], 3),
                "val_r2": round(val_eval["r2_overall"], 4),
            })

            if val_mae < best_val_mae - 1e-3:
                best_val_mae = val_mae
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        train_duration_sec = round(time.perf_counter() - t_start, 2)

        model.load_state_dict(best_state)
        ckpt_path = ckpt_dir / f"{track_id}_{model_name}_best.pt"
        torch.save(
            {
                "track_id": track_id,
                "model_name": model_name,
                "best_epoch": best_epoch,
                "best_val_mae_deg": best_val_mae,
                "state_dict": best_state,
                "x_scaler": x_scaler.to_dict(),
                "y_scaler": y_scaler.to_dict(),
                "target_names": list(target_names),
            },
            ckpt_path,
        )

        val_pred_best = _predict_in_batches(model, x_val, batch_size=256) * y_std + y_mean
        test_pred_best = _predict_in_batches(model, x_test, batch_size=256) * y_std + y_mean
        test_predictions_deg[model_name] = test_pred_best

        val_metrics = evaluate_regression_predictions(y_val_deg, val_pred_best, target_names=target_names)
        test_metrics = evaluate_regression_predictions(
            y_test_deg,
            test_pred_best,
            target_names=target_names,
            subject_ids=test_subject_ids,
            task_names=test_task_names,
        )
        eff_metrics = profile_model_efficiency(
            model,
            input_shape=(1, window_len, in_channels),
            checkpoint_path=ckpt_path,
        )
        eff_metrics["training_duration_sec"] = train_duration_sec
        eff_metrics["best_epoch"] = best_epoch
        eff_metrics["epochs_trained"] = len(history)

        track_logs["models"][model_name] = {
            "efficiency": eff_metrics,
            "history": history,
        }
        track_metrics["models"][model_name] = {
            "efficiency": eff_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        t0_name = target_names[0]
        t1_name = target_names[1] if len(target_names) > 1 else target_names[0]
        print(
            f"[{track_id}] {model_name:6s} | BestEp={best_epoch:02d} ({train_duration_sec:5.1f}s) | "
            f"Val MAE={val_metrics['mae_deg_overall']:5.2f}° | "
            f"Test MAE={test_metrics['mae_deg_overall']:5.2f}° "
            f"(Flex={test_metrics['per_target'][t0_name]['mae_deg']:5.2f}°, "
            f"ProSup={test_metrics['per_target'][t1_name]['mae_deg']:5.2f}°) | "
            f"ROM MAE={test_metrics['rom_mae_deg_overall']:5.2f}°"
        )

    with open(logs_dir / f"{track_id}_training_log.json", "w", encoding="utf-8") as f:
        json.dump(track_logs, f, indent=2)
    with open(metrics_dir / f"{track_id}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(track_metrics, f, indent=2)

    return track_metrics, test_predictions_deg
