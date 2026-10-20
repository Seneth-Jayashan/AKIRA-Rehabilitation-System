"""Train and evaluate TCN, 1D-CNN, LSTM, and GRU on the Harmonized 7-Class Rehabilitation Dataset.

Strict subject-independent 80/20 train/val/test splits.
Saves checkpoints to models/checkpoints/task3_7class_exercise_{arch}.pt
Saves benchmark report to experiments/metrics/task3_7class_exercise_comparison.json
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.architectures import build_four_models

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"
METRICS_DIR = PROJECT_ROOT / "experiments" / "metrics"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Elbow Flexion-Extension",
    "Forearm Pronation-Supination",
    "Shoulder Flexion / Overhead Reach",
    "Shoulder Abduction / Scaption",
    "Shoulder Internal-External Rotation",
    "Crossbody Reach / Horizontal Adduction",
    "Functional Drinking / Reach-to-Grasp",
]

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_split(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    path = PROCESSED_DIR / f"task3_7class_exercise_{split}.npz"
    with np.load(path) as data:
        x = torch.from_numpy(data["x"]).float()
        y = torch.from_numpy(data["y"]).long()
    return x, y


def train_and_eval_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> Dict[str, Any]:
    print(f"\n--- Training {model_name} (7-Class Classifier) ---")
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_f1 = -1.0
    best_state = None
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item() * len(by)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(DEVICE)
                logits = model(bx)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(by.numpy())

        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_targets, val_preds)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == EPOCHS:
            train_loss = total_loss / len(train_loader.dataset)
            print(f"  Epoch {epoch:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.1f}% | Val Macro-F1: {val_f1*100:.1f}%")

    train_time_sec = round(time.time() - t0, 2)
    model.load_state_dict(best_state)
    model.eval()

    # Final evaluation on held-out test split
    test_preds, test_targets = [], []
    inference_times = []

    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(DEVICE)
            t_start = time.perf_counter()
            logits = model(bx)
            t_infer = (time.perf_counter() - t_start) * 1000.0 / len(bx)  # ms per sample
            inference_times.append(t_infer)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(by.numpy())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)

    test_acc = float(accuracy_score(test_targets, test_preds))
    test_bal_acc = float(balanced_accuracy_score(test_targets, test_preds))
    test_macro_f1 = float(f1_score(test_targets, test_preds, average="macro", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        test_targets, test_preds, labels=list(range(7)), zero_division=0
    )

    per_class_metrics = {}
    for idx, cname in enumerate(CLASSES):
        per_class_metrics[cname] = {
            "precision": round(float(precision[idx]), 4),
            "recall": round(float(recall[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(support[idx]),
        }

    conf_mat = confusion_matrix(test_targets, test_preds, labels=list(range(7))).tolist()

    num_params = sum(p.numel() for p in model.parameters())
    p50_latency = float(np.median(inference_times))

    print(f"  -> Test Macro-F1: {test_macro_f1*100:.2f}% | Test Accuracy: {test_acc*100:.2f}% | P50 Latency: {p50_latency:.2f} ms")

    # Save Checkpoint
    ckpt_path = CHECKPOINTS_DIR / f"task3_7class_exercise_{model_name.lower()}.pt"
    ckpt_payload = {
        "model_name": model_name,
        "task_name": "Harmonized 7-Class Upper-Limb Rehabilitation Exercise Classification",
        "task_mode": "classification",
        "num_classes": 7,
        "classes": CLASSES,
        "in_channels": 12,
        "state_dict": best_state,
        "metrics": {
            "val_best_macro_f1": round(best_val_f1, 4),
            "test_accuracy": round(test_acc, 4),
            "test_balanced_accuracy": round(test_bal_acc, 4),
            "test_macro_f1": round(test_macro_f1, 4),
        },
    }
    torch.save(ckpt_payload, ckpt_path)
    print(f"  Saved checkpoint to: {ckpt_path.name}")

    return {
        "model_name": model_name,
        "validation_metrics": {
            "best_macro_f1": round(best_val_f1, 4),
        },
        "held_out_test_metrics": {
            "overall_accuracy": round(test_acc, 4),
            "balanced_accuracy": round(test_bal_acc, 4),
            "macro_f1": round(test_macro_f1, 4),
            "per_class": per_class_metrics,
            "confusion_matrix": conf_mat,
        },
        "efficiency": {
            "num_parameters": num_params,
            "inference_latency_ms_p50": round(p50_latency, 3),
            "training_time_sec": train_time_sec,
        },
    }


def main():
    print("=" * 70)
    print("Training 4 AI Architectures on Harmonized 7-Class Dataset")
    print("=" * 70)
    set_seed(42)

    X_train, y_train = load_dataset_split("train")
    X_val, y_val = load_dataset_split("val")
    X_test, y_test = load_dataset_split("test")

    print(f"Loaded: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    models_dict = build_four_models(in_channels=12, task_mode="classification", num_classes=7)

    results = {}
    best_model_name = None
    best_f1 = -1.0

    for m_name in ["TCN", "1D_CNN", "LSTM", "GRU"]:
        model = models_dict[m_name]
        res = train_and_eval_model(m_name, model, train_loader, val_loader, test_loader)
        results[m_name] = res

        f1 = res["held_out_test_metrics"]["macro_f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = m_name

    summary = {
        "task_name": "Harmonized 7-Class Upper-Limb Rehabilitation Exercise Classification",
        "num_classes": 7,
        "classes": CLASSES,
        "best_model": best_model_name,
        "models": results,
    }

    out_metrics_path = METRICS_DIR / "task3_7class_exercise_comparison.json"
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"7-CLASS BENCHMARK COMPLETED | BEST MODEL: {best_model_name} (Macro-F1: {best_f1*100:.2f}%)")
    print(f"Report saved to: {out_metrics_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
