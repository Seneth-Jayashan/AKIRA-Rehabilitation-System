"""Generate confusion matrices, learning curves, and continuous angle trajectory plots in outputs/."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_track_confusion_matrices_plot(
    track_metrics: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Save a 2x2 figure showing held-out test confusion matrices for TCN, 1D_CNN, LSTM, and GRU."""
    output_dir.mkdir(parents=True, exist_ok=True)
    track_id = track_metrics["track_id"]
    class_names = track_metrics["class_names"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(f"{track_metrics['track_title']} — Test Confusion Matrices", fontsize=13, fontweight="bold")

    for ax, model_name in zip(axes.flat, ("TCN", "1D_CNN", "LSTM", "GRU")):
        m_data = track_metrics["models"][model_name]["test"]
        cm = np.asarray(m_data["confusion_matrix"], dtype=np.int64)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(
            f"{model_name} | Macro-F1={m_data['macro_f1']:.3f} | BalAcc={m_data['balanced_accuracy']:.3f}",
            fontsize=11,
        )
        ticks = np.arange(len(class_names))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        short_names = [str(c)[:12] for c in class_names]
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_ylabel("True Label", fontsize=9)
        ax.set_xlabel("Predicted Label", fontsize=9)

        if len(class_names) <= 10:
            thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=8,
                    )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path = output_dir / f"{track_id}_confusion_matrices.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def save_track_learning_curves_plot(
    track_log_path: Path,
    output_dir: Path,
) -> Path:
    """Save training loss and validation performance learning curves across epochs for the 4 models."""
    import json

    with open(track_log_path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    track_id = log_data["track_id"]
    task_mode = log_data["task_mode"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"{log_data['track_title']} — Learning Curves", fontsize=12, fontweight="bold")

    colors = {"TCN": "#1f77b4", "1D_CNN": "#ff7f0e", "LSTM": "#2ca02c", "GRU": "#d62728"}
    for model_name in ("TCN", "1D_CNN", "LSTM", "GRU"):
        hist = log_data["models"][model_name]["history"]
        epochs = [h["epoch"] for h in hist]
        tr_loss = [h["train_loss"] for h in hist]
        ax1.plot(epochs, tr_loss, marker="o", markersize=3, label=model_name, color=colors[model_name])

        if task_mode == "classification":
            val_metric = [h["val_macro_f1"] for h in hist]
            ax2.set_ylabel("Validation Macro-F1")
            ax2.set_title("Validation Macro-F1 vs. Epoch")
        else:
            val_metric = [h["val_mae_deg"] for h in hist]
            ax2.set_ylabel("Validation MAE (deg)")
            ax2.set_title("Validation MAE (deg) vs. Epoch")
        ax2.plot(epochs, val_metric, marker="s", markersize=3, label=model_name, color=colors[model_name])

    ax1.set_title("Training Loss vs. Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("Epoch")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    out_path = output_dir / f"{track_id}_learning_curves.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def save_angle_trajectory_comparison_plot(
    track_id: str,
    track_title: str,
    test_predictions_deg: Dict[str, np.ndarray],
    target_names: Sequence[str],
    output_dir: Path,
    n_windows_to_stitch: int = 5,
) -> Path:
    """Plot continuous Ground-Truth vs. TCN/1D_CNN/LSTM/GRU predicted joint angles (deg) on held-out test windows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = test_predictions_deg["y_true_deg"]
    # Pick windows with largest elbow flexion dynamic range for clear visual inspection
    roms = np.ptp(y_true[:, :, 0], axis=1)
    top_idx = int(np.argmax(roms))
    start_idx = max(0, top_idx - n_windows_to_stitch // 2)
    end_idx = min(len(y_true), start_idx + n_windows_to_stitch)

    yt_concat = y_true[start_idx:end_idx].reshape(-1, y_true.shape[2])
    t_steps = np.arange(len(yt_concat))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    fig.suptitle(f"{track_title} — Held-Out Test Joint Angle Tracking (deg)", fontsize=12, fontweight="bold")

    colors = {"TCN": "#1f77b4", "1D_CNN": "#ff7f0e", "LSTM": "#2ca02c", "GRU": "#d62728"}
    for d_idx in range(min(2, y_true.shape[2])):
        ax = axes[d_idx]
        ax.plot(t_steps, yt_concat[:, d_idx], color="black", linewidth=2.0, label="Ground Truth (MoCap/IK)")
        for model_name in ("TCN", "1D_CNN", "LSTM", "GRU"):
            yp = test_predictions_deg[model_name][start_idx:end_idx].reshape(-1, y_true.shape[2])
            ax.plot(t_steps, yp[:, d_idx], linewidth=1.3, alpha=0.85, label=model_name, color=colors[model_name])
        ax.set_ylabel(f"{target_names[d_idx]} (deg)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=5, fontsize=8)

    axes[-1].set_xlabel("Sample Index Across Held-Out Test Windows")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    out_path = output_dir / f"{track_id}_angle_trajectories.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
