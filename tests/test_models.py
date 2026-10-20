"""Tests for the model architectures and trained checkpoints (including 7-class expansion)."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from src.models.architectures import build_four_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"

ARCHITECTURES = ["TCN", "1D_CNN", "LSTM", "GRU"]
TASKS = [
    "task1_elbow_flexion",
    "task2_forearm_prosup",
    "task3_exercise_classification",
    "task4_quality_classification",
]


def test_build_four_models_regression():
    models = build_four_models(in_channels=12, task_mode="regression", num_reg_targets=1)
    x = torch.randn(2, 200, 12)
    for name in ARCHITECTURES:
        assert name in models, f"Model {name} missing in build_four_models"
        model = models[name]
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 200, 1), f"{name} regression shape mismatch: {out.shape}"


def test_build_four_models_classification():
    for nc in [3, 7]:
        models = build_four_models(in_channels=12, task_mode="classification", num_classes=nc)
        x = torch.randn(2, 200, 12)
        for name in ARCHITECTURES:
            assert name in models, f"Model {name} missing in build_four_models"
            model = models[name]
            model.eval()
            with torch.no_grad():
                out = model(x)
            assert out.shape == (2, nc), f"{name} {nc}-class shape mismatch: {out.shape}"


def test_legacy_16_checkpoints_exist_and_load():
    for task in TASKS:
        for arch in ARCHITECTURES:
            ckpt_path = CHECKPOINTS_DIR / f"{task}_{arch.lower()}.pt"
            assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path.name}"

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            assert "state_dict" in ckpt, f"No state_dict in {ckpt_path.name}"
            assert "task_mode" in ckpt, f"No task_mode in {ckpt_path.name}"

            task_mode = ckpt["task_mode"]
            in_channels = ckpt.get("in_channels", 12)
            if task_mode == "regression":
                models = build_four_models(in_channels=in_channels, task_mode="regression", num_reg_targets=1)
            else:
                num_classes = ckpt.get("num_classes", 2)
                models = build_four_models(in_channels=in_channels, task_mode="classification", num_classes=num_classes)

            model = models[arch]
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            x = torch.randn(1, 200, 12)
            with torch.no_grad():
                out = model(x)
            assert out is not None


def test_7class_checkpoints_exist_and_load():
    for arch in ARCHITECTURES:
        ckpt_path = CHECKPOINTS_DIR / f"task3_7class_exercise_{arch.lower()}.pt"
        assert ckpt_path.exists(), f"Missing 7-class checkpoint: {ckpt_path.name}"

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt.get("num_classes") == 7
        assert len(ckpt.get("classes", [])) == 7

        models = build_four_models(in_channels=12, task_mode="classification", num_classes=7)
        model = models[arch]
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        x = torch.randn(1, 200, 12)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 7)
