"""Tests for standardized dataset, subject-independent splits, and samples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.schemas import CANONICAL_12_CHANNELS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"


def test_split_manifest_exists_and_valid():
    manifest_path = PROCESSED_DIR / "split_manifest.json"
    assert manifest_path.exists(), "split_manifest.json must exist in data/processed"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = data.get("tasks", {})
    assert len(tasks) >= 4, f"Expected at least 4 tasks in manifest, found {len(tasks)}"

    for t_key, t_info in tasks.items():
        if "subjects" in t_info and isinstance(t_info["subjects"], dict):
            splits = t_info["subjects"]
            if "train" in splits and "val" in splits and "test" in splits:
                train_sub = set(splits.get("train", []))
                val_sub = set(splits.get("val", []))
                test_sub = set(splits.get("test", []))

                # Strict subject disjointness check
                assert train_sub.isdisjoint(val_sub), f"{t_key}: train and val subjects overlap!"
                assert train_sub.isdisjoint(test_sub), f"{t_key}: train and test subjects overlap!"
                assert val_sub.isdisjoint(test_sub), f"{t_key}: val and test subjects overlap!"


def test_processed_npz_files_exist_and_match_canonical_channels():
    for task_num in [1, 2, 3, 4]:
        task_prefixes = {
            1: "task1_elbow_flexion",
            2: "task2_forearm_prosup",
            3: "task3_exercise_classification",
            4: "task4_quality_classification",
        }
        prefix = task_prefixes[task_num]
        for split in ["train", "val", "test"]:
            npz_path = PROCESSED_DIR / f"{prefix}_{split}.npz"
            assert npz_path.exists(), f"Missing {npz_path.name}"
            with np.load(npz_path) as data:
                assert "x" in data, f"Missing x in {npz_path.name}"
                assert "y" in data, f"Missing y in {npz_path.name}"
                X = data["x"]
                assert X.ndim == 3, f"Expected 3D array (N, T, C), got {X.shape}"
                assert X.shape[1] == 200, f"Expected window length 200 (2.0s), got {X.shape[1]}"
                assert X.shape[2] == 12, f"Expected 12 canonical channels, got {X.shape[2]}"


def test_7class_harmonized_dataset():
    manifest_p = PROCESSED_DIR / "task3_7class_manifest.json"
    assert manifest_p.exists(), "task3_7class_manifest.json must exist in data/processed"
    with open(manifest_p, "r", encoding="utf-8") as f:
        m7 = json.load(f)
    assert m7["num_classes"] == 7
    assert len(m7["classes"]) == 7

    for split in ["train", "val", "test"]:
        p = PROCESSED_DIR / f"task3_7class_exercise_{split}.npz"
        assert p.exists(), f"Missing {p.name}"
        with np.load(p) as d:
            assert "x" in d and "y" in d
            assert d["x"].shape[1] == 200
            assert d["x"].shape[2] == 12
            # All 7 classes should be present in each split
            classes_present = set(np.unique(d["y"]).tolist())
            assert len(classes_present) == 7, f"Not all 7 classes present in {split}: {classes_present}"


def test_sample_files_and_manifest():
    manifest_p = SAMPLES_DIR / "manifest.json"
    assert manifest_p.exists(), "Missing data/samples/manifest.json"

    with open(manifest_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", [])
    assert len(samples) >= 12, f"Expected at least 12 test samples, found {len(samples)}"

    canonical_cols = [
        "RUA_Ax", "RUA_Ay", "RUA_Az", "RUA_Gx", "RUA_Gy", "RUA_Gz",
        "RLA_Ax", "RLA_Ay", "RLA_Az", "RLA_Gx", "RLA_Gy", "RLA_Gz",
    ]

    for s in samples:
        csv_file = SAMPLES_DIR / s["filename"]
        assert csv_file.exists(), f"Sample CSV missing: {s['filename']}"
        with open(csv_file, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            for col in canonical_cols:
                assert col in header, f"Column {col} missing in {s['filename']}"
