"""Subject-independent split loading, validation, and leakage prevention utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


DEFAULT_CONFIG_PATH = Path(r"D:\me\project\configs\benchmark_tracks.json")


def load_benchmark_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load the master benchmark tracks configuration JSON."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_disjoint_subject_splits(
    train_subjects: Sequence[str],
    val_subjects: Sequence[str],
    test_subjects: Sequence[str],
    dataset_name: str = "dataset",
) -> None:
    """Assert that train, val, and test subject sets are non-empty and pairwise disjoint."""
    s_train = set(train_subjects)
    s_val = set(val_subjects)
    s_test = set(test_subjects)

    if not s_train or not s_val or not s_test:
        raise ValueError(f"[{dataset_name}] Empty partition detected: train={len(s_train)}, val={len(s_val)}, test={len(s_test)}")

    if len(s_train) != len(train_subjects) or len(s_val) != len(val_subjects) or len(s_test) != len(test_subjects):
        raise ValueError(f"[{dataset_name}] Duplicate subject IDs inside a single split list.")

    overlap_tv = s_train & s_val
    overlap_tt = s_train & s_test
    overlap_vt = s_val & s_test

    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            f"[{dataset_name}] Subject leakage detected! "
            f"train&val={overlap_tv}, train&test={overlap_tt}, val&test={overlap_vt}"
        )


def verify_trial_partition_isolation(
    records: Iterable[dict],
    subject_key: str = "subject_id",
    trial_key: str = "trial_id",
    split_key: str = "split",
) -> Dict[str, Set[str]]:
    """Verify that no subject ID or trial ID appears in more than one split."""
    split_subjects: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    split_trials: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

    for rec in records:
        sp = rec[split_key]
        if sp not in split_subjects:
            continue
        split_subjects[sp].add(str(rec[subject_key]))
        split_trials[sp].add(str(rec[trial_key]))

    verify_disjoint_subject_splits(
        list(split_subjects["train"]),
        list(split_subjects["val"]),
        list(split_subjects["test"]),
        dataset_name="trial_manifest_subjects",
    )

    trial_overlap = (
        (split_trials["train"] & split_trials["val"])
        | (split_trials["train"] & split_trials["test"])
        | (split_trials["val"] & split_trials["test"])
    )
    if trial_overlap:
        raise ValueError(f"Trial leakage across partitions: {sorted(trial_overlap)[:5]}")

    return split_subjects
