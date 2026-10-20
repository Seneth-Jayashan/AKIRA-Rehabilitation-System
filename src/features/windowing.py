"""Leakage-free sliding window segmentation, train-only z-score normalization, and ROM utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ChannelStandardScaler:
    """Channel-wise z-score scaler fitted strictly on training partition arrays."""

    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None
    eps: float = 1e-8
    fitted_on_split: str = "train"

    def fit(self, x_train: np.ndarray, split_name: str = "train") -> "ChannelStandardScaler":
        """Fit channel-wise mean and std strictly on training data of shape (N, W, C) or (T, C)."""
        if split_name != "train":
            raise ValueError(
                f"Leakage Guard: ChannelStandardScaler.fit() may ONLY be called on 'train' data, got {split_name!r}."
            )
        arr = np.asarray(x_train, dtype=np.float64)
        if arr.ndim == 3:
            flat = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim == 2:
            flat = arr
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

        if np.isnan(flat).any() or np.isinf(flat).any():
            raise ValueError("Cannot fit scaler on data containing NaN or Inf.")

        self.mean_ = flat.mean(axis=0)
        self.std_ = np.maximum(flat.std(axis=0), self.eps)
        self.fitted_on_split = split_name
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply frozen training channel mean and std to an input array."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("ChannelStandardScaler must be fitted on training data before transform().")
        arr = np.asarray(x, dtype=np.float64)
        return ((arr - self.mean_) / self.std_).astype(np.float32)

    def to_dict(self) -> dict:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler not fitted.")
        return {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "eps": self.eps,
            "fitted_on_split": self.fitted_on_split,
        }


def extract_trial_windows(
    signal_tc: np.ndarray,
    window_size: int,
    stride: int,
    continuous_target_td: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract sliding windows strictly within a single trial (never crossing trial boundaries).

    Args:
        signal_tc: Array of shape (T, C).
        window_size: Window length W in samples.
        stride: Window step size S in samples.
        continuous_target_td: Optional continuous target of shape (T, D) in degrees.

    Returns:
        x_windows of shape (N_win, W, C) and optional y_windows of shape (N_win, W, D).
    """
    arr = np.asarray(signal_tc, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"signal_tc must be 2D (T, C), got {arr.shape}")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive integers.")

    t_len, n_ch = arr.shape
    if continuous_target_td is not None:
        tgt = np.asarray(continuous_target_td, dtype=np.float32)
        if tgt.shape[0] != t_len:
            raise ValueError(f"Target length {tgt.shape[0]} != signal length {t_len}")
    else:
        tgt = None

    if t_len < window_size:
        return (
            np.empty((0, window_size, n_ch), dtype=np.float32),
            np.empty((0, window_size, tgt.shape[1]), dtype=np.float32) if tgt is not None else None,
        )

    starts = range(0, t_len - window_size + 1, stride)
    x_win = np.stack([arr[s : s + window_size] for s in starts], axis=0)
    y_win = np.stack([tgt[s : s + window_size] for s in starts], axis=0) if tgt is not None else None
    return x_win, y_win


def compute_rom_deg(angle_trajectory_deg: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute Range of Motion (max - min, in degrees) along a time axis."""
    arr = np.asarray(angle_trajectory_deg, dtype=np.float64)
    return np.max(arr, axis=axis) - np.min(arr, axis=axis)
