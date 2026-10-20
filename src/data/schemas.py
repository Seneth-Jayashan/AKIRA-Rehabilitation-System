"""Canonical sensor channel definitions, unit conversions, and taxonomy mappings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

STANDARD_GRAVITY_MS2: float = 9.80665
DEG_TO_RAD: float = math.pi / 180.0
RAD_TO_DEG: float = 180.0 / math.pi

CANONICAL_12_CHANNELS: Tuple[str, ...] = (
    "ua_acc_x_ms2",
    "ua_acc_y_ms2",
    "ua_acc_z_ms2",
    "ua_gyr_x_rads",
    "ua_gyr_y_rads",
    "ua_gyr_z_rads",
    "fa_acc_x_ms2",
    "fa_acc_y_ms2",
    "fa_acc_z_ms2",
    "fa_gyr_x_rads",
    "fa_gyr_y_rads",
    "fa_gyr_z_rads",
)

# PHYTMO upper-limb exercise and execution-quality label definitions
PHYTMO_3CLASS_MAP: Dict[str, int] = {
    "EFE": 0,  # Elbow Flexion-Extension
    "EAH": 1,  # Extension of Arms Overhead
    "SQZ": 2,  # Squeezing
}

PHYTMO_6CLASS_MAP: Dict[str, int] = {
    "EFE0": 0,  # Elbow Flexion-Extension (Correct)
    "EFE1": 1,  # Elbow Flexion-Extension (Incorrect)
    "EAH0": 2,  # Extension of Arms Overhead (Correct)
    "EAH1": 3,  # Extension of Arms Overhead (Incorrect)
    "SQZ0": 4,  # Squeezing (Correct)
    "SQZ1": 5,  # Squeezing (Incorrect)
}

# Optotrak vs DOT 10 evaluation tasks (trials 11..20, 1-indexed)
OPTOTRAK_10CLASS_MAP: Dict[int, Tuple[int, str]] = {
    11: (0, "elbow_flex_ext"),
    12: (1, "forearm_pro_sup"),
    13: (2, "shoulder_flex"),
    14: (3, "shoulder_abd"),
    15: (4, "shoulder_int_ext_rot"),
    16: (5, "wrist_flex_ext"),
    17: (6, "wrist_rad_uln_dev"),
    18: (7, "drinking_cup"),
    19: (8, "box_off_shelf"),
    20: (9, "drawing_circles"),
}

# JU-IMU 14 upper-limb ROM and rehab classes
JU_IMU_14CLASS_MAP: Dict[str, int] = {
    "ElbFlex": 0,
    "SupPro": 1,
    "FrontSupPro": 2,
    "WrstFlex": 3,
    "WrstExt": 4,
    "FrontWrstFlex": 5,
    "FrontWrstExt": 6,
    "ShFlex90": 7,
    "ShExt": 8,
    "ShAbd90": 9,
    "Scaption90": 10,
    "ShHorAdd": 11,
    "ShRot": 12,
    "PickCup": 13,
}

# Track 5A: Verified 7-class cross-dataset shared taxonomy (Optotrak vs DOT <-> JU-IMU)
SHARED_7CLASS_NAMES: Tuple[str, ...] = (
    "elbow_flex_ext",
    "forearm_pro_sup",
    "shoulder_flex",
    "shoulder_abd",
    "shoulder_rot",
    "wrist_flex_ext",
    "drinking_cup_adl",
)

OPTOTRAK_TO_SHARED_7CLASS: Dict[int, int] = {
    11: 0,  # elbow_flex_ext
    12: 1,  # forearm_pro_sup
    13: 2,  # shoulder_flex
    14: 3,  # shoulder_abd
    15: 4,  # shoulder_rot
    16: 5,  # wrist_flex_ext
    18: 6,  # drinking_cup_adl
}

JU_IMU_TO_SHARED_7CLASS: Dict[str, int] = {
    "ElbFlex": 0,
    "SupPro": 1,
    "ShFlex90": 2,
    "ShAbd90": 3,
    "ShRot": 4,
    "WrstFlex": 5,
    "WrstExt": 5,
    "PickCup": 6,
}

PHYTMO_TO_SHARED_2CLASS: Dict[str, int] = {
    "EFE0": 0,  # elbow_flex_ext
    "EAH0": 2,  # shoulder_flex / overhead arm extension
}


def convert_acc_to_ms2(acc: np.ndarray, unit: str) -> np.ndarray:
    """Convert 3-axis accelerometer array to standard SI units (m/s^2)."""
    arr = np.asarray(acc, dtype=np.float64)
    if unit == "m/s^2" or unit == "ms2":
        return arr.copy()
    if unit == "g":
        return arr * STANDARD_GRAVITY_MS2
    if unit == "mG" or unit == "mm/s^2":
        return arr * (STANDARD_GRAVITY_MS2 / 1000.0)
    raise ValueError(f"Unsupported accelerometer unit: {unit!r}")


def convert_gyr_to_rads(gyr: np.ndarray, unit: str) -> np.ndarray:
    """Convert 3-axis gyroscope array to standard SI units (rad/s)."""
    arr = np.asarray(gyr, dtype=np.float64)
    if unit == "rad/s" or unit == "rads":
        return arr.copy()
    if unit == "deg/s" or unit == "degs":
        return arr * DEG_TO_RAD
    raise ValueError(f"Unsupported gyroscope unit: {unit!r}")


def convert_angle_to_deg(angle: np.ndarray, unit: str) -> np.ndarray:
    """Ensure ground-truth joint angles are expressed in degrees (deg)."""
    arr = np.asarray(angle, dtype=np.float64)
    if unit == "deg" or unit == "degrees":
        return arr.copy()
    if unit == "rad" or unit == "radians":
        return arr * RAD_TO_DEG
    raise ValueError(f"Unsupported angle unit: {unit!r}")


def assemble_12ch_dual_imu(
    ua_acc: np.ndarray,
    ua_gyr: np.ndarray,
    fa_acc: np.ndarray,
    fa_gyr: np.ndarray,
    acc_unit: str,
    gyr_unit: str,
) -> np.ndarray:
    """Assemble a (T, 12) standardized dual-IMU matrix [UA_acc, UA_gyr, FA_acc, FA_gyr].

    Never fabricates missing channels: all four (T, 3) arrays must be explicitly provided
    and have identical sample length T.
    """
    ua_a = convert_acc_to_ms2(ua_acc, acc_unit)
    ua_g = convert_gyr_to_rads(ua_gyr, gyr_unit)
    fa_a = convert_acc_to_ms2(fa_acc, acc_unit)
    fa_g = convert_gyr_to_rads(fa_gyr, gyr_unit)

    for name, mat in (("ua_acc", ua_a), ("ua_gyr", ua_g), ("fa_acc", fa_a), ("fa_gyr", fa_g)):
        if mat.ndim != 2 or mat.shape[1] != 3:
            raise ValueError(f"{name} must have shape (T, 3), got {mat.shape}")

    lengths = {ua_a.shape[0], ua_g.shape[0], fa_a.shape[0], fa_g.shape[0]}
    if len(lengths) != 1:
        raise ValueError(
            f"Length mismatch across dual-IMU streams: "
            f"ua_acc={ua_a.shape[0]}, ua_gyr={ua_g.shape[0]}, "
            f"fa_acc={fa_a.shape[0]}, fa_gyr={fa_g.shape[0]}"
        )

    out = np.concatenate([ua_a, ua_g, fa_a, fa_g], axis=1)
    if np.isnan(out).any() or np.isinf(out).any():
        raise ValueError("NaN or Inf detected in assembled 12-channel dual-IMU array.")
    return out
