"""Read-only trial loaders with verified unit conversions, deduplication, and resampling."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal

from src.data.schemas import (
    assemble_12ch_dual_imu,
    convert_acc_to_ms2,
    convert_angle_to_deg,
    convert_gyr_to_rads,
)


def deduplicate_and_resample_uniform(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    target_hz: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deduplicate non-increasing timestamps and linearly interpolate to a uniform grid."""
    t = np.asarray(timestamps_s, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if t.ndim != 1 or v.shape[0] != t.shape[0]:
        raise ValueError("timestamps_s must be 1D and match values length.")

    # Sort and keep strictly increasing timestamps (keeps first occurrence of duplicates)
    order = np.argsort(t, kind="mergesort")
    t_sorted = t[order]
    v_sorted = v[order]
    unique_mask = np.concatenate([[True], np.diff(t_sorted) > 1e-9])
    t_unique = t_sorted[unique_mask]
    v_unique = v_sorted[unique_mask]

    dt = 1.0 / float(target_hz)
    t_grid = np.arange(t_unique[0], t_unique[-1] + 1e-9, dt)
    v_interp = np.column_stack([
        np.interp(t_grid, t_unique, v_unique[:, c]) for c in range(v_unique.shape[1])
    ])
    return t_grid, v_interp


def resample_multirate_signal(
    values: np.ndarray,
    source_hz: float,
    target_hz: float,
) -> np.ndarray:
    """Resample a uniformly sampled (T, C) array from source_hz to target_hz with anti-aliasing."""
    arr = np.asarray(values, dtype=np.float64)
    if abs(source_hz - target_hz) < 1e-6:
        return arr.copy()

    # If downsampling significantly (e.g., 2000 Hz -> 100 Hz), apply Butterworth anti-aliasing filter
    if target_hz < source_hz:
        nyq = 0.5 * source_hz
        cutoff = min(0.4 * target_hz, 0.9 * nyq)
        sos = signal.butter(4, cutoff / nyq, btype="low", output="sos")
        arr = signal.sosfiltfilt(sos, arr, axis=0)

    n_target = int(round(arr.shape[0] * float(target_hz) / float(source_hz)))
    t_src = np.linspace(0.0, 1.0, arr.shape[0])
    t_dst = np.linspace(0.0, 1.0, n_target)
    return np.column_stack([np.interp(t_dst, t_src, arr[:, c]) for c in range(arr.shape[1])])


def load_phytmo_trial_12ch(
    ua_csv_path: Path,
    fa_csv_path: Path,
    target_hz: float = 100.0,
) -> Tuple[np.ndarray, int]:
    """Load a PHYTMO Upper-Arm + Forearm CSV pair, deduplicate timestamps, and return (T, 12) SI array + n_dups."""
    df_ua = pd.read_csv(ua_csv_path)
    df_fa = pd.read_csv(fa_csv_path)

    t_ua = df_ua["Time (s)"].to_numpy(dtype=np.float64)
    t_fa = df_fa["Time (s)"].to_numpy(dtype=np.float64)
    n_dups = int((np.diff(t_ua) <= 0).sum() + (np.diff(t_fa) <= 0).sum())

    acc_cols = ["Accelerometer X (g)", "Accelerometer Y (g)", "Accelerometer Z (g)"]
    gyr_cols = ["Gyroscope X (deg/s)", "Gyroscope Y (deg/s)", "Gyroscope Z (deg/s)"]

    ua_raw = np.hstack([df_ua[acc_cols].to_numpy(), df_ua[gyr_cols].to_numpy()])
    fa_raw = np.hstack([df_fa[acc_cols].to_numpy(), df_fa[gyr_cols].to_numpy()])

    # Deduplicate each sensor stream
    _, ua_dedup = deduplicate_and_resample_uniform(t_ua, ua_raw, target_hz=target_hz)
    t_grid_ua, _ = deduplicate_and_resample_uniform(t_ua, ua_raw[:, :1], target_hz=target_hz)
    t_grid_fa, fa_dedup = deduplicate_and_resample_uniform(t_fa, fa_raw, target_hz=target_hz)

    # Align on overlapping duration
    t_start = max(t_grid_ua[0], t_grid_fa[0])
    t_end = min(t_grid_ua[-1], t_grid_fa[-1])
    dt = 1.0 / float(target_hz)
    t_common = np.arange(t_start, t_end + 1e-9, dt)

    ua_aligned = np.column_stack([np.interp(t_common, t_grid_ua, ua_dedup[:, c]) for c in range(6)])
    fa_aligned = np.column_stack([np.interp(t_common, t_grid_fa, fa_dedup[:, c]) for c in range(6)])

    x_12ch = assemble_12ch_dual_imu(
        ua_acc=ua_aligned[:, 0:3],
        ua_gyr=ua_aligned[:, 3:6],
        fa_acc=fa_aligned[:, 0:3],
        fa_gyr=fa_aligned[:, 3:6],
        acc_unit="g",
        gyr_unit="deg/s",
    )
    return x_12ch, n_dups


def load_optotrak_trial_12ch(
    b_dot_path: Path,
    trial_index_0idx: int,
    h_angles_path: Optional[Path] = None,
    target_hz: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load an Optotrak vs DOT trial (RUA + RLA -> 12ch SI) and optional (T, 2) [elbow_flex_deg, pro_sup_deg]."""
    b_mat = sio.loadmat(str(b_dot_path), simplify_cells=True)["dotComplete"]
    trial_struct = b_mat[trial_index_0idx]
    rua = np.asarray(trial_struct["RUA"], dtype=np.float64)
    rla = np.asarray(trial_struct["RLA"], dtype=np.float64)

    # Columns in B_dotComplete: 0=PacketCounter, 1=SampleTimeFine, 2..5=Quat_W..Z, 6..8=Acc_X..Z (m/s^2), 9..11=Gyr_X..Z (deg/s)
    # Guard against row-0 zero artifact if ever present
    if np.allclose(rua[0, 6:12], 0.0) or np.allclose(rla[0, 6:12], 0.0):
        rua = rua[1:]
        rla = rla[1:]

    x_12ch = assemble_12ch_dual_imu(
        ua_acc=rua[:, 6:9],
        ua_gyr=rua[:, 9:12],
        fa_acc=rla[:, 6:9],
        fa_gyr=rla[:, 9:12],
        acc_unit="m/s^2",
        gyr_unit="deg/s",
    )

    y_angles_deg: Optional[np.ndarray] = None
    if h_angles_path is not None and Path(h_angles_path).exists():
        h_mat = sio.loadmat(str(h_angles_path), simplify_cells=True)["j_aligned_eu"]
        elbow_3d = np.asarray(h_mat["opto"][trial_index_0idx]["Elbow"], dtype=np.float64)
        if elbow_3d.shape[0] == x_12ch.shape[0] + 1:
            elbow_3d = elbow_3d[1:]
        # Verified axis mapping: col 2 is Elbow Flexion/Extension (deg), col 1 is Forearm Pronation/Supination (deg)
        y_angles_deg = convert_angle_to_deg(elbow_3d[:, [2, 1]], unit="deg")

    if target_hz is not None and abs(target_hz - 120.0) > 1e-6:
        x_12ch = resample_multirate_signal(x_12ch, source_hz=120.0, target_hz=target_hz)
        if y_angles_deg is not None:
            y_angles_deg = resample_multirate_signal(y_angles_deg, source_hz=120.0, target_hz=target_hz)

    return x_12ch, y_angles_deg


def load_ju_imu_trial_12ch(
    csv_path: Path,
    side: str = "R",
    target_hz: Optional[float] = None,
) -> np.ndarray:
    """Load a JU-IMU trial into (T, 12) SI array (Right side: sensor4=UA, sensor1=FA; Left: sensor5=UA, sensor2=FA)."""
    df = pd.read_csv(csv_path)
    if side.upper() == "R":
        ua_prefix, fa_prefix = "sensor4", "sensor1"
    elif side.upper() == "L":
        ua_prefix, fa_prefix = "sensor5", "sensor2"
    else:
        raise ValueError(f"side must be 'R' or 'L', got {side!r}")

    ua_acc = df[[f"{ua_prefix}.accx", f"{ua_prefix}.accy", f"{ua_prefix}.accz"]].to_numpy()
    ua_gyr = df[[f"{ua_prefix}.gyrx", f"{ua_prefix}.gyry", f"{ua_prefix}.gyrz"]].to_numpy()
    fa_acc = df[[f"{fa_prefix}.accx", f"{fa_prefix}.accy", f"{fa_prefix}.accz"]].to_numpy()
    fa_gyr = df[[f"{fa_prefix}.gyrx", f"{fa_prefix}.gyry", f"{fa_prefix}.gyrz"]].to_numpy()

    x_12ch = assemble_12ch_dual_imu(
        ua_acc=ua_acc,
        ua_gyr=ua_gyr,
        fa_acc=fa_acc,
        fa_gyr=fa_gyr,
        acc_unit="m/s^2",
        gyr_unit="rad/s",
    )
    if target_hz is not None and abs(target_hz - 80.0) > 1e-6:
        x_12ch = resample_multirate_signal(x_12ch, source_hz=80.0, target_hz=target_hz)
    return x_12ch


def load_strokerehab_trial_12ch(
    csv_path: Path,
    side: str = "RT",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a StrokeRehab Schema-A CSV into (T, 12) SI array and (T, 2) [elbow_flex_deg, wrist_sup_deg]."""
    df = pd.read_csv(csv_path)
    s = side.upper()
    if s not in ("RT", "LT"):
        raise ValueError(f"side must be 'RT' or 'LT', got {side!r}")

    ua_acc_mg = df[[
        f"UpperarmAccelSensorX{s}mG",
        f"UpperarmAccelSensorY{s}mG",
        f"UpperarmAccelSensorZ{s}mG",
    ]].to_numpy(dtype=np.float64)
    fa_acc_mg = df[[
        f"ForearmAccelSensorX{s}mG",
        f"ForearmAccelSensorY{s}mG",
        f"ForearmAccelSensorZ{s}mG",
    ]].to_numpy(dtype=np.float64)

    ua_euler_deg = df[[
        f"Upperarmcourse{s}deg",
        f"Upperarmpitch{s}deg",
        f"Upperarmroll{s}deg",
    ]].to_numpy(dtype=np.float64)
    fa_euler_deg = df[[
        f"Forearmcourse{s}deg",
        f"Forearmpitch{s}deg",
        f"Forearmroll{s}deg",
    ]].to_numpy(dtype=np.float64)

    # Differentiate unwrapped Euler angles (in rad) at 100 Hz (dt = 0.01s) to obtain angular rate in rad/s
    dt = 0.01
    ua_gyr_rads = np.gradient(np.unwrap(np.deg2rad(ua_euler_deg), axis=0), dt, axis=0)
    fa_gyr_rads = np.gradient(np.unwrap(np.deg2rad(fa_euler_deg), axis=0), dt, axis=0)

    x_12ch = assemble_12ch_dual_imu(
        ua_acc=ua_acc_mg,
        ua_gyr=ua_gyr_rads,
        fa_acc=fa_acc_mg,
        fa_gyr=fa_gyr_rads,
        acc_unit="mG",
        gyr_unit="rad/s",
    )
    y_angles_deg = df[[f"ElbowFlexion{s}deg", f"WristSupination{s}deg"]].to_numpy(dtype=np.float64)
    return x_12ch, convert_angle_to_deg(y_angles_deg, unit="deg")


def load_ultra_mocap_trial_12ch(
    imu_csv_path: Path,
    ik_csv_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load an ULTra-MoCap 2000Hz IMU CSV + 100Hz OpenSim IK CSV and align at 100 Hz."""
    df_imu = pd.read_csv(imu_csv_path)
    df_ik = pd.read_csv(ik_csv_path)

    # Verified placement: IMU 4 = proximal Upper Arm, IMU 1 = distal Forearm/Hand
    # Verified units: Vicon Nexus ACC in mm/s^2 (divide by 1000 -> m/s^2), GYRO in deg/s
    ua_acc_ms2 = df_imu[["ACCX4", "ACCY4", "ACCZ4"]].to_numpy(dtype=np.float64) / 1000.0
    ua_gyr_degs = df_imu[["GYROX4", "GYROY4", "GYROZ4"]].to_numpy(dtype=np.float64)
    fa_acc_ms2 = df_imu[["ACCX1", "ACCY1", "ACCZ1"]].to_numpy(dtype=np.float64) / 1000.0
    fa_gyr_degs = df_imu[["GYROX1", "GYROY1", "GYROZ1"]].to_numpy(dtype=np.float64)

    x_2000hz = assemble_12ch_dual_imu(
        ua_acc=ua_acc_ms2,
        ua_gyr=ua_gyr_degs,
        fa_acc=fa_acc_ms2,
        fa_gyr=fa_gyr_degs,
        acc_unit="m/s^2",
        gyr_unit="deg/s",
    )
    x_100hz = resample_multirate_signal(x_2000hz, source_hz=2000.0, target_hz=100.0)
    y_angles_deg = df_ik[["elbow_flex_r", "pro_sup_r"]].to_numpy(dtype=np.float64)

    min_len = min(x_100hz.shape[0], y_angles_deg.shape[0])
    return x_100hz[:min_len], convert_angle_to_deg(y_angles_deg[:min_len], unit="deg")


ULF_IN_ADL_MAX_SEGMENTED_FRAMES = 1500


class ULFInADLUnsegmentedRecordingError(ValueError):
    """Raised when an unsegmented ULF_in_ADL .mvnx recording exceeds the 1,500-frame (25s) threshold."""


def validate_ulf_in_adl_mvnx_segmentation(
    mvnx_path: Path,
    max_frames: int = ULF_IN_ADL_MAX_SEGMENTED_FRAMES,
) -> Dict[str, Any]:
    """Inspect an Xsens .mvnx file and verify whether it is properly segmented (<= max_frames)."""
    import lxml.etree as ET

    tree = ET.parse(str(mvnx_path))
    root = tree.getroot()
    m_ns = {"m": root.nsmap[None]}
    frames = root.findall(".//m:frame[@type='normal']", m_ns)
    n_frames = len(frames)
    is_valid = 0 < n_frames <= max_frames
    return {
        "mvnx_path": str(mvnx_path),
        "num_frames": n_frames,
        "duration_sec_60hz": round(n_frames / 60.0, 2),
        "max_allowed_frames": max_frames,
        "is_segmented_valid": bool(is_valid),
        "rejection_reason": (
            None
            if is_valid
            else f"Recording contains {n_frames} frames ({n_frames / 60.0:.1f}s at 60 Hz), exceeding the "
            f"{max_frames}-frame (25.0s) unsegmented threshold (Stage 9/10 P28 safeguard)."
        ),
    }


def load_ulf_in_adl_trial_12ch(
    mvnx_path: Path,
    side: str = "R",
    max_frames: Optional[int] = None,
    enforce_p28_safeguard: bool = False,
    segmentation_verified: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse an Xsens .mvnx file at 60 Hz into (T, 12) SI array and (T, 2) [elbow_flex_deg, pro_sup_deg].

    When `enforce_p28_safeguard=True` or `max_frames` is specified (Stage 10 standard: 1,500 frames),
    unsegmented multi-minute recordings (such as Subject P28's ~39,935-frame files) raise
    `ULFInADLUnsegmentedRecordingError` unless `segmentation_verified=True`.
    """
    import lxml.etree as ET

    tree = ET.parse(str(mvnx_path))
    root = tree.getroot()
    m_ns = {"m": root.nsmap[None]}
    frames = root.findall(".//m:frame[@type='normal']", m_ns)
    if not frames:
        raise ValueError(f"No normal frames found in {mvnx_path}")

    effective_max_frames = (
        max_frames
        if max_frames is not None
        else (ULF_IN_ADL_MAX_SEGMENTED_FRAMES if enforce_p28_safeguard else None)
    )
    if effective_max_frames is not None and not segmentation_verified and len(frames) > effective_max_frames:
        raise ULFInADLUnsegmentedRecordingError(
            f"ULF_in_ADL P28 safeguard triggered for {mvnx_path.name}: {len(frames)} frames "
            f"({len(frames) / 60.0:.1f}s at 60 Hz) exceeds max_frames={effective_max_frames}. "
            "Exclude unsegmented .mvnx files or pass segmentation_verified=True after task segmentation."
        )

    s = side.upper()
    if s == "R":
        # Segment 9 (0-idx 8 -> 24:27) = RightUpperArm; Segment 10 (0-idx 9 -> 27:30) = RightForeArm
        # Joint 8 (0-idx 8 -> 24:27) = jRightElbow
        ua_slice, fa_slice, j_slice = slice(24, 27), slice(27, 30), slice(24, 27)
    elif s == "L":
        # Segment 13 (0-idx 12 -> 36:39) = LeftUpperArm; Segment 14 (0-idx 13 -> 39:42) = LeftForeArm
        # Joint 12 (0-idx 12 -> 36:39) = jLeftElbow
        ua_slice, fa_slice, j_slice = slice(36, 39), slice(39, 42), slice(36, 39)
    else:
        raise ValueError(f"side must be 'R' or 'L', got {side!r}")

    acc_list, gyr_list, jnt_list = [], [], []
    for f in frames:
        acc_vals = [float(v) for v in f.find("m:acceleration", m_ns).text.split()]
        gyr_vals = [float(v) for v in f.find("m:angularVelocity", m_ns).text.split()]
        jnt_vals = [float(v) for v in f.find("m:jointAngle", m_ns).text.split()]
        acc_list.append(acc_vals)
        gyr_list.append(gyr_vals)
        jnt_list.append(jnt_vals)

    acc_mat = np.asarray(acc_list, dtype=np.float64)
    gyr_mat = np.asarray(gyr_list, dtype=np.float64)
    jnt_mat = np.asarray(jnt_list, dtype=np.float64)

    x_12ch = assemble_12ch_dual_imu(
        ua_acc=acc_mat[:, ua_slice],
        ua_gyr=gyr_mat[:, ua_slice],
        fa_acc=acc_mat[:, fa_slice],
        fa_gyr=gyr_mat[:, fa_slice],
        acc_unit="m/s^2",
        gyr_unit="rad/s",
    )
    elbow_3d = jnt_mat[:, j_slice]
    # ISB ZXY convention in Xsens MVN: col 2 = Elbow Flexion/Extension (deg), col 1 = Pronation/Supination (deg)
    y_angles_deg = convert_angle_to_deg(elbow_3d[:, [2, 1]], unit="deg")
    return x_12ch, y_angles_deg


def load_vidimu_trial_8ch_quat(
    sto_path: Path,
    mot_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load VIDIMU 50 Hz .sto (humerus_r_imu + radius_r_imu quaternions -> 8ch) and .mot [elbow_flex_r, pro_sup_r]."""
    with open(sto_path, "r", encoding="utf-8") as f:
        sto_lines = f.readlines()
    sto_start = next(i for i, l in enumerate(sto_lines) if l.strip() == "endheader") + 1
    sto_header = sto_lines[sto_start].strip().split("\t")
    hum_idx = sto_header.index("humerus_r_imu")
    rad_idx = sto_header.index("radius_r_imu")

    quats = []
    for line in sto_lines[sto_start + 1 :]:
        parts = line.strip().split("\t")
        if len(parts) <= max(hum_idx, rad_idx):
            continue
        q_hum = [float(v) for v in parts[hum_idx].split(",")]
        q_rad = [float(v) for v in parts[rad_idx].split(",")]
        quats.append(q_hum + q_rad)
    x_quat = np.asarray(quats, dtype=np.float64)

    with open(mot_path, "r", encoding="utf-8") as f:
        mot_lines = f.readlines()
    mot_start = next(i for i, l in enumerate(mot_lines) if l.strip() == "endheader") + 1
    mot_header = mot_lines[mot_start].strip().split("\t")
    ef_idx = mot_header.index("elbow_flex_r")
    ps_idx = mot_header.index("pro_sup_r")

    angles = []
    for line in mot_lines[mot_start + 1 :]:
        parts = line.strip().split("\t")
        if len(parts) <= max(ef_idx, ps_idx):
            continue
        angles.append([float(parts[ef_idx]), float(parts[ps_idx])])
    y_angles = np.asarray(angles, dtype=np.float64)

    # Drop frame 0 (default OpenSim initialization pose at t=0.0) and align lengths
    min_len = min(x_quat.shape[0], y_angles.shape[0])
    if min_len > 2:
        x_quat = x_quat[1:min_len]
        y_angles = y_angles[1:min_len]
    return x_quat, convert_angle_to_deg(y_angles, unit="deg")


def load_rehab_assessment_proximal_splits(
    raw_root: Path,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, list]]:
    """Load REHAB-120 Rehab_assessment proximal upper-limb trials (T=150, 12ch) with FMA-UE scores (0, 1, 2).

    Deterministic session split across the 222 sessions:
      - train: sessions 001..154 (154 sessions)
      - val:   sessions 155..188 (34 sessions)
      - test:  sessions 189..222 (34 sessions)
    """
    assess_dir = raw_root / "REHAB" / "Rehab_assessment"
    labels_mat = np.load(assess_dir / "Label.npy")  # (222, 27)
    proc_dir = assess_dir / "d02_processed_data"

    # 16 proximal arm tasks where _1.npy and _2.npy contain active dual-IMU signals
    proximal_tasks = list(range(0, 15)) + [26]

    splits: Dict[str, Tuple[list, list, list]] = {
        "train": ([], [], []),
        "val": ([], [], []),
        "test": ([], [], []),
    }

    for sess_idx in range(1, 223):
        sess_name = f"{sess_idx:03d}"
        sess_dir = proc_dir / sess_name
        if sess_idx <= 154:
            sp = "train"
        elif sess_idx <= 188:
            sp = "val"
        else:
            sp = "test"

        for t_idx in proximal_tasks:
            f1 = sess_dir / f"{t_idx:03d}_1.npy"
            f2 = sess_dir / f"{t_idx:03d}_2.npy"
            if not f1.exists() or not f2.exists():
                continue
            arr1 = np.load(f1)  # (150, 12): cols 3..5=acc, 9..11=gyr
            arr2 = np.load(f2)  # (150, 13): cols 4..6=acc, 10..12=gyr
            if arr1.shape != (150, 12) or arr2.shape != (150, 13):
                continue
            x_12 = np.hstack([arr1[:, 3:6], arr1[:, 9:12], arr2[:, 4:7], arr2[:, 10:13]]).astype(np.float32)
            if np.isnan(x_12).any() or np.allclose(x_12, 0.0):
                continue
            y_fma = int(labels_mat[sess_idx - 1, t_idx])
            splits[sp][0].append(x_12)
            splits[sp][1].append(y_fma)
            splits[sp][2].append(sess_name)

    return {
        sp: (
            np.stack(xs, axis=0),
            np.asarray(ys, dtype=np.int64),
            s_ids,
        )
        for sp, (xs, ys, s_ids) in splits.items()
    }


def _compute_orientation_invariant_single_sequence(
    seq_tc: np.ndarray,
    mode: str = "legacy_full_trial",
) -> np.ndarray:
    """Internal helper computing 12 orientation-invariant channels on a 2D (T, 12) sequence."""
    arr = np.asarray(seq_tc, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 12:
        raise ValueError(f"Expected 2D (T, 12) array, got shape {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    ua_acc = arr[:, 0:3]
    ua_gyr = arr[:, 3:6]
    fa_acc = arr[:, 6:9]
    fa_gyr = arr[:, 9:12]

    eps = 1e-6
    ua_acc_norm = np.linalg.norm(ua_acc, axis=1)
    ua_gyr_norm = np.linalg.norm(ua_gyr, axis=1)
    fa_acc_norm = np.linalg.norm(fa_acc, axis=1)
    fa_gyr_norm = np.linalg.norm(fa_gyr, axis=1)

    if mode in ("legacy_full_trial", "per_window"):
        ua_mean = ua_acc.mean(axis=0, keepdims=True)
        fa_mean = fa_acc.mean(axis=0, keepdims=True)
        ua_dyn_norm = np.linalg.norm(ua_acc - ua_mean, axis=1)
        fa_dyn_norm = np.linalg.norm(fa_acc - fa_mean, axis=1)
        if mode == "legacy_full_trial" and arr.shape[0] >= 2:
            ua_gyr_jerk = np.linalg.norm(np.gradient(ua_gyr, axis=0), axis=1)
            fa_gyr_jerk = np.linalg.norm(np.gradient(fa_gyr, axis=0), axis=1)
        else:
            # Causal backward first difference inside window (zero at frame 0)
            ua_diff = np.vstack([np.zeros((1, 3), dtype=np.float64), np.diff(ua_gyr, axis=0)])
            fa_diff = np.vstack([np.zeros((1, 3), dtype=np.float64), np.diff(fa_gyr, axis=0)])
            ua_gyr_jerk = np.linalg.norm(ua_diff, axis=1)
            fa_gyr_jerk = np.linalg.norm(fa_diff, axis=1)
    elif mode == "causal_expanding":
        # Strictly causal sample-by-sample expanding mean: mu_t = (1/(t+1)) * sum_{k=0..t} a_k
        counts = np.arange(1, arr.shape[0] + 1, dtype=np.float64)[:, None]
        ua_causal_mean = np.cumsum(ua_acc, axis=0) / counts
        fa_causal_mean = np.cumsum(fa_acc, axis=0) / counts
        ua_dyn_norm = np.linalg.norm(ua_acc - ua_causal_mean, axis=1)
        fa_dyn_norm = np.linalg.norm(fa_acc - fa_causal_mean, axis=1)

        ua_diff = np.vstack([np.zeros((1, 3), dtype=np.float64), np.diff(ua_gyr, axis=0)])
        fa_diff = np.vstack([np.zeros((1, 3), dtype=np.float64), np.diff(fa_gyr, axis=0)])
        ua_gyr_jerk = np.linalg.norm(ua_diff, axis=1)
        fa_gyr_jerk = np.linalg.norm(fa_diff, axis=1)
    else:
        raise ValueError(
            f"Unsupported orientation-invariant mode {mode!r}. "
            "Expected 'legacy_full_trial', 'per_window', or 'causal_expanding'."
        )

    rel_acc_norm = np.linalg.norm(fa_acc - ua_acc, axis=1)
    rel_gyr_norm = np.linalg.norm(fa_gyr - ua_gyr, axis=1)

    cos_acc = np.sum(ua_acc * fa_acc, axis=1) / (ua_acc_norm * fa_acc_norm + eps)
    cos_gyr = np.sum(ua_gyr * fa_gyr, axis=1) / (ua_gyr_norm * fa_gyr_norm + eps)

    return np.column_stack([
        ua_acc_norm,
        ua_dyn_norm,
        ua_gyr_norm,
        fa_acc_norm,
        fa_dyn_norm,
        fa_gyr_norm,
        rel_acc_norm,
        rel_gyr_norm,
        cos_acc,
        cos_gyr,
        ua_gyr_jerk,
        fa_gyr_jerk,
    ]).astype(np.float32)


def compute_orientation_invariant_12ch(
    x_12ch: np.ndarray,
    mode: str = "legacy_full_trial",
) -> np.ndarray:
    """Compute a 12-channel orientation-invariant kinematic representation from canonical 12-ch dual IMU.

    Modes
    -----
    - ``"legacy_full_trial"`` (default): Preserves Stage 8/9 behavior on 2D ``(T, 12)`` trials
      for exact checkpoint reproducibility on ``Track 5A-1`` and ``Track 5A-2``.
    - ``"per_window"``: Computes mean acceleration and backward difference strictly inside each
      individual window ``(W, 12)`` or ``(N, W, 12)`` with zero cross-window leakage.
    - ``"causal_expanding"``: Strictly causal sample-by-sample transform where frame ``t`` depends
      only on samples ``0..t`` (expanding running mean and backward difference).
    """
    arr = np.asarray(x_12ch, dtype=np.float64)
    if arr.ndim == 2:
        return _compute_orientation_invariant_single_sequence(arr, mode=mode)
    if arr.ndim == 3:
        win_mode = "per_window" if mode == "legacy_full_trial" else mode
        return np.stack(
            [_compute_orientation_invariant_single_sequence(arr[i], mode=win_mode) for i in range(arr.shape[0])],
            axis=0,
        )
    raise ValueError(f"Expected 2D (T, 12) or 3D (N, W, 12) input, got shape {arr.shape}")


def compute_orientation_invariant_windows_12ch(
    x_windows_nwc: np.ndarray,
    mode: str = "causal_expanding",
) -> np.ndarray:
    """Apply causal or per-window orientation-invariant feature extraction identically in training & inference."""
    arr = np.asarray(x_windows_nwc, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 12:
        raise ValueError(f"Expected 3D (N, W, 12) window tensor, got shape {arr.shape}")
    if mode not in ("per_window", "causal_expanding"):
        raise ValueError(f"Window-level invariant transform requires 'per_window' or 'causal_expanding', got {mode!r}")
    return compute_orientation_invariant_12ch(arr, mode=mode)


