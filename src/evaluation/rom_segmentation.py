"""Repetition-cycle segmentation and three-tier Range of Motion (ROM) evaluation for Stage 10.

Strictly separates three distinct biomechanical ROM definitions on continuous angle trajectories (deg):
  1. Window-Level ROM (`window_2s_mean_rom_deg`):
     Mean of (max - min) within fixed 2.0-second sliding windows. Matches Stage 8 `rom_mae_deg`.
  2. Trial-Level ROM (`full_trial_peak_to_trough_rom_deg`):
     Global (max - min) across the entire reconstructed recording trajectory.
  3. Repetition-Level ROM (`repetition_level_mean_rom_deg`):
     Mean of per-cycle peak-to-trough excursions across individual segmented movement repetitions
     with validated trough-to-peak-to-trough cycle boundaries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from scipy.signal import find_peaks, savgol_filter


def segment_repetition_cycles_rom(
    angle_deg: np.ndarray,
    sampling_rate_hz: float,
    min_prominence_deg: float = 10.0,
    min_cycle_duration_sec: float = 0.6,
) -> Dict[str, Any]:
    """Segment a 1D continuous joint angle trajectory (deg) into individual repetition cycles.

    Parameters
    ----------
    angle_deg : np.ndarray
        1D array of continuous joint angles in degrees of shape (T,).
    sampling_rate_hz : float
        Sampling rate in Hz (e.g., 100.0 or 120.0).
    min_prominence_deg : float
        Minimum angular excursion (deg) required to qualify as a true movement peak/trough
        rather than postural micro-tremor.
    min_cycle_duration_sec : float
        Minimum physiological duration (seconds) between consecutive repetition peaks.

    Returns
    -------
    dict
        Structured dictionary containing detected repetition count, per-cycle boundaries and ROMs,
        and summary statistics (mean, std, min, max repetition ROM in degrees).
    """
    sig = np.asarray(angle_deg, dtype=np.float64).reshape(-1)
    n_samples = int(sig.shape[0])
    hz = float(sampling_rate_hz)
    if n_samples < 5 or hz <= 0:
        return {
            "num_repetitions_detected": 0,
            "mean_repetition_rom_deg": None,
            "std_repetition_rom_deg": None,
            "min_repetition_rom_deg": None,
            "max_repetition_rom_deg": None,
            "segmentation_method": "insufficient_signal_length",
            "cycles": [],
        }

    # Light zero-phase smoothing strictly for robust peak/trough index localization
    win_smooth = min(n_samples if n_samples % 2 == 1 else n_samples - 1, max(5, int(round(0.25 * hz)) | 1))
    if win_smooth >= 5 and n_samples > win_smooth:
        sig_smooth = savgol_filter(sig, window_length=win_smooth, polyorder=2)
    else:
        sig_smooth = sig.copy()

    min_dist = max(1, int(round(min_cycle_duration_sec * hz)))
    peaks, _ = find_peaks(sig_smooth, distance=min_dist, prominence=min_prominence_deg)
    troughs, _ = find_peaks(-sig_smooth, distance=min_dist, prominence=min_prominence_deg)

    cycles: List[Dict[str, Any]] = []

    if len(peaks) > 0:
        # Construct bounding troughs around each detected movement peak
        boundary_troughs = np.unique(np.concatenate(([0], troughs, [n_samples - 1])))
        for idx, p_frame in enumerate(peaks):
            preceding = boundary_troughs[boundary_troughs < p_frame]
            succeeding = boundary_troughs[boundary_troughs > p_frame]
            start_frame = int(preceding[-1]) if len(preceding) > 0 else 0
            end_frame = int(succeeding[0]) if len(succeeding) > 0 else n_samples - 1

            if end_frame <= start_frame:
                continue

            seg_raw = sig[start_frame : end_frame + 1]
            c_min = float(np.min(seg_raw))
            c_max = float(np.max(seg_raw))
            c_rom = c_max - c_min
            if c_rom < min_prominence_deg * 0.75:
                continue

            cycles.append({
                "rep_index": len(cycles) + 1,
                "start_frame": start_frame,
                "peak_frame": int(p_frame),
                "end_frame": end_frame,
                "start_time_s": round(start_frame / hz, 3),
                "peak_time_s": round(int(p_frame) / hz, 3),
                "end_time_s": round(end_frame / hz, 3),
                "duration_sec": round((end_frame - start_frame) / hz, 3),
                "min_angle_deg": round(c_min, 2),
                "max_angle_deg": round(c_max, 2),
                "rom_deg": round(c_rom, 2),
            })

    # Fallback if a single monotonic/arching movement has excursion >= min_prominence_deg but no interior peak
    if not cycles:
        total_exc = float(np.max(sig) - np.min(sig))
        if total_exc >= min_prominence_deg:
            p_frame = int(np.argmax(sig))
            c_min = float(np.min(sig))
            c_max = float(np.max(sig))
            cycles.append({
                "rep_index": 1,
                "start_frame": 0,
                "peak_frame": p_frame,
                "end_frame": n_samples - 1,
                "start_time_s": 0.0,
                "peak_time_s": round(p_frame / hz, 3),
                "end_time_s": round((n_samples - 1) / hz, 3),
                "duration_sec": round((n_samples - 1) / hz, 3),
                "min_angle_deg": round(c_min, 2),
                "max_angle_deg": round(c_max, 2),
                "rom_deg": round(total_exc, 2),
            })

    if not cycles:
        return {
            "num_repetitions_detected": 0,
            "mean_repetition_rom_deg": 0.0,
            "std_repetition_rom_deg": 0.0,
            "min_repetition_rom_deg": 0.0,
            "max_repetition_rom_deg": 0.0,
            "segmentation_method": "prominence_peak_trough_savgol",
            "min_prominence_threshold_deg": min_prominence_deg,
            "cycles": [],
        }

    roms = np.asarray([c["rom_deg"] for c in cycles], dtype=np.float64)
    return {
        "num_repetitions_detected": len(cycles),
        "mean_repetition_rom_deg": round(float(np.mean(roms)), 2),
        "std_repetition_rom_deg": round(float(np.std(roms)), 2),
        "min_repetition_rom_deg": round(float(np.min(roms)), 2),
        "max_repetition_rom_deg": round(float(np.max(roms)), 2),
        "segmentation_method": "prominence_peak_trough_savgol",
        "min_prominence_threshold_deg": min_prominence_deg,
        "cycles": cycles,
    }


def compute_three_tier_rom_comparison(
    pred_traj_deg: np.ndarray,
    pred_windows_deg: np.ndarray,
    sampling_rate_hz: float,
    gt_traj_deg: Optional[np.ndarray] = None,
    gt_windows_deg: Optional[np.ndarray] = None,
    min_prominence_deg: float = 10.0,
) -> Dict[str, Any]:
    """Compute distinct Window-Level, Trial-Level, and Repetition-Level ROM metrics for a joint trajectory.

    Parameters
    ----------
    pred_traj_deg : np.ndarray
        Reconstructed 1D predicted trajectory of shape (T,).
    pred_windows_deg : np.ndarray
        Windowed 2D predicted trajectories of shape (N_win, W).
    sampling_rate_hz : float
        Sampling frequency in Hz.
    gt_traj_deg : Optional[np.ndarray]
        Ground-truth 1D trajectory of shape (T,), if available.
    gt_windows_deg : Optional[np.ndarray]
        Ground-truth 2D windowed trajectories of shape (N_win, W), if available.
    min_prominence_deg : float
        Prominence threshold in degrees for cycle segmentation.
    """
    pred_1d = np.asarray(pred_traj_deg, dtype=np.float64).reshape(-1)
    pred_win_2d = np.asarray(pred_windows_deg, dtype=np.float64)

    # Tier 1: Window-level ROM (2.0s sliding window)
    win_roms_pred = np.max(pred_win_2d, axis=1) - np.min(pred_win_2d, axis=1)
    window_rom_pred_deg = float(np.mean(win_roms_pred))

    # Tier 2: Trial-level ROM (global max - min over recording)
    trial_min_pred = float(np.min(pred_1d))
    trial_max_pred = float(np.max(pred_1d))
    trial_rom_pred_deg = trial_max_pred - trial_min_pred

    # Tier 3: Repetition-level ROM (segmented trough-to-peak-to-trough cycles)
    rep_seg_pred = segment_repetition_cycles_rom(
        pred_1d, sampling_rate_hz=sampling_rate_hz, min_prominence_deg=min_prominence_deg
    )
    rep_rom_pred_deg = float(rep_seg_pred["mean_repetition_rom_deg"] or trial_rom_pred_deg)

    out: Dict[str, Any] = {
        "window_level_rom": {
            "definition": "Mean (max - min) within 2.0-second sliding windows (Stage 8 benchmark metric)",
            "pred_mean_rom_deg": round(window_rom_pred_deg, 2),
            "gt_mean_rom_deg": None,
            "mae_deg": None,
        },
        "trial_level_rom": {
            "definition": "Global peak-to-trough (max - min) across the entire trial trajectory",
            "pred_min_deg": round(trial_min_pred, 2),
            "pred_max_deg": round(trial_max_pred, 2),
            "pred_rom_deg": round(trial_rom_pred_deg, 2),
            "gt_min_deg": None,
            "gt_max_deg": None,
            "gt_rom_deg": None,
            "abs_error_deg": None,
        },
        "repetition_level_rom": {
            "definition": "Mean peak-to-trough excursion across segmented individual repetition cycles",
            "num_repetitions_pred": rep_seg_pred["num_repetitions_detected"],
            "pred_mean_rep_rom_deg": round(rep_rom_pred_deg, 2),
            "pred_std_rep_rom_deg": rep_seg_pred["std_repetition_rom_deg"],
            "pred_cycles": rep_seg_pred["cycles"],
            "num_repetitions_gt": None,
            "gt_mean_rep_rom_deg": None,
            "gt_cycles": None,
            "mean_rep_rom_abs_error_deg": None,
            "matched_cycle_rom_mae_deg": None,
        },
    }

    if gt_traj_deg is not None:
        gt_1d = np.asarray(gt_traj_deg, dtype=np.float64).reshape(-1)
        gt_min = float(np.min(gt_1d))
        gt_max = float(np.max(gt_1d))
        gt_trial_rom = gt_max - gt_min

        out["trial_level_rom"]["gt_min_deg"] = round(gt_min, 2)
        out["trial_level_rom"]["gt_max_deg"] = round(gt_max, 2)
        out["trial_level_rom"]["gt_rom_deg"] = round(gt_trial_rom, 2)
        out["trial_level_rom"]["abs_error_deg"] = round(abs(trial_rom_pred_deg - gt_trial_rom), 2)

        rep_seg_gt = segment_repetition_cycles_rom(
            gt_1d, sampling_rate_hz=sampling_rate_hz, min_prominence_deg=min_prominence_deg
        )
        gt_rep_rom = float(rep_seg_gt["mean_repetition_rom_deg"] or gt_trial_rom)
        out["repetition_level_rom"]["num_repetitions_gt"] = rep_seg_gt["num_repetitions_detected"]
        out["repetition_level_rom"]["gt_mean_rep_rom_deg"] = round(gt_rep_rom, 2)
        out["repetition_level_rom"]["gt_cycles"] = rep_seg_gt["cycles"]
        out["repetition_level_rom"]["mean_rep_rom_abs_error_deg"] = round(abs(rep_rom_pred_deg - gt_rep_rom), 2)

        # Evaluate predicted ROM inside each ground-truth repetition boundary [start_frame, end_frame]
        if rep_seg_gt["cycles"]:
            cycle_errs = []
            for cyc in rep_seg_gt["cycles"]:
                sf, ef = cyc["start_frame"], cyc["end_frame"]
                pred_seg = pred_1d[sf : ef + 1]
                pred_cyc_rom = float(np.max(pred_seg) - np.min(pred_seg)) if len(pred_seg) > 0 else 0.0
                cycle_errs.append(abs(pred_cyc_rom - float(cyc["rom_deg"])))
            out["repetition_level_rom"]["matched_cycle_rom_mae_deg"] = round(float(np.mean(cycle_errs)), 2)

    if gt_windows_deg is not None:
        gt_win_2d = np.asarray(gt_windows_deg, dtype=np.float64)
        win_roms_gt = np.max(gt_win_2d, axis=1) - np.min(gt_win_2d, axis=1)
        out["window_level_rom"]["gt_mean_rom_deg"] = round(float(np.mean(win_roms_gt)), 2)
        out["window_level_rom"]["mae_deg"] = round(float(np.mean(np.abs(win_roms_pred - win_roms_gt))), 2)

    return out


def compute_angular_velocity(angle_deg: np.ndarray, sampling_rate_hz: float = 100.0) -> np.ndarray:
    """Compute instantaneous angular velocity in deg/s using Savitzky-Golay derivative or central gradient."""
    arr = np.asarray(angle_deg, dtype=np.float64)
    dt = 1.0 / max(float(sampling_rate_hz), 1.0)
    if len(arr) >= 9:
        vel = savgol_filter(arr, window_length=7, polyorder=2, deriv=1, delta=dt)
    elif len(arr) >= 3:
        vel = np.gradient(arr, dt)
    else:
        vel = np.zeros_like(arr)
    return vel


def compute_sparc_smoothness(
    speed_profile: np.ndarray,
    sampling_rate_hz: float = 100.0,
    padlevel: int = 4,
    fc: float = 10.0,
    amp_th: float = 0.05,
) -> float:
    """Compute Spectral Arc Length (SPARC) movement smoothness metric (Balasubramanian et al. 2015).

    SPARC is dimensionless and negative. Values closer to 0 (e.g. -1.2 to -2.0) indicate smooth,
    uninterrupted coordination, while more negative values (< -3.0) quantify jerkiness, arrest, or tremor.
    """
    v = np.abs(np.asarray(speed_profile, dtype=np.float64))
    if len(v) < 10 or float(np.max(v)) < 1e-4:
        return 0.0

    N = len(v)
    N_fft = int(2 ** (np.ceil(np.log2(N)) + padlevel))
    fft_v = np.abs(np.fft.rfft(v, n=N_fft))
    freqs = np.fft.rfftfreq(N_fft, d=1.0 / sampling_rate_hz)

    if fft_v[0] < 1e-9:
        return -10.0
    fft_norm = fft_v / fft_v[0]

    mask = (freqs <= fc) & (fft_norm >= amp_th)
    if not np.any(mask):
        return -10.0
    idx_max = np.where(mask)[0][-1]
    f_cut = freqs[: idx_max + 1]
    v_cut = fft_norm[: idx_max + 1]

    if len(f_cut) < 2:
        return -10.0

    df = freqs[1] - freqs[0]
    d_v = np.diff(v_cut) / df
    d_f = 1.0 / (f_cut[-1] - f_cut[0] + 1e-9)
    arc_len = np.sum(np.sqrt(d_f**2 + d_v**2)) * df
    return float(-arc_len)

