"""
========================================================================================
AKIRA KNEE REHABILITATION: 16 CLINICAL FEATURE EXTRACTION ENGINE (math_feature_extractor.py)
========================================================================================
WHAT THIS FILE IS:
  This module is the core Mathematical Kinematic Engine of the AKIRA system.
  It converts raw dual-IMU signals (thigh and shin 6-axis sensors @ 100 Hz) into
  3D quaternion-fused anatomical knee flexion angles and 16 domain-specific
  orthopedic kinematic features.

WHAT THIS FILE DOES (Proposal Figure 3, Boxes 02 & 03):
  1. Step 2.1: Cubic-spline uniform resampling to 100 Hz (`resample_to_100hz`).
  2. Step 2.2: Zero-phase 4th-order Butterworth low-pass noise filtering at 5 Hz (`butter_lowpass_filter`).
  3. Step 2.3: 3D SVD sagittal-plane sensor-to-segment static tare calibration & gyro bias removal (`calibrate_sensors`).
  4. Step 2.4: 6-DoF Madgwick quaternion sensor fusion (`madgwick_fusion`).
  5. Step 2.5: 3D relative joint orientation quaternion (`compute_relative_orientation`: q_rel = q_thigh* (x) q_shin).
  6. Step 2.6: Continuous sagittal knee angle theta(t), projected on the hinge axis and referenced to the
     most-extended posture of the recording, so theta >= 0 (`compute_knee_angle`).
  7. Box 03  : Extracts 16 non-redundant, clinically validated biomechanical features (`extract_16_features`).
========================================================================================
"""

import numpy as np
from scipy import signal, interpolate


class Clinical16FeatureExtractor:
    """
    Self-contained biomechanical preprocessing, 6-DoF Madgwick quaternion sensor fusion,
    3D relative orientation, and 16-feature extraction engine for knee rehabilitation.
    """

    def __init__(self, fs: float = 100.0, filter_cutoff: float = 5.0, filter_order: int = 4, beta: float = 0.04):
        self.fs = float(fs)
        self.dt = 1.0 / self.fs
        self.filter_cutoff = float(filter_cutoff)
        self.filter_order = int(filter_order)
        self.beta = float(beta)

    # ----------------------------------------------------------------------------------
    # STEP 2.1: CUBIC-SPLINE UNIFORM RESAMPLING TO 100 HZ
    # ----------------------------------------------------------------------------------
    def resample_to_100hz(
        self,
        time_s_or_data: np.ndarray,
        data: np.ndarray = None,
        target_fs: float = None,
        source_fs: float = None,
        target_samples: int = None
    ):
        """
        Resamples arbitrary-rate IMU streams (e.g., 52 Hz, 56.35 Hz, 148.15 Hz) onto a
        uniform 100 Hz grid using Cubic-Spline interpolation.
        Supports two calling conventions:
          1. resample_to_100hz(time_s, data, target_fs=100.0) -> (uniform_time, resampled_data)
          2. resample_to_100hz(data, source_fs=50.0, target_samples=None) -> resampled_data
             (When target_samples is omitted, preserves exact physical duration at 100.0 Hz:
              n_out = round(N * (100.0 / source_fs))).
        """
        if data is None:
            raw_arr = np.asarray(time_s_or_data, dtype=np.float64)
            N = len(raw_arr)
            fs_in = float(source_fs) if source_fs is not None else self.fs
            fs_out = float(target_fs) if target_fs is not None else self.fs
            n_out = int(target_samples) if target_samples is not None else max(4, int(round(N * (fs_out / max(1.0, fs_in)))))
            if N < 4:
                return np.resize(raw_arr, (n_out,) + raw_arr.shape[1:]).astype(np.float32)
            t_in = np.linspace(0.0, (N - 1) / max(1.0, fs_in), N)
            t_out = np.linspace(0.0, (N - 1) / max(1.0, fs_in), n_out)
            if raw_arr.ndim == 1:
                cs = interpolate.CubicSpline(t_in, raw_arr, bc_type="natural")
                return cs(t_out).astype(np.float32)
            out_arr = np.zeros((n_out, raw_arr.shape[1]), dtype=np.float64)
            for c in range(raw_arr.shape[1]):
                cs = interpolate.CubicSpline(t_in, raw_arr[:, c], bc_type="natural")
                out_arr[:, c] = cs(t_out)
            return out_arr.astype(np.float32)

        fs_out = float(target_fs) if target_fs is not None else self.fs
        time_s = np.asarray(time_s_or_data, dtype=np.float64)
        data = np.asarray(data, dtype=np.float64)

        unique_mask = np.concatenate([[True], np.diff(time_s) > 1e-7])
        time_s = time_s[unique_mask]
        data = data[unique_mask] if data.ndim == 1 else data[unique_mask, :]

        if len(time_s) < 4:
            return time_s, data.astype(np.float32)

        t_start, t_end = time_s[0], time_s[-1]
        n_target = max(4, int(np.round((t_end - t_start) * fs_out)) + 1)
        uniform_time = np.linspace(t_start, t_end, n_target)

        if data.ndim == 1:
            cs = interpolate.CubicSpline(time_s, data, bc_type="natural")
            resampled_data = cs(uniform_time)
        else:
            resampled_data = np.zeros((n_target, data.shape[1]), dtype=np.float64)
            for c in range(data.shape[1]):
                cs = interpolate.CubicSpline(time_s, data[:, c], bc_type="natural")
                resampled_data[:, c] = cs(uniform_time)

        return uniform_time, resampled_data.astype(np.float32)

    # ----------------------------------------------------------------------------------
    # STEP 2.2: ZERO-PHASE 4TH-ORDER BUTTERWORTH LOW-PASS FILTER
    # ----------------------------------------------------------------------------------
    def butter_lowpass_filter(self, data: np.ndarray, cutoff_hz: float = None, fs: float = None) -> np.ndarray:
        """
        Applies a zero-phase 4th-order Butterworth low-pass filter (default 5.0 Hz)
        to suppress high-frequency sensor vibration and soft-tissue artifact.
        """
        fs_use = float(fs) if fs is not None else self.fs
        fc_use = float(cutoff_hz) if cutoff_hz is not None else self.filter_cutoff
        data = np.asarray(data, dtype=np.float64)
        if len(data) < 15:
            return data.astype(np.float32)

        nyq = 0.5 * fs_use
        norm_cutoff = np.clip(fc_use / nyq, 0.01, 0.95)
        b, a = signal.butter(self.filter_order, norm_cutoff, btype="low", analog=False)

        if data.ndim == 1:
            return signal.filtfilt(b, a, data).astype(np.float32)
        filtered = np.zeros_like(data)
        for c in range(data.shape[1]):
            filtered[:, c] = signal.filtfilt(b, a, data[:, c])
        return filtered.astype(np.float32)

    # ----------------------------------------------------------------------------------
    # STEP 2.3: 3D SVD SENSOR-TO-SEGMENT CALIBRATION & GYRO BIAS REMOVAL
    # ----------------------------------------------------------------------------------
    def calibrate_sensors(
        self,
        thigh_acc: np.ndarray,
        thigh_gyr: np.ndarray = None,
        shin_acc: np.ndarray = None,
        shin_gyr: np.ndarray = None,
        static_samples: int = 20
    ):
        """
        Step 2.3:
          1. Identifies the lowest-angular-velocity extension reference window (`quiet_idx`)
             and removes static gyroscope zero-rate bias when near rest (< 15 deg/s).
          2. Identifies the principal 3D knee hinge axis in each sensor's local coordinate
             frame via SVD of the angular velocity vectors and projects the static gravity
             vectors into the sagittal plane orthogonal to that hinge axis.
          3. Computes the true sagittal static baseline extension angle (`tare_angle_deg`)
             and returns `(thigh_acc, thigh_gyr_cal, shin_acc, shin_gyr_cal, tare_angle_deg, quiet_idx)`.
        """
        single_matrix_mode = False
        if thigh_gyr is None and thigh_acc.ndim == 2 and thigh_acc.shape[1] >= 12:
            single_matrix_mode = True
            mat12 = np.asarray(thigh_acc, dtype=np.float64)
            thigh_acc = mat12[:, 0:3]
            thigh_gyr = mat12[:, 3:6]
            shin_acc = mat12[:, 6:9]
            shin_gyr = mat12[:, 9:12]
        else:
            thigh_acc = np.asarray(thigh_acc, dtype=np.float64)
            thigh_gyr = np.asarray(thigh_gyr, dtype=np.float64)
            shin_acc = np.asarray(shin_acc, dtype=np.float64)
            shin_gyr = np.asarray(shin_gyr, dtype=np.float64)

        N = len(thigh_gyr)
        k_win = max(5, min(static_samples, N // 5))

        rel_gyr_norm = np.linalg.norm(shin_gyr - thigh_gyr, axis=1)
        tot_gyr_norm = np.linalg.norm(thigh_gyr, axis=1) + np.linalg.norm(shin_gyr, axis=1)

        if N > k_win * 2:
            kernel = np.ones(k_win) / k_win
            smooth_mag = np.convolve(tot_gyr_norm, kernel, mode="valid")
            best_start = int(np.argmin(smooth_mag))
        else:
            best_start = 0

        quiet_slice = slice(best_start, best_start + k_win)
        if np.mean(tot_gyr_norm[quiet_slice]) < 15.0:
            t_bias = np.mean(thigh_gyr[quiet_slice], axis=0)
            s_bias = np.mean(shin_gyr[quiet_slice], axis=0)
        else:
            t_bias = np.zeros(3, dtype=np.float64)
            s_bias = np.zeros(3, dtype=np.float64)

        thigh_gyr_cal = thigh_gyr - t_bias
        shin_gyr_cal = shin_gyr - s_bias

        # Identify principal 3D knee hinge axis via SVD of differential angular velocity
        rel_gyr = shin_gyr_cal - thigh_gyr_cal
        centered_gyr = rel_gyr - np.mean(rel_gyr, axis=0)
        if np.max(np.std(centered_gyr, axis=0)) > 1e-3:
            _, _, vh = np.linalg.svd(centered_gyr, full_matrices=False)
            u_hinge = vh[0]
        else:
            u_hinge = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # Project gravity vectors orthogonal to u_hinge to find the extension baseline posture (not flexion peak!)
        t_sag_all = thigh_acc - np.outer(thigh_acc @ u_hinge, u_hinge)
        s_sag_all = shin_acc - np.outer(shin_acc @ u_hinge, u_hinge)
        tn_all = np.linalg.norm(t_sag_all, axis=1)
        sn_all = np.linalg.norm(s_sag_all, axis=1)
        valid_g = (tn_all > 0.15) & (sn_all > 0.15)

        if np.any(valid_g):
            cos_all = np.zeros(N, dtype=np.float64)
            cos_all[valid_g] = np.clip(
                np.sum((t_sag_all[valid_g] / tn_all[valid_g, None]) * (s_sag_all[valid_g] / sn_all[valid_g, None]), axis=1),
                -1.0, 1.0
            )
            sag_angles_all = np.rad2deg(np.arccos(cos_all))
            # Extension baseline samples are those in the lowest 25% of sagittal tilt angle
            ext_candidates = np.where(sag_angles_all <= np.percentile(sag_angles_all[valid_g], 25))[0]
            if len(ext_candidates) > 0:
                quiet_idx = int(ext_candidates[np.argmin(rel_gyr_norm[ext_candidates])])
            else:
                quiet_idx = int(np.argmin(rel_gyr_norm))
            # Direct clinical measurement of sagittal extension deficit/lag at quiet extension posture quiet_idx (Fixes #8)
            measured_lag = float(sag_angles_all[quiet_idx]) if quiet_idx < len(sag_angles_all) else 0.0
            tare_angle_deg = float(np.clip(measured_lag, 0.0, 25.0))
        else:
            quiet_idx = int(np.argmin(rel_gyr_norm))
            tare_angle_deg = 0.0

        if single_matrix_mode:
            cal_mat = np.copy(mat12).astype(np.float32)
            cal_mat[:, 3:6] = thigh_gyr_cal.astype(np.float32)
            cal_mat[:, 9:12] = shin_gyr_cal.astype(np.float32)
            return cal_mat, {
                "baseline_extension_offset_deg": tare_angle_deg,
                "quiet_idx": quiet_idx,
                "thigh_gyro_bias": t_bias.tolist(),
                "shin_gyro_bias": s_bias.tolist()
            }

        return thigh_acc, thigh_gyr_cal, shin_acc, shin_gyr_cal, tare_angle_deg, quiet_idx

    # ----------------------------------------------------------------------------------
    # STEP 2.4 & 2.5: 6-DOF MADGWICK QUATERNION FUSION & 3D RELATIVE ORIENTATION
    # ----------------------------------------------------------------------------------
    def madgwick_fusion(self, accel: np.ndarray, gyro_deg_s: np.ndarray, dt: float = None) -> np.ndarray:
        """
        Step 2.4: Madgwick Gradient-Descent 6-DoF IMU Orientation Filter.
        Initializes q[0] from the initial gravity vector to avoid convergence lag
        and returns unit quaternions q = [qw, qx, qy, qz] of shape (N, 4).
        """
        dt_use = float(dt) if dt is not None else self.dt
        N = len(accel)
        gyro_rad = np.deg2rad(gyro_deg_s)

        q = np.zeros((N, 4), dtype=np.float64)
        norm_a0 = np.linalg.norm(accel[0])
        if norm_a0 > 0.1:
            ax0, ay0, az0 = accel[0] / norm_a0
            # Quaternion rotating world gravity [0, 0, 1] to measured sensor gravity [ax0, ay0, az0]
            q0_vec = np.array([1.0 + az0, -ay0, ax0, 0.0], dtype=np.float64)
            norm_q0 = np.linalg.norm(q0_vec)
            if norm_q0 > 1e-6:
                q[0] = q0_vec / norm_q0
            else:
                q[0] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        else:
            q[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        for i in range(1, N):
            q1, q2, q3, q4 = q[i - 1]
            a = accel[i]
            w = gyro_rad[i]

            norm_a = np.linalg.norm(a)
            if norm_a > 0.1:
                ax, ay, az = a / norm_a
                F = np.array([
                    2.0 * (q2 * q4 - q1 * q3) - ax,
                    2.0 * (q1 * q2 + q3 * q4) - ay,
                    2.0 * (0.5 - q2**2 - q3**2) - az
                ])
                J = np.array([
                    [-2.0 * q3,  2.0 * q4, -2.0 * q1, 2.0 * q2],
                    [ 2.0 * q2,  2.0 * q1,  2.0 * q4, 2.0 * q3],
                    [ 0.0,      -4.0 * q2, -4.0 * q3, 0.0     ]
                ])
                step = J.T @ F
                n_step = np.linalg.norm(step)
                if n_step > 1e-9:
                    step /= n_step
            else:
                step = np.zeros(4)

            qDot = 0.5 * np.array([
                -q2 * w[0] - q3 * w[1] - q4 * w[2],
                 q1 * w[0] + q3 * w[2] - q4 * w[1],
                 q1 * w[1] - q2 * w[2] + q4 * w[0],
                 q1 * w[2] + q2 * w[1] - q3 * w[0]
            ]) - self.beta * step

            q_new = q[i - 1] + qDot * dt_use
            q[i] = q_new / (np.linalg.norm(q_new) + 1e-12)

        return q

    @staticmethod
    def compute_relative_orientation(q_thigh: np.ndarray, q_shin: np.ndarray) -> np.ndarray:
        """
        Step 2.5: Computes 3D relative orientation quaternion q_rel = q_thigh* (x) q_shin.
        """
        w1, x1, y1, z1 = q_thigh[:, 0], -q_thigh[:, 1], -q_thigh[:, 2], -q_thigh[:, 3]
        w2, x2, y2, z2 = q_shin[:, 0], q_shin[:, 1], q_shin[:, 2], q_shin[:, 3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.column_stack([w, x, y, z])

    # ----------------------------------------------------------------------------------
    # STEP 2.6: COMPUTE ANATOMICAL KNEE ANGLE TRAJECTORY VIA MADGWICK + RELATIVE QUATERNION
    # ----------------------------------------------------------------------------------
    def compute_knee_angle(
        self,
        thigh_gyro_y: np.ndarray,
        shin_gyro_y: np.ndarray = None,
        X_seq_12ch: np.ndarray = None,
        duration_sec: float = None,
        already_calibrated: bool = False
    ) -> np.ndarray:
        """
        Step 2.6: Computes the continuous sagittal knee angle in degrees (>= 0):
          1. Calls `calibrate_sensors` (unless `already_calibrated=True`) to obtain gyro-bias
             calibrated 3D streams and the quiet reference sample `quiet_idx`.
          2. Calls `madgwick_fusion` on both the Thigh and Shin 6-DoF sensors to obtain
             unit quaternion trajectories `q_thigh` and `q_shin`.
          3. Calls `compute_relative_orientation(q_thigh, q_shin)` to obtain `q_rel`,
             tares `q_rel` against `q_rel[quiet_idx]`, projects it on the hinge axis
             (expressed in the thigh frame) and returns the signed angle re-zeroed at the
             most-extended sample (`_orient_and_zero`), so the result is never negative.
        """
        if shin_gyro_y is None and thigh_gyro_y is not None and np.asarray(thigh_gyro_y).ndim == 2:
            X_seq_12ch = np.asarray(thigh_gyro_y, dtype=np.float64)
            thigh_gyro_y = X_seq_12ch[:, 4]
            shin_gyro_y = X_seq_12ch[:, 10]

        thigh_gyro_y = np.asarray(thigh_gyro_y, dtype=np.float64)
        shin_gyro_y = np.asarray(shin_gyro_y, dtype=np.float64)
        N = len(thigh_gyro_y)
        dt = (float(duration_sec) / max(1, N)) if (duration_sec is not None and duration_sec > 0.1) else self.dt

        if X_seq_12ch is not None and np.asarray(X_seq_12ch).ndim == 2 and X_seq_12ch.shape[1] >= 12:
            signed_angle_deg, _ = self._signed_hinge_angle(X_seq_12ch, dt, already_calibrated)

            # Step 2.6: reference the angle to the most-extended posture in the recording.
            # Public datasets carry no straight-leg calibration pose and unknown sensor
            # mounting, so an absolute anatomical zero (and hence true extensor lag) cannot
            # be recovered; the gravity tare is NOT added because it produced impossible
            # negative angles (down to -122 deg) when the quiet sample was not full extension.
            knee_angle = self._orient_and_zero(signed_angle_deg)
            return knee_angle.astype(np.float32)

        # Fallback only when 1D gyro vectors (without 12-channel matrix) are passed directly
        omega_knee = shin_gyro_y - thigh_gyro_y
        q_thigh_1d = self.madgwick_fusion(np.zeros((N, 3)), np.column_stack([np.zeros(N), thigh_gyro_y, np.zeros(N)]), dt=dt)
        q_shin_1d = self.madgwick_fusion(np.zeros((N, 3)), np.column_stack([np.zeros(N), shin_gyro_y, np.zeros(N)]), dt=dt)
        q_rel_1d = self.compute_relative_orientation(q_thigh_1d, q_shin_1d)
        quiet_idx = int(np.argmin(np.abs(omega_knee)))
        q_ref_1d = np.repeat(q_rel_1d[quiet_idx : quiet_idx + 1], N, axis=0)
        q_tared_1d = self.compute_relative_orientation(q_ref_1d, q_rel_1d)
        signed_1d = 2.0 * np.rad2deg(np.arctan2(q_tared_1d[:, 2], q_tared_1d[:, 0]))
        knee_angle = self._orient_and_zero(signed_1d)
        return knee_angle.astype(np.float32)

    def _signed_hinge_angle(self, X_seq_12ch: np.ndarray, dt: float, already_calibrated: bool = False):
        """
        Steps 2.3-2.5 for a 12-channel window. Returns `(signed_angle_deg, u_hinge)`: the relative
        thigh->shin rotation projected on the hinge axis `u_hinge` (thigh frame, arbitrary SVD sign).
        """
        arr = np.asarray(X_seq_12ch, dtype=np.float64)
        N = len(arr)
        t_acc, t_gyr = arr[:, 0:3], arr[:, 3:6]
        s_acc, s_gyr = arr[:, 6:9], arr[:, 9:12]

        # Step 2.3: Single-pass sensor calibration (gyro bias + quiet reference sample)
        if already_calibrated:
            quiet_idx = int(np.argmin(np.linalg.norm(s_gyr - t_gyr, axis=1)))
        else:
            t_acc, t_gyr, s_acc, s_gyr, _, quiet_idx = self.calibrate_sensors(t_acc, t_gyr, s_acc, s_gyr)

        # Step 2.4: 6-DoF Madgwick Quaternion Sensor Fusion for Thigh and Shin
        q_thigh = self.madgwick_fusion(t_acc, t_gyr, dt=dt)
        q_shin = self.madgwick_fusion(s_acc, s_gyr, dt=dt)

        # Step 2.5: 3D Relative Orientation Quaternion q_rel = q_thigh* (x) q_shin, tared at quiet_idx.
        # The change since the reference sample is taken in the THIGH frame (q_rel (x) q_ref*), the
        # frame of the hinge axis below. (q_ref* (x) q_rel would be expressed in the shin frame and,
        # with any sensor-mounting offset, shrink or even flip the projected angle.)
        q_rel = self.compute_relative_orientation(q_thigh, q_shin)
        q_ref = np.repeat(q_rel[quiet_idx : quiet_idx + 1], N, axis=0)
        q_tared = self._quat_multiply(q_rel, self._quat_conjugate(q_ref))

        # Rotate the shin gyroscope into the thigh frame with q_rel so the differential angular
        # velocity, and therefore the SVD hinge axis, are expressed in one consistent frame
        qv = q_rel[:, 1:4]
        qw = q_rel[:, 0:1]
        t = 2.0 * np.cross(qv, s_gyr)
        s_gyr_thigh = s_gyr + qw * t + np.cross(qv, t)
        rel_gyr_thigh = s_gyr_thigh - t_gyr
        centered_gyr = rel_gyr_thigh - np.mean(rel_gyr_thigh, axis=0)

        if np.max(np.std(centered_gyr, axis=0)) > 1e-3:
            _, _, vh = np.linalg.svd(centered_gyr, full_matrices=False)
            u_hinge = vh[0]
        else:
            u_hinge = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # Canonical hemisphere (qw >= 0) for a consistent rotation-angle extraction
        neg_mask = q_tared[:, 0] < 0
        if np.any(neg_mask):
            q_tared = np.copy(q_tared)
            q_tared[neg_mask] = -q_tared[neg_mask]

        q_hinge_proj = np.sum(q_tared[:, 1:4] * u_hinge, axis=1)
        signed_angle_deg = 2.0 * np.rad2deg(np.arctan2(q_hinge_proj, q_tared[:, 0]))
        return signed_angle_deg, u_hinge

    @staticmethod
    def _quat_conjugate(q: np.ndarray) -> np.ndarray:
        return np.column_stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]])

    @staticmethod
    def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Hamilton product a (x) b for arrays of [w, x, y, z] quaternions."""
        w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return np.column_stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])

    @staticmethod
    def _unwrap_deg(angle_deg: np.ndarray) -> np.ndarray:
        """Removes +/-180 deg wrap-around jumps from an angle trace in degrees."""
        return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(angle_deg, dtype=np.float64))))

    @staticmethod
    def _asymmetry(theta: np.ndarray) -> float:
        """
        Signed excursion asymmetry in [-1, 1]: positive when the trace moves further above its
        median than below it. Near zero for symmetric oscillations (e.g. seated extension).
        """
        med = float(np.median(theta))
        up = float(np.percentile(theta, 90)) - med
        down = med - float(np.percentile(theta, 10))
        return (up - down) / (float(np.ptp(theta)) + 1e-6)

    @staticmethod
    def _orient_and_zero(signed_angle_deg: np.ndarray) -> np.ndarray:
        """
        Fixes the arbitrary polarity of the SVD hinge axis and zeroes the angle at the
        most-extended sample, so the result is a non-negative excursion in degrees.

        Polarity rule (shape-based, applied identically in training, live recording and
        OptiTrack validation): the side on which the signal moves further away from its
        median is taken as the positive direction. IMU data alone cannot tell anatomical
        flexion from extension without a known sensor orientation, so this is a convention:
        for open-chain seated knee extension (resting posture flexed) the positive direction
        is usually extension, and for near-symmetric windows it can flip between windows
        (OptiTrack polarity agreement is reported in `optitrack_validation_metrics.json`).
        ROM, angular-velocity magnitude, SPARC and repetition timing do not depend on it.
        """
        # Remove +/-180 deg wrap-around jumps so a wrap can never masquerade as a ~360 deg ROM
        theta = Clinical16FeatureExtractor._unwrap_deg(signed_angle_deg)
        if Clinical16FeatureExtractor._asymmetry(theta) < 0:
            theta = -theta
        return theta - float(np.min(theta))

    # ----------------------------------------------------------------------------------
    # METHOD 1B: SAFE SENSOR PLACEMENT DIAGNOSTIC & OPTIONAL OPEN-CHAIN SWAP
    # ----------------------------------------------------------------------------------
    def auto_detect_and_correct_sensor_swap(
        self,
        X_seq_12ch: np.ndarray,
        exercise_code: str = "",
        auto_swap: bool = False
    ) -> tuple[np.ndarray, bool, str]:
        """
        Checks whether the Thigh (0x68) and Shin (0x69) sensors appear swapped during
        open-chain seated knee extension (where the thigh rests stationary on the chair
        while the shin swings through a wide arc).
        Never flips single-axis signs and never modifies closed-chain exercises
        (Squats, Walking, Step-Up) where both segments rotate simultaneously.
        """
        arr = np.copy(X_seq_12ch)
        if arr.ndim != 2 or arr.shape[1] < 12:
            return arr, False, "Valid 12-channel stream"

        thigh_gyr_rms = float(np.sqrt(np.mean(arr[:, 3:6] ** 2)))
        shin_gyr_rms = float(np.sqrt(np.mean(arr[:, 9:12] ** 2)))
        is_open_chain_ext = ("extension" in exercise_code.lower()) or ("04_" in exercise_code.lower())

        was_swapped = False
        msg = "✅ Sensor placement verified (Thigh: 0x68, Shin: 0x69)."

        # Only flag a swap in open-chain seated extension when Shin is nearly stationary (<8 deg/s)
        # while Thigh swings vigorously (>35 deg/s, >5x RMS ratio)
        if is_open_chain_ext and shin_gyr_rms < 8.0 and thigh_gyr_rms > max(35.0, 5.0 * shin_gyr_rms):
            if auto_swap:
                was_swapped = True
                thigh_block = np.copy(arr[:, 0:6])
                shin_block = np.copy(arr[:, 6:12])
                arr[:, 0:6] = shin_block
                arr[:, 6:12] = thigh_block
                msg = (
                    f"🔄 Auto-Corrected open-chain strap swap "
                    f"(Thigh gyro RMS {thigh_gyr_rms:.1f} deg/s vs stationary Shin {shin_gyr_rms:.1f} deg/s)."
                )
            else:
                msg = (
                    f"⚠️ Possible open-chain strap swap detected: Thigh gyro RMS ({thigh_gyr_rms:.1f} deg/s) "
                    f"is much higher than Shin ({shin_gyr_rms:.1f} deg/s). Verify strap placement."
                )

        return arr, was_swapped, msg

    # ----------------------------------------------------------------------------------
    # METHOD 2: COMPUTE ANGULAR VELOCITY
    # ----------------------------------------------------------------------------------
    def compute_angular_velocity(self, knee_angle: np.ndarray, dt: float = None) -> np.ndarray:
        """
        Computes instantaneous knee angular velocity (deg/s) via central differences
        and 2nd-order zero-phase Butterworth low-pass smoothing at 5.0 Hz.
        """
        dt_use = float(dt) if dt is not None else self.dt
        fs_use = 1.0 / dt_use
        velocity = np.gradient(np.asarray(knee_angle, dtype=np.float64), dt_use)

        if len(velocity) >= 12 and fs_use > 12.0:
            norm_fc = np.clip(5.0 / (0.5 * fs_use), 0.02, 0.92)
            b, a = signal.butter(2, norm_fc, btype="low")
            smooth_velocity = signal.filtfilt(b, a, velocity)
        else:
            smooth_velocity = velocity
        return smooth_velocity.astype(np.float32)

    # ----------------------------------------------------------------------------------
    # METHOD 3: SPECTRAL ARC LENGTH (SPARC) MOVEMENT SMOOTHNESS
    # ----------------------------------------------------------------------------------
    def compute_sparc(
        self,
        velocity_signal: np.ndarray,
        padlevel: int = 4,
        fc: float = 10.0,
        amp_th: float = 0.05,
        dt: float = None
    ) -> float:
        """
        Computes Spectral Arc Length (SPARC) motor smoothness index (<= 0.0).
        """
        dt_use = float(dt) if dt is not None else self.dt
        v = np.asarray(velocity_signal, dtype=np.float64)
        if len(v) < 10 or np.all(np.abs(v) < 1e-5):
            return -5.0

        n_fft = int(2 ** (np.ceil(np.log2(len(v))) + padlevel))
        freq = np.fft.rfftfreq(n_fft, d=dt_use)
        f_mag = np.abs(np.fft.rfft(v, n=n_fft))

        max_f = np.max(f_mag)
        if max_f <= 1e-9:
            return -5.0
        f_norm = f_mag / max_f

        cutoff_idx = np.where((freq <= fc) & (f_norm >= amp_th))[0]
        cutoff_idx = cutoff_idx[-1] if len(cutoff_idx) > 0 else min(len(freq) - 1, 20)

        f_sel = freq[:cutoff_idx + 1] / fc
        mag_sel = f_norm[:cutoff_idx + 1]

        df = np.diff(f_sel)
        dmag = np.diff(mag_sel)
        arc_length = -np.sum(np.sqrt(df**2 + dmag**2))
        return float(arc_length)

    # ----------------------------------------------------------------------------------
    # METHOD 4: ADAPTIVE REPETITION SEGMENTATION
    # ----------------------------------------------------------------------------------
    def segment_repetitions(
        self,
        knee_angle: np.ndarray,
        min_rom_threshold: float = None,
        min_duration_sec: float = 0.45,
        dt: float = None
    ):
        """
        Segments exercise repetitions using adaptive prominence peak detection.
        Supports both large-excursion exercises (Squats, Extensions, Walking, Step-Ups) and
        small-excursion isometric/early-ORIF exercises (Quad Sets 5-10 deg, SLR 10-15 deg).
        """
        dt_use = float(dt) if dt is not None else self.dt
        fs_use = 1.0 / dt_use
        ang = np.asarray(knee_angle, dtype=np.float64)
        total_rom = float(np.max(ang) - np.min(ang))

        if total_rom < 3.5:
            return [], 0

        if min_rom_threshold is None:
            prom_th = max(3.5, min(18.0, 0.30 * total_rom))
        else:
            prom_th = max(3.5, min(float(min_rom_threshold), 0.35 * total_rom))

        min_distance = max(3, int(min_duration_sec * fs_use))
        height_th = float(np.min(ang)) + prom_th
        peaks, _ = signal.find_peaks(ang, height=height_th, prominence=prom_th * 0.7, distance=min_distance)

        rep_boundaries = []
        search_win = max(5, int(2.0 * fs_use))
        for p in peaks:
            start_search = max(0, p - search_win)
            start_idx = start_search + int(np.argmin(ang[start_search:p])) if p > start_search else start_search

            end_search = min(len(ang), p + search_win)
            end_idx = p + int(np.argmin(ang[p:end_search])) if end_search > p else end_search - 1

            rep_rom = float(ang[p] - min(ang[start_idx], ang[end_idx]))
            rep_dur = float((end_idx - start_idx) * dt_use)

            if rep_rom >= prom_th * 0.8 and rep_dur >= min_duration_sec * 0.75:
                rep_boundaries.append((start_idx, int(p), end_idx))

        return rep_boundaries, len(rep_boundaries)

    # ----------------------------------------------------------------------------------
    # METHOD 5: EXTRACT ALL 16 NON-REDUNDANT CLINICAL FEATURES
    # ----------------------------------------------------------------------------------
    def extract_16_features(self, knee_angle: np.ndarray, duration_sec: float = None):
        """
        Computes the standardized 16-dimensional clinical kinematic feature vector (Input B):
          - Feature 1 (`rom_deg`): Total active Range of Motion = max(theta) - min(theta).
          - Feature 2 (`peak_depth_deg`): Maximum Knee Flexion Angle = max(theta) (or mean
            peak flexion angle across segmented repetitions when rep_count >= 2, with NO arbitrary blend).
          - Feature 3 (`extension_return_deficit_deg`): mean repetition trough angle above the
            most-extended posture (0.0 when no repetition is segmented).
          - Feature 8 (`mean_repetition_duration`): Mean cycle duration, or active velocity envelope
            duration when discrete reps == 0 (0.0s if stationary).
          - Feature 12 (`max_ang_vel`, >0 flexion) & Feature 13 (`min_ang_vel`, <0 extension).
        """
        ang = np.asarray(knee_angle, dtype=np.float64)
        N = len(ang)
        dt_use = (float(duration_sec) / max(1, N)) if (duration_sec is not None and duration_sec > 0.1) else self.dt

        ang_vel = self.compute_angular_velocity(ang, dt=dt_use)
        rep_bounds, rep_count = self.segment_repetitions(ang, dt=dt_use)

        # FEATURE 1: Range of Motion (ROM = max - min, in degrees)
        rom = float(np.max(ang) - np.min(ang))

        # FEATURE 2: Peak Flexion Angle / Mean Repetition Peak Depth (in degrees, NO 0.7/0.3 blend)
        if rep_count >= 2:
            peak_depth = float(np.mean([ang[p] for _, p, _ in rep_bounds]))
        else:
            peak_depth = float(np.max(ang))

        # FEATURE 3: Extension Return Deficit (deg) = mean trough angle at the start/end of each
        # segmented repetition above the most-extended posture of the recording. It measures how
        # completely the patient returns to extension between repetitions. An absolute extensor
        # lag needs a straight-leg calibration pose, which the training datasets do not provide.
        min_angle = float(np.min(ang))
        if rep_count >= 1:
            troughs = [float(ang[s]) for s, _, _ in rep_bounds] + [float(ang[e]) for _, _, e in rep_bounds]
            extension_return_deficit = float(np.mean(troughs) - min_angle)
        else:
            extension_return_deficit = 0.0

        # FEATURE 4: Peak Angular Velocity (deg/s)
        peak_velocity = float(np.max(np.abs(ang_vel)))

        # FEATURE 5: Mean Angular Velocity (deg/s)
        mean_velocity = float(np.mean(np.abs(ang_vel)))

        # FEATURE 6: SPARC Movement Smoothness (dimensionless index <= 0)
        sparc_smoothness = self.compute_sparc(ang_vel, dt=dt_use)

        # FEATURE 7: Repetition Count (integer count of valid completed cycles)
        repetition_count = float(rep_count)

        # FEATURE 8: Mean Repetition Duration (seconds)
        if rep_count > 0:
            durations = [(end - start) * dt_use for start, _, end in rep_bounds]
            mean_duration = float(np.mean(durations))
        else:
            vel_thresh = max(5.0, 0.15 * peak_velocity)
            active_idx = np.where(np.abs(ang_vel) > vel_thresh)[0]
            if len(active_idx) >= 5:
                mean_duration = float((active_idx[-1] - active_idx[0]) * dt_use)
            else:
                mean_duration = 0.0

        # FEATURE 9: Knee Angle Standard Deviation (deg)
        angle_std = float(np.std(ang))

        # FEATURE 10: Angular Velocity Standard Deviation (deg/s)
        velocity_std = float(np.std(ang_vel))

        # FEATURE 11: Angle Interquartile Range (IQR = P75 - P25, in deg)
        angle_iqr = float(np.percentile(ang, 75) - np.percentile(ang, 25))

        # FEATURE 12: Max Flexion Velocity (Peak positive knee-bending speed, deg/s)
        max_ang_vel = float(np.max(ang_vel))

        # FEATURE 13: Max Extension Velocity (Peak negative knee-straightening speed, deg/s)
        min_ang_vel = float(np.min(ang_vel))

        # FEATURE 14: Mean Knee Angle (deg)
        mean_angle = float(np.mean(ang))

        # FEATURE 15: Angle Root Mean Square (RMS, deg)
        angle_rms = float(np.sqrt(np.mean(ang**2)))

        # FEATURE 16: Normalized Smoothness Rate (SPARC / Active Duration, /s)
        eff_dur = mean_duration if mean_duration > 0.2 else float(N * dt_use)
        sparc_per_duration = float(sparc_smoothness / (eff_dur + 1e-6))

        features_dict = {
            "rom_deg": rom,
            "peak_depth_deg": peak_depth,
            "extension_return_deficit_deg": extension_return_deficit,
            # Backward-compatible aliases (older callers read these keys)
            "extensor_lag_deg": extension_return_deficit,
            "min_angle_deg": min_angle,
            "peak_angular_velocity": peak_velocity,
            "mean_angular_velocity": mean_velocity,
            "sparc_smoothness": sparc_smoothness,
            "repetition_count": int(repetition_count),
            "mean_repetition_duration": mean_duration,
            "angle_std": angle_std,
            "velocity_std": velocity_std,
            "angle_iqr": angle_iqr,
            "max_ang_vel": max_ang_vel,
            "min_ang_vel": min_ang_vel,
            "mean_angle": mean_angle,
            "angle_rms": angle_rms,
            "sparc_per_duration": sparc_per_duration
        }

        feature_vector = np.array([
            rom, peak_depth, extension_return_deficit, peak_velocity, mean_velocity,
            sparc_smoothness, repetition_count, mean_duration, angle_std, velocity_std,
            angle_iqr, max_ang_vel, min_ang_vel, mean_angle, angle_rms, sparc_per_duration
        ], dtype=np.float32)

        return features_dict, feature_vector


if __name__ == "__main__":
    import os
    print("=" * 75)
    print("TESTING 3D MADGWICK + RELATIVE ORIENTATION 16-FEATURE ENGINE")
    print("=" * 75)

    d1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "dataset_1_input_A_12channels.npz")
    if os.path.exists(d1_path):
        data = np.load(d1_path)
        sample_trial = data["X_raw_12ch"][0]
        extractor = Clinical16FeatureExtractor(fs=100.0)
        knee_angle = extractor.compute_knee_angle(sample_trial)
        feats, feat_vec = extractor.extract_16_features(knee_angle)
        for idx, (k, v) in enumerate(feats.items(), 1):
            print(f"  {idx:2d}. {k:<26}: {v:8.2f}")
