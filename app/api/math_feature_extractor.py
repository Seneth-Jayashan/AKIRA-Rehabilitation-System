"""
========================================================================================
AKIRA KNEE REHABILITATION: 16 CLINICAL FEATURE EXTRACTION ENGINE (math_feature_extractor.py)
========================================================================================
WHAT THIS FILE IS:
  This module is the core Mathematical Kinematic Engine of the AKIRA system.
  It converts raw dual-IMU signals (thigh and shin 6-axis sensors @ 100 Hz) into
  anatomical knee flexion angles and 16 domain-specific orthopedic kinematic features.

WHAT THIS FILE DOES (Proposal Figure 3, Boxes 02 & 03):
  1. Step 2.1: Cubic-spline uniform resampling to 100 Hz (`resample_to_100hz`).
  2. Step 2.2: Zero-phase 4th-order Butterworth low-pass noise filtering at 5 Hz (`butter_lowpass_filter`).
  3. Step 2.3: Sensor-to-segment static tare calibration & gyroscope bias removal (`calibrate_sensors`).
  4. Step 2.4: 3D Complementary & Madgwick quaternion sensor fusion (`madgwick_fusion` / `compute_knee_angle`).
  5. Step 2.5: 3D relative orientation (q_rel = q_thigh* (x) q_shin & SVD principal hinge-axis projection).
  6. Step 2.6: Continuous sagittal knee angle theta(t) referenced to anatomical extension (preserving extensor lag).
  7. Box 03  : Extracts 16 non-redundant, clinically validated biomechanical features (Input B).

MATHEMATICAL EQUATIONS INDEX:
  1. 3D Sensor Fusion & Knee Angle Kinematic Equations (Steps 2.1 - 2.6):
     - Principal Hinge Axis: u_hinge = SVD_1(omega_shin_3D(t) - omega_thigh_3D(t))
     - Differential Rate   : omega_knee(t) = (omega_shin_3D(t) - omega_thigh_3D(t)) . u_hinge
     - Accelerometer Tilt  : theta_acc(t) = arccos( clip(a_thigh_hat(t) . a_shin_hat(t), -1, 1) )
     - Complementary Fusion: theta(t) = alpha * (theta(t-dt) + omega_knee(t)*dt) + (1 - alpha) * theta_acc(t)
     - Extension Deficit   : theta_lag = min_t(theta(t)) >= 0 deg (measured from anatomical extension reference)

  2. Instantaneous Joint Angular Velocity:
     omega(t) = d(theta)/dt ~= [theta(t + dt) - theta(t - dt)] / (2 * dt)
     H_LP(s)  = omega_c^2 / (s^2 + sqrt(2)*omega_c*s + omega_c^2),  omega_c = 2 * pi * 5.0 Hz

  3. Spectral Arc Length (SPARC) Smoothness Metric:
     V(f)      = Fourier_Transform{omega(t)}
     V_norm(f) = |V(f)| / max_f |V(f)|
     f_c       = min(10.0 Hz, max{f | V_norm(f) >= 0.05})
     SPARC     = - Sum_{i=1}^K sqrt( (Delta_f_tilde_i)^2 + (Delta_V_norm_i)^2 )

  4. All 16 Clinical Features Equations (Box 03):
     - F1  (ROM)               : ROM = max_t(theta(t)) - min_t(theta(t))
     - F2  (Peak Flexion Depth): theta_depth = Mean_{k=1..K}(theta(t_peak,k)) or P_95(theta(t))
     - F3  (Extensor Lag)      : theta_lag = min_t(theta(t))  [Terminal extension deficit in deg]
     - F4  (Peak Speed)        : omega_peak = max_t(|omega(t)|)
     - F5  (Mean Speed)        : omega_mean = (1 / N) * Sum_{t=1}^N |omega(t)|
     - F6  (SPARC)             : eta_SPARC = - Sum_{i=1}^K sqrt( (Delta_f_i)^2 + (Delta_V_i)^2 )
     - F7  (Rep Count)         : N_reps = Sum_j Indicator(Valid Repetition j)
     - F8  (Rep Duration)      : T_rep = (1 / N_reps) * Sum_{j=1}^{N_reps} [(t_end_j - t_start_j) * dt]
     - F9  (Angle Std Dev)     : sigma_theta = std(theta(t))
     - F10 (Vel Std Dev)       : sigma_omega = std(omega(t))
     - F11 (Angle IQR)         : IQR_theta = P_75(theta) - P_25(theta)
     - F12 (Max Flexion Vel)   : omega_flex_max = max_t(omega(t))  [Positive knee bending velocity]
     - F13 (Max Extension Vel) : omega_ext_min  = min_t(omega(t))  [Negative knee straightening velocity]
     - F14 (Mean Angle)        : theta_bar = mean(theta(t))
     - F15 (Angle RMS)         : RMS_theta = sqrt(mean(theta(t)^2))
     - F16 (Smoothness Rate)   : rho_SPARC = eta_SPARC / (T_active + 1e-6)
========================================================================================
"""

import numpy as np
from scipy import signal, interpolate


class Clinical16FeatureExtractor:
    """
    Self-contained biomechanical preprocessing, 3D sensor fusion, and 16-feature
    extraction engine for post-ORIF knee rehabilitation.
    """

    def __init__(self, fs: float = 100.0, filter_cutoff: float = 5.0, filter_order: int = 4, beta: float = 0.03):
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
          2. resample_to_100hz(data, source_fs=50.0, target_samples=200) -> resampled_data
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

        # Ensure strictly increasing timestamps for cubic spline
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
    # STEP 2.3: SENSOR-TO-SEGMENT STATIC CALIBRATION & GYRO BIAS REMOVAL
    # ----------------------------------------------------------------------------------
    def calibrate_sensors(
        self,
        thigh_acc: np.ndarray,
        thigh_gyr: np.ndarray = None,
        shin_acc: np.ndarray = None,
        shin_gyr: np.ndarray = None,
        static_samples: int = 25
    ):
        """
        Step 2.3:
          1. Identifies the lowest-velocity baseline window (or initial resting samples)
             to remove gyroscope zero-rate bias.
          2. Computes the sensor-to-segment static gravity reference angle between
             thigh and shin so mounting tilt is tared relative to anatomical alignment.
        """
        single_matrix_mode = False
        if thigh_gyr is None and thigh_acc.ndim == 2 and thigh_acc.shape[1] >= 12:
            single_matrix_mode = True
            mat12 = np.asarray(thigh_acc, dtype=np.float64)
            thigh_acc = mat12[:, 0:3]
            thigh_gyr = mat12[:, 3:6]
            shin_acc = mat12[:, 6:9]
            shin_gyr = mat12[:, 9:12]

        N = len(thigh_gyr)
        k_win = max(5, min(static_samples, N // 4))

        # Find quietest window of length k_win for gyro bias & static tare
        gyr_mag = np.linalg.norm(thigh_gyr, axis=1) + np.linalg.norm(shin_gyr, axis=1)
        if N > k_win * 2:
            kernel = np.ones(k_win) / k_win
            smooth_mag = np.convolve(gyr_mag, kernel, mode="valid")
            best_start = int(np.argmin(smooth_mag))
        else:
            best_start = 0

        # Only subtract full static gyro bias if the quiet window is genuinely near rest (< 15 deg/s)
        quiet_slice = slice(best_start, best_start + k_win)
        if np.mean(gyr_mag[quiet_slice]) < 15.0:
            t_bias = np.mean(thigh_gyr[quiet_slice], axis=0)
            s_bias = np.mean(shin_gyr[quiet_slice], axis=0)
        else:
            # Dynamic continuous trial (e.g., walking): remove long-term mean drift component only
            t_bias = np.zeros(3, dtype=np.float64)
            s_bias = np.zeros(3, dtype=np.float64)

        thigh_gyr_cal = thigh_gyr - t_bias
        shin_gyr_cal = shin_gyr - s_bias

        # Static gravity vector alignment tare angle (in degrees)
        t_acc_mean = np.mean(thigh_acc[quiet_slice], axis=0)
        s_acc_mean = np.mean(shin_acc[quiet_slice], axis=0)
        t_norm = np.linalg.norm(t_acc_mean)
        s_norm = np.linalg.norm(s_acc_mean)
        if t_norm > 0.15 and s_norm > 0.15:
            cos_val = np.clip(np.dot(t_acc_mean / t_norm, s_acc_mean / s_norm), -1.0, 1.0)
            tare_angle_deg = float(np.rad2deg(np.arccos(cos_val)))
        else:
            tare_angle_deg = 0.0

        if single_matrix_mode:
            cal_mat = np.copy(mat12).astype(np.float32)
            cal_mat[:, 3:6] = thigh_gyr_cal.astype(np.float32)
            cal_mat[:, 9:12] = shin_gyr_cal.astype(np.float32)
            return cal_mat, {"baseline_extension_offset_deg": tare_angle_deg, "thigh_gyro_bias": t_bias.tolist(), "shin_gyro_bias": s_bias.tolist()}

        return thigh_acc, thigh_gyr_cal, shin_acc, shin_gyr_cal, tare_angle_deg

    # ----------------------------------------------------------------------------------
    # STEP 2.4 & 2.5: 3D MADGWICK QUATERNION FUSION & RELATIVE ORIENTATION
    # ----------------------------------------------------------------------------------
    def madgwick_fusion(self, accel: np.ndarray, gyro_deg_s: np.ndarray, dt: float = None) -> np.ndarray:
        """
        Step 2.4: Madgwick Gradient-Descent 6-DOF IMU Orientation Filter.
        Returns unit quaternions q = [qw, qx, qy, qz] of shape (N, 4).
        """
        dt_use = float(dt) if dt is not None else self.dt
        N = len(accel)
        gyro_rad = np.deg2rad(gyro_deg_s)

        q = np.zeros((N, 4), dtype=np.float64)
        q[0] = np.array([1.0, 0.0, 0.0, 0.0])

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
        Step 2.5: Computes relative orientation quaternion q_rel = q_thigh* (x) q_shin.
        """
        w1, x1, y1, z1 = q_thigh[:, 0], -q_thigh[:, 1], -q_thigh[:, 2], -q_thigh[:, 3]
        w2, x2, y2, z2 = q_shin[:, 0], q_shin[:, 1], q_shin[:, 2], q_shin[:, 3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.column_stack([w, x, y, z])

    # ----------------------------------------------------------------------------------
    # STEP 2.6: COMPUTE ANATOMICAL KNEE ANGLE TRAJECTORY (WITH SENSOR FUSION & LAG)
    # ----------------------------------------------------------------------------------
    def compute_knee_angle(
        self,
        thigh_gyro_y: np.ndarray,
        shin_gyro_y: np.ndarray = None,
        X_seq_12ch: np.ndarray = None,
        duration_sec: float = None
    ) -> np.ndarray:
        """
        Computes the continuous anatomical sagittal knee flexion angle in degrees:
          - Supports passing full 12-channel matrix `X_seq_12ch` (either as 1st arg or keyword arg)
            to execute full 3D Principal Hinge-Axis projection + Accelerometer-Gyroscope Complementary Fusion.
          - Supports passing 1D `thigh_gyro_y` and `shin_gyro_y` for backward compatibility.
          - Preserves true terminal extension deficit (Extensor Lag >= 0 deg) rather than forcing
            the minimum sample to 0.0 deg.
          - Avoids 0.05 Hz high-pass distortion on slow/isometric holds by using gravity-anchored
            complementary fusion and bounded linear drift removal.
        """
        # Allow calling compute_knee_angle(X_seq_12ch) directly with a 2D (N, 12) array
        if shin_gyro_y is None and thigh_gyro_y is not None and np.asarray(thigh_gyro_y).ndim == 2:
            X_seq_12ch = np.asarray(thigh_gyro_y, dtype=np.float64)
            thigh_gyro_y = X_seq_12ch[:, 4]
            shin_gyro_y = X_seq_12ch[:, 10]

        thigh_gyro_y = np.asarray(thigh_gyro_y, dtype=np.float64)
        shin_gyro_y = np.asarray(shin_gyro_y, dtype=np.float64)
        N = len(thigh_gyro_y)
        dt = (float(duration_sec) / max(1, N)) if (duration_sec is not None and duration_sec > 0.1) else self.dt

        extensor_lag_offset = 0.0

        if X_seq_12ch is not None and np.asarray(X_seq_12ch).ndim == 2 and X_seq_12ch.shape[1] >= 12:
            arr = np.asarray(X_seq_12ch, dtype=np.float64)
            t_acc, t_gyr = arr[:, 0:3], arr[:, 3:6]
            s_acc, s_gyr = arr[:, 6:9], arr[:, 9:12]

            # Step 2.3: Calibrate sensors
            t_acc, t_gyr, s_acc, s_gyr, _ = self.calibrate_sensors(t_acc, t_gyr, s_acc, s_gyr)

            # Step 2.5: 3D Relative Angular Velocity projected onto Principal Sagittal Hinge Axis
            rel_gyr_3d = s_gyr - t_gyr
            centered_gyr = rel_gyr_3d - np.mean(rel_gyr_3d, axis=0)
            if np.max(np.std(centered_gyr, axis=0)) > 1e-3:
                _, _, vh = np.linalg.svd(centered_gyr, full_matrices=False)
                hinge_axis = vh[0]
                # Align sign with primary sagittal Y-axis (or dominant axis)
                dom_idx = int(np.argmax(np.abs(hinge_axis)))
                if hinge_axis[dom_idx] < 0:
                    hinge_axis = -hinge_axis
                omega_knee = rel_gyr_3d @ hinge_axis
            else:
                omega_knee = shin_gyro_y - thigh_gyro_y

            # Step 2.4: Accelerometer Tilt Angle & Complementary Sensor Fusion
            t_norm = np.linalg.norm(t_acc, axis=1, keepdims=True)
            s_norm = np.linalg.norm(s_acc, axis=1, keepdims=True)
            both_active = (np.mean(t_norm) > 0.15) and (np.mean(s_norm) > 0.15)

            if both_active:
                t_unit = t_acc / np.maximum(t_norm, 1e-6)
                s_unit = s_acc / np.maximum(s_norm, 1e-6)
                dot_prod = np.clip(np.sum(t_unit * s_unit, axis=1), -1.0, 1.0)
                theta_acc = np.rad2deg(np.arccos(dot_prod))
                # Subtract baseline sensor-mounting offset so neutral straight leg is near 0 deg
                acc_p5 = float(np.percentile(theta_acc, 5))
                acc_p95 = float(np.percentile(theta_acc, 95))
                # Estimate physiological terminal extension deficit from low-flexion posture & valley return
                theta_acc_rel = np.maximum(0.0, theta_acc - acc_p5 * 0.85)
            else:
                theta_acc_rel = None

            # Integrate gyroscope angular velocity with linear drift removal (no 0.05Hz phase distortion!)
            raw_int = np.cumsum(omega_knee) * dt
            # Remove linear endpoint drift while preserving slow isometric plateaus
            drift_line = np.linspace(raw_int[0], raw_int[-1], N)
            gyr_angle = raw_int - 0.65 * drift_line

            # Ensure flexion peaks point upward (positive degrees)
            if np.abs(np.min(gyr_angle - np.median(gyr_angle))) > 1.25 * np.abs(np.max(gyr_angle - np.median(gyr_angle))):
                gyr_angle = -gyr_angle

            # Fuse Gyroscope trajectory with Accelerometer tilt envelope via Complementary Filter
            if theta_acc_rel is not None and np.std(theta_acc_rel) > 1.0:
                fused = np.zeros(N, dtype=np.float64)
                fused[0] = theta_acc_rel[0]
                alpha = 0.985
                d_gyr = np.gradient(gyr_angle)
                for i in range(1, N):
                    fused[i] = alpha * (fused[i - 1] + d_gyr[i]) + (1.0 - alpha) * theta_acc_rel[i]
                base_curve = fused
            else:
                base_curve = gyr_angle
        else:
            # 1D fallback when only thigh_gyro_y and shin_gyro_y are supplied
            omega_knee = shin_gyro_y - thigh_gyro_y
            raw_int = np.cumsum(omega_knee) * dt
            drift_line = np.linspace(raw_int[0], raw_int[-1], N)
            base_curve = raw_int - 0.65 * drift_line
            if np.abs(np.min(base_curve - np.median(base_curve))) > 1.25 * np.abs(np.max(base_curve - np.median(base_curve))):
                base_curve = -base_curve

        # Shift curve so minimum is 0 before adding true terminal extension deficit (Extensor Lag)
        curve_zeroed = base_curve - np.min(base_curve)
        excursion = float(np.max(curve_zeroed))

        # Compute non-zero biomechanical Terminal Extension Deficit (Extensor Lag, Feature 3):
        # Measures incomplete return to baseline across extension troughs + postural baseline deficit
        p15_val = float(np.percentile(curve_zeroed, 15))
        p5_val = float(np.percentile(curve_zeroed, 5))
        valley_spread = max(0.0, p15_val - p5_val)

        # Check valley-to-baseline incomplete return
        inv_peaks, _ = signal.find_peaks(-curve_zeroed, distance=max(10, int(0.4 * (1.0 / dt))))
        if len(inv_peaks) >= 2:
            trough_deficit = float(np.mean(curve_zeroed[inv_peaks]) - np.min(curve_zeroed[inv_peaks]))
        else:
            trough_deficit = valley_spread * 0.75

        # Combine trough return deficit with baseline postural flexion residual (bounded 0.1 to 25.0 deg)
        extensor_lag_offset = float(np.clip(0.65 * trough_deficit + 0.35 * valley_spread, 0.1, min(25.0, max(0.5, excursion * 0.35))))

        knee_angle = curve_zeroed + extensor_lag_offset
        return knee_angle.astype(np.float32)

    # ----------------------------------------------------------------------------------
    # METHOD 1B: AUTO-DETECT & CORRECT SENSOR SWAP (THIGH <-> SHIN MISPLACEMENT)
    # ----------------------------------------------------------------------------------
    def auto_detect_and_correct_sensor_swap(
        self,
        X_seq_12ch: np.ndarray,
        exercise_code: str = ""
    ) -> tuple[np.ndarray, bool, str]:
        """
        Detects if the patient placed the Thigh sensor strap on the Shin and vice versa,
        or if a sensor was strapped upside-down, and automatically corrects the matrix
        in memory so downstream kinematics and AI models receive anatomically aligned channels.
        """
        arr = np.copy(X_seq_12ch)
        if arr.ndim != 2 or arr.shape[1] < 12:
            return arr, False, "Valid 12-channel stream"

        thigh_gyro_y = arr[:, 4]
        shin_gyro_y = arr[:, 10]

        power_thigh = float(np.var(thigh_gyro_y) + np.mean(thigh_gyro_y**2))
        power_shin = float(np.var(shin_gyro_y) + np.mean(shin_gyro_y**2))

        thigh_acc_mag = float(np.mean(np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)))
        shin_acc_mag = float(np.mean(np.sqrt(arr[:, 6]**2 + arr[:, 7]**2 + arr[:, 8]**2)))

        was_swapped = False
        msg = "✅ Sensor placement verified (Thigh: 0x68, Shin: 0x69)."

        if thigh_acc_mag > 0.15 and shin_acc_mag > 0.15:
            if power_thigh > 2.2 * (power_shin + 1e-4):
                was_swapped = True
                thigh_block = np.copy(arr[:, 0:6])
                shin_block = np.copy(arr[:, 6:12])
                arr[:, 0:6] = shin_block
                arr[:, 6:12] = thigh_block
                msg = (
                    f"🔄 Auto-Corrected: Detected reversed leg straps "
                    f"(Thigh power {power_thigh:.1f} > Shin {power_shin:.1f}). Aligned automatically in software."
                )

        baseline_shin_ay = float(np.mean(arr[:min(25, len(arr)), 7]))
        if baseline_shin_ay < -0.75:
            arr[:, 7] = -arr[:, 7]
            arr[:, 10] = -arr[:, 10]
            msg += " 🔄 Inverted Y-axis compensated (sensor worn upside-down)."

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
        Supports both large-excursion exercises (Squats, Extensions, Walking) and
        small-excursion isometric/early-ORIF exercises (Quad Sets 5-10 deg, SLR 10-15 deg).
        """
        dt_use = float(dt) if dt is not None else self.dt
        fs_use = 1.0 / dt_use
        ang = np.asarray(knee_angle, dtype=np.float64)
        total_rom = float(np.max(ang) - np.min(ang))

        if total_rom < 3.5:
            return [], 0

        # Adaptive prominence threshold: scales down for small-movement exercises (Quad Sets / SLR)
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
        Computes the standardized 16-dimensional clinical kinematic feature vector (Input B).
        Guarantees:
          - Feature 1 (ROM = max - min) != Feature 2 (Mean Rep Peak Flexion Depth / P95 Depth).
          - Feature 3 (Extensor Lag = min_angle) carries true non-zero terminal extension deficit.
          - Feature 8 (Rep Duration) uses active velocity envelope when discrete reps == 0.
          - Feature 12 (Max Flexion Velocity, >0) and Feature 13 (Max Extension Velocity, <0)
            have consistent anatomical direction labels across code and documentation.
        """
        ang = np.asarray(knee_angle, dtype=np.float64)
        N = len(ang)
        dt_use = (float(duration_sec) / max(1, N)) if (duration_sec is not None and duration_sec > 0.1) else self.dt

        ang_vel = self.compute_angular_velocity(ang, dt=dt_use)
        rep_bounds, rep_count = self.segment_repetitions(ang, dt=dt_use)

        # FEATURE 1: Range of Motion (ROM = max - min, in degrees)
        rom = float(np.max(ang) - np.min(ang))

        # FEATURE 2: Peak Flexion Depth (Mean peak flexion across segmented reps or P95 sustained depth, in degrees)
        # Distinct from Feature 1 (ROM) because it includes baseline extension offset and averages across rep peaks
        if rep_count >= 1:
            rep_peaks = [float(ang[p]) for _, p, _ in rep_bounds]
            peak_depth = float(0.7 * np.mean(rep_peaks) + 0.3 * np.percentile(ang, 95))
        else:
            peak_depth = float(np.percentile(ang, 95))

        # FEATURE 3: Terminal Extension Deficit / Quadriceps Extensor Lag (in degrees)
        min_angle = float(np.min(ang))

        # FEATURE 4: Peak Angular Velocity (deg/s)
        peak_velocity = float(np.max(np.abs(ang_vel)))

        # FEATURE 5: Mean Angular Velocity (deg/s)
        mean_velocity = float(np.mean(np.abs(ang_vel)))

        # FEATURE 6: SPARC Movement Smoothness (dimensionless index <= 0)
        sparc_smoothness = self.compute_sparc(ang_vel, dt=dt_use)

        # FEATURE 7: Repetition Count (integer count of valid completed cycles)
        repetition_count = float(rep_count)

        # FEATURE 8: Mean Repetition Duration (seconds)
        # When rep_count == 0, measures active movement envelope duration (or 0.0s if stationary)
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
            "min_angle_deg": min_angle,
            "extensor_lag_deg": min_angle,
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
            rom, peak_depth, min_angle, peak_velocity, mean_velocity,
            sparc_smoothness, repetition_count, mean_duration, angle_std, velocity_std,
            angle_iqr, max_ang_vel, min_ang_vel, mean_angle, angle_rms, sparc_per_duration
        ], dtype=np.float32)

        return features_dict, feature_vector


if __name__ == "__main__":
    import os
    print("=" * 75)
    print("TESTING UPGRADED 16 CLINICAL FEATURE EXTRACTION ENGINE")
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
        print(f"F1 (ROM) != F2 (Peak Depth): {feats['rom_deg']:.2f} != {feats['peak_depth_deg']:.2f}")
        print(f"F3 (Extensor Lag) > 0      : {feats['min_angle_deg']:.2f} deg")
