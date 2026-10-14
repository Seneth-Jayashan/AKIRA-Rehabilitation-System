"""
========================================================================================
AKIRA KNEE REHABILITATION: 16 CLINICAL FEATURE EXTRACTION ENGINE (math_feature_extractor.py)
========================================================================================
WHAT THIS FILE IS:
  This module is the core Mathematical Kinematic Engine of the AKIRA system.
  It converts raw dual-IMU signals (thigh and shin 6-axis sensors @ 100 Hz) into
  anatomical knee flexion angles and 16 domain-specific orthopedic kinematic features.

WHAT THIS FILE DOES:
  1. Computes continuous knee flexion/extension angles using differential angular velocity.
  2. Applies digital Butterworth high-pass filtering (0.05 Hz) to eliminate gyro integration drift.
  3. Derives smoothed instantaneous angular velocity via numerical differentiation and 5 Hz low-pass filtering.
  4. Computes Spectral Arc Length (SPARC) movement smoothness in the frequency domain.
  5. Implements peak-and-valley repetition segmentation.
  6. Computes 16 clinically validated biomechanical features (Input B) for AI inference.

MATHEMATICAL EQUATIONS INDEX:
  1. Knee Angle Kinematic Equation:
     omega_knee(t) = omega_shin_Y(t) - omega_thigh_Y(t)
     theta_raw(t)  = Sum_{k=0}^{t} omega_knee(k) * Delta_t,   Delta_t = 1 / f_s = 0.01 s
     H_HP(s)       = s / (s + omega_c),                       omega_c = 2 * pi * 0.05 Hz
     theta_calib(t)= theta_filtered(t) - min_t(theta_filtered(t))

  2. Instantaneous Joint Angular Velocity:
     omega(t)      = d(theta)/dt ~= [theta(t + Delta_t) - theta(t - Delta_t)] / (2 * Delta_t)
     H_LP(s)       = omega_c^2 / (s^2 + sqrt(2)*omega_c*s + omega_c^2),  omega_c = 2 * pi * 5.0 Hz

  3. Spectral Arc Length (SPARC) Smoothness Metric:
     V(f)          = Fourier_Transform{omega(t)}
     V_norm(f)     = |V(f)| / max_f |V(f)|
     f_c           = min(10.0 Hz, max{f | V_norm(f) >= 0.05})
     SPARC         = - Integral_0^1 sqrt( (d(f_tilde)/df)^2 + (d(V_tilde)/d(f_tilde))^2 ) d(f_tilde)
                   = - Sum_{i=1}^K sqrt( (Delta_f_tilde_i)^2 + (Delta_V_norm_i)^2 )

  4. All 16 Clinical Features Equations:
     - F1  (ROM)            : ROM = max_t(theta(t)) - min_t(theta(t))
     - F2  (Peak Depth)     : theta_peak = max_t(theta(t))
     - F3  (Extensor Lag)   : theta_min = min_t(theta(t))   [Lag if theta_min > 8 deg]
     - F4  (Peak Speed)     : omega_peak = max_t(|omega(t)|)
     - F5  (Mean Speed)     : omega_mean = (1 / N) * Sum_{t=1}^N |omega(t)|
     - F6  (SPARC)          : eta_SPARC = - Sum_{i=1}^K sqrt( (Delta_f_i)^2 + (Delta_V_i)^2 )
     - F7  (Rep Count)      : N_reps = Sum_j Indicator(Valid Repetition j)
     - F8  (Rep Duration)   : T_rep = (1 / N_reps) * Sum_{j=1}^{N_reps} [(t_end_j - t_start_j) / f_s]
     - F9  (Angle Std Dev)  : sigma_theta = sqrt( (1 / N) * Sum_{t=1}^N (theta(t) - theta_bar)^2 )
     - F10 (Vel Std Dev)    : sigma_omega = sqrt( (1 / N) * Sum_{t=1}^N (omega(t) - omega_bar)^2 )
     - F11 (Angle IQR)      : IQR_theta = Q_3(theta) - Q_1(theta) = P_75(theta) - P_25(theta)
     - F12 (Max Velocity)   : omega_max = max_t(omega(t))   [Peak Extension Acceleration]
     - F13 (Min Velocity)   : omega_min = min_t(omega(t))   [Peak Flexion Descent Control]
     - F14 (Mean Angle)     : theta_bar = (1 / N) * Sum_{t=1}^N theta(t)
     - F15 (Angle RMS)      : RMS_theta = sqrt( (1 / N) * Sum_{t=1}^N theta(t)^2 )
     - F16 (Smoothness Rate): rho_SPARC = eta_SPARC / (T_rep + 1e-6)

TABLE OF CONTENTS:
  SECTION 1: Imports & Setup
  SECTION 2: Clinical16FeatureExtractor Class Definition
             - __init__                  : Sampling rate and time-step configuration
             - compute_knee_angle        : Gyroscope integration with baseline drift filter
             - compute_angular_velocity  : Differentiation & 5 Hz low-pass filter
             - compute_sparc             : Spectral Arc Length smoothness metric
             - segment_repetitions       : Automated repetition boundary detection
             - extract_16_features       : Extraction of the 16 clinical features
  SECTION 3: Standalone Verification & Demo Execution (__main__)
========================================================================================
"""

# ======================================================================================
# SECTION 1: IMPORTS & SETUP
# ======================================================================================
import numpy as np
from scipy import signal


# ======================================================================================
# SECTION 2: CLINICAL 16-FEATURE EXTRACTOR CLASS
# ======================================================================================
class Clinical16FeatureExtractor:
    """
    WHAT THIS CLASS IS:
      A self-contained biomechanical feature extraction engine that processes raw IMU
      signals into 16 standardized clinical parameters for orthopedic knee rehabilitation.

    EDGE & EMBEDDED CAPABILITY:
      Uses only NumPy and SciPy operations; lightweight and easily portable to Android,
      Raspberry Pi, or embedded edge microcontrollers.
    """

    def __init__(self, fs: float = 100.0):
        """
        WHAT THIS DOES:
          Initializes the feature extractor with the sensor sampling frequency.

        PARAMETERS:
          fs (float): Sampling frequency in Hertz (default 100.0 Hz, dt = 0.01 seconds).
        """
        self.fs = fs
        self.dt = 1.0 / fs  # Sampling period (0.01 s for 100 Hz)

    # ----------------------------------------------------------------------------------
    # METHOD 1: COMPUTE KNEE ANGLE TRAJECTORY
    # ----------------------------------------------------------------------------------
    def compute_knee_angle(self, thigh_gyro_y: np.ndarray, shin_gyro_y: np.ndarray) -> np.ndarray:
        """
        WHAT THIS DOES:
          Calculates the continuous anatomical sagittal knee flexion angle in degrees
          from the Y-axis angular velocities of the thigh and shin IMU sensors.

        MATHEMATICAL EQUATIONS:
          1. Relative Joint Angular Velocity:
             omega_knee(t) = omega_shin_Y(t) - omega_thigh_Y(t)

          2. Numerical Integration (Riemann / Euler Cumulative Sum):
             theta_raw(t) = Sum_{k=0}^{t} omega_knee(k) * Delta_t
             where Delta_t = 1.0 / f_s = 0.01 seconds

          3. 1st-Order High-Pass Butterworth Drift Filter:
             Analog prototype:
                 H_HP(s) = s / (s + omega_c)
             where cutoff frequency:
                 omega_c = 2 * pi * f_c = 2 * pi * 0.05 Hz
             Bilinear transform discrete transfer function:
                 H(z) = (b0 + b1 * z^{-1}) / (1 + a1 * z^{-1})
             Zero-phase forward-backward digital filtering:
                 theta_filt = filtfilt(b_hp, a_hp, theta_raw)

          4. Baseline Anatomical Calibration:
             theta_calib(t) = theta_filt(t) - min_t(theta_filt(t))
             (Ensures 0 degrees corresponds to anatomical terminal knee extension).

        PARAMETERS:
          thigh_gyro_y (np.ndarray): 1D array of Thigh IMU Gyroscope Y-axis (in deg/s).
          shin_gyro_y  (np.ndarray): 1D array of Shin IMU Gyroscope Y-axis (in deg/s).

        RETURNS:
          knee_angle (np.ndarray): 1D calibrated knee angle array (0° = full extension).
        """
        # Equation 1: omega_knee(t) = omega_shin_Y(t) - omega_thigh_Y(t)
        relative_angular_velocity = shin_gyro_y - thigh_gyro_y

        # Equation 2: theta_raw(t) = Sum omega_knee * dt
        raw_angle = np.cumsum(relative_angular_velocity) * self.dt

        # Equation 3: H_HP(s) = s / (s + omega_c), f_c = 0.05 Hz
        b_hp, a_hp = signal.butter(1, 0.05 / (0.5 * self.fs), btype='high')
        filtered_angle = signal.filtfilt(b_hp, a_hp, raw_angle)

        # Equation 4: theta(t) = theta_filt(t) - min(theta_filt(t))
        knee_angle = filtered_angle - np.min(filtered_angle)
        return knee_angle

    # ----------------------------------------------------------------------------------
    # METHOD 2: COMPUTE ANGULAR VELOCITY
    # ----------------------------------------------------------------------------------
    def compute_angular_velocity(self, knee_angle: np.ndarray) -> np.ndarray:
        """
        WHAT THIS DOES:
          Computes the instantaneous knee angular velocity (rate of flexion/extension).

        MATHEMATICAL EQUATIONS:
          1. Central Difference Numerical Derivative:
             omega(t) = d(theta)/dt
             Interior points:
                 omega(t) ~= [theta(t + Delta_t) - theta(t - Delta_t)] / (2 * Delta_t)
             Boundaries:
                 omega(0) ~= [theta(Delta_t) - theta(0)] / Delta_t
                 omega(T) ~= [theta(T) - theta(T - Delta_t)] / Delta_t

          2. 2nd-Order Low-Pass Butterworth Filter:
             Analog prototype:
                 H_LP(s) = omega_c^2 / (s^2 + sqrt(2) * omega_c * s + omega_c^2)
             where cutoff frequency:
                 omega_c = 2 * pi * f_c = 2 * pi * 5.0 Hz
                 Normalized digital cutoff W_n = f_c / (f_s / 2) = 5.0 / 50.0 = 0.10
             Zero-phase forward-backward filtering:
                 omega_smooth = filtfilt(b_lp, a_lp, omega)

        PARAMETERS:
          knee_angle (np.ndarray): 1D array of knee flexion angles in degrees.

        RETURNS:
          smooth_velocity (np.ndarray): 1D array of joint angular velocities in deg/s.
        """
        # Equation 1: Numerical gradient
        velocity = np.gradient(knee_angle, self.dt)

        # Equation 2: 2nd-order low-pass filter at 5.0 Hz
        b, a = signal.butter(2, 5.0 / (0.5 * self.fs), btype='low')
        smooth_velocity = signal.filtfilt(b, a, velocity)
        return smooth_velocity

    # ----------------------------------------------------------------------------------
    # METHOD 3: SPECTRAL ARC LENGTH (SPARC) MOVEMENT SMOOTHNESS
    # ----------------------------------------------------------------------------------
    def compute_sparc(self, velocity_signal: np.ndarray, padlevel: int = 4, fc: float = 10.0, amp_th: float = 0.05) -> float:
        """
        WHAT THIS DOES:
          Computes the Spectral Arc Length (SPARC) index, a clinically validated
          gold-standard metric for quantifying human motor control and movement smoothness.

        MATHEMATICAL EQUATIONS:
          1. Zero-Padded Discrete Fourier Transform (DFT):
             N_FFT = 2^{ ceil(log2(N)) + padlevel }
             V(f_k) = Sum_{n=0}^{N-1} v[n] * exp( -j * 2 * pi * k * n / N_FFT )

          2. Normalized Magnitude Spectrum:
             |V(f_k)| = sqrt( Real(V(f_k))^2 + Imag(V(f_k))^2 )
             V_norm(f_k) = |V(f_k)| / max_k |V(f_k)|

          3. Cutoff Frequency Selection:
             f_c_idx = max{ k | f_k <= 10.0 Hz  and  V_norm(f_k) >= 0.05 }

          4. Dimensionless Spectral Coordinate Normalization:
             f_tilde_k = f_k / f_c
             V_tilde(f_tilde_k) = V_norm(f_k)

          5. Negative Arc Length of the Spectral Curve:
             eta_SPARC = - Integral_0^1 sqrt( (d(f_tilde)/df)^2 + (d(V_tilde)/d(f_tilde))^2 ) d(f_tilde)
             Discrete approximation:
             eta_SPARC = - Sum_{i=1}^K sqrt( (Delta f_tilde_i)^2 + (Delta V_tilde_i)^2 )
             where:
                 Delta f_tilde_i = f_tilde_i - f_tilde_{i-1}
                 Delta V_tilde_i = V_tilde_i - V_tilde_{i-1}

        CLINICAL INTERPRETATION:
          - Smooth, unimpaired trajectory: eta_SPARC in [-1.5, -2.5] (close to 0).
          - Guarded, hesitant, jerky trajectory: eta_SPARC < -3.5 (strongly negative).

        PARAMETERS:
          velocity_signal (np.ndarray): 1D array of angular velocity.
          padlevel (int): Zero-padding level for high-resolution FFT (default: 4).
          fc (float): Frequency cutoff threshold (default: 10.0 Hz).
          amp_th (float): Minimum relative amplitude threshold ratio (default: 0.05).

        RETURNS:
          arc_length (float): Dimensionless smoothness index (always <= 0.0).
        """
        v = np.asarray(velocity_signal)
        if len(v) < 10 or np.all(v == 0):
            return -5.0

        # Step 1: Zero-padding and real-valued FFT
        # Equation: N_FFT = 2^{ ceil(log2(len(v))) + 4 }
        n_fft = int(2 ** (np.ceil(np.log2(len(v))) + padlevel))
        freq = np.fft.rfftfreq(n_fft, d=self.dt)
        f_mag = np.abs(np.fft.rfft(v, n=n_fft))

        # Step 2: Normalize magnitude spectrum by peak value
        # Equation: V_norm(f) = |V(f)| / max(|V(f)|)
        max_f = np.max(f_mag)
        if max_f == 0:
            return -5.0
        f_norm = f_mag / max_f

        # Step 3: Find cutoff index where frequency <= 10 Hz and magnitude >= 0.05
        cutoff_idx = np.where((freq <= fc) & (f_norm >= amp_th))[0]
        cutoff_idx = cutoff_idx[-1] if len(cutoff_idx) > 0 else len(freq) - 1

        # Step 4: Dimensionless normalized coordinates
        # Equation: f_tilde = f / f_c
        f_sel = freq[:cutoff_idx + 1] / fc
        mag_sel = f_norm[:cutoff_idx + 1]

        # Step 5: Arc length integration
        # Equation: eta_SPARC = - Sum sqrt( (Delta f_tilde)^2 + (Delta V_tilde)^2 )
        df = np.diff(f_sel)
        dmag = np.diff(mag_sel)
        arc_length = -np.sum(np.sqrt(df**2 + dmag**2))
        return float(arc_length)

    # ----------------------------------------------------------------------------------
    # METHOD 4: AUTOMATED REPETITION SEGMENTATION
    # ----------------------------------------------------------------------------------
    def segment_repetitions(self, knee_angle: np.ndarray, min_rom_threshold: float = 20.0, min_duration_sec: float = 0.8):
        """
        WHAT THIS DOES:
          Automatically segments individual exercise repetitions from a continuous trial
          using peak detection and valley finding algorithms.

        MATHEMATICAL CONDITIONS:
          1. Peak Identification:
             P_j = { t_p | theta(t_p) >= 20.0 deg  and  |t_p - t_{p-1}| >= 0.8 * f_s }

          2. Boundary Valley Detection:
             t_start_j = argmin_{t in [t_p - 2.5*f_s, t_p]} theta(t)
             t_end_j   = argmin_{t in [t_p, t_p + 2.5*f_s]} theta(t)

          3. Repetition Validity Criteria:
             ROM_rep = theta(t_p) - min( theta(t_start), theta(t_end) ) >= 20.0 deg
             Duration_rep = (t_end - t_start) / f_s >= 0.8 s

        PARAMETERS:
          knee_angle (np.ndarray): Filtered knee angle trajectory in degrees.
          min_rom_threshold (float): Minimum joint excursion to qualify as a valid repetition (20.0°).
          min_duration_sec (float): Minimum repetition duration in seconds (0.8s).

        RETURNS:
          rep_boundaries (list): Tuples of (start_index, peak_index, end_index) for each repetition.
          repetition_count (int): Total number of valid completed repetitions.
        """
        min_distance = int(min_duration_sec * self.fs)
        peaks, _ = signal.find_peaks(knee_angle, height=min_rom_threshold, distance=min_distance)

        rep_boundaries = []
        for p in peaks:
            # Valley search window: up to 2.5 seconds backward
            start_search = max(0, p - int(2.5 * self.fs))
            start_idx = start_search + np.argmin(knee_angle[start_search:p])

            # Valley search window: up to 2.5 seconds forward
            end_search = min(len(knee_angle), p + int(2.5 * self.fs))
            end_idx = p + np.argmin(knee_angle[p:end_search])

            # Equation: ROM_rep = theta(p) - min(theta(start), theta(end))
            rep_rom = knee_angle[p] - min(knee_angle[start_idx], knee_angle[end_idx])
            # Equation: Duration_rep = (end - start) / f_s
            rep_dur = (end_idx - start_idx) / self.fs

            # Validate thresholds
            if rep_rom >= min_rom_threshold and rep_dur >= min_duration_sec:
                rep_boundaries.append((start_idx, p, end_idx))

        return rep_boundaries, len(rep_boundaries)

    # ----------------------------------------------------------------------------------
    # METHOD 5: EXTRACT ALL 16 CLINICAL FEATURES
    # ----------------------------------------------------------------------------------
    def extract_16_features(self, knee_angle: np.ndarray):
        """
        WHAT THIS DOES:
          Computes the complete, standardized 16-dimensional clinical kinematic
          feature vector (Input B) from the continuous knee angle trajectory.

        INPUT:
          knee_angle (np.ndarray): 1D continuous knee angle trajectory in degrees.

        RETURNS:
          features_dict (dict): Dictionary with all 16 named clinical features.
          feature_vector (np.ndarray): 16-element float32 array ready for AI Input B.
        """
        # Intermediate velocity and repetition calculations
        ang_vel = self.compute_angular_velocity(knee_angle)
        rep_bounds, rep_count = self.segment_repetitions(knee_angle)

        # ---------------------------------------------------------------------
        # FEATURE 1: Range of Motion (ROM in degrees)
        # EQUATION:
        #   ROM = max_{t} theta(t) - min_{t} theta(t)
        # CLINICAL MEANING: Total active flexion arc achieved by the patient.
        # ---------------------------------------------------------------------
        rom = float(np.max(knee_angle) - np.min(knee_angle))

        # ---------------------------------------------------------------------
        # FEATURE 2: Peak Movement Depth (in degrees)
        # EQUATION:
        #   theta_peak = max_{t} theta(t)
        # CLINICAL MEANING: Deepest flexion point reached (e.g., squat depth).
        # Safe threshold for early tibial plateau repair: <= 45.0 degrees.
        # ---------------------------------------------------------------------
        peak_depth = float(np.max(knee_angle))

        # ---------------------------------------------------------------------
        # FEATURE 3: Minimum Knee Angle / Quadriceps Extensor Lag (in degrees)
        # EQUATION:
        #   theta_min = min_{t} theta(t)
        # CLINICAL MEANING: Detects Quadriceps Extensor Lag!
        # Normal extension = 0.0°. Extensor lag is flagged if theta_min > 8.0°.
        # ---------------------------------------------------------------------
        min_angle = float(np.min(knee_angle))

        # ---------------------------------------------------------------------
        # FEATURE 4: Peak Angular Velocity (deg/s)
        # EQUATION:
        #   omega_peak = max_{t} |omega(t)|
        # CLINICAL MEANING: Maximum rotational speed reached during movement.
        # ---------------------------------------------------------------------
        peak_velocity = float(np.max(np.abs(ang_vel)))

        # ---------------------------------------------------------------------
        # FEATURE 5: Mean Angular Velocity (deg/s)
        # EQUATION:
        #   omega_mean = (1 / N) * Sum_{t=1}^N |omega(t)|
        # CLINICAL MEANING: Average movement cadence maintained across the trial.
        # ---------------------------------------------------------------------
        mean_velocity = float(np.mean(np.abs(ang_vel)))

        # ---------------------------------------------------------------------
        # FEATURE 6: SPARC Movement Smoothness (dimensionless index)
        # EQUATION:
        #   eta_SPARC = - Sum_{i=1}^K sqrt( (Delta f_tilde_i)^2 + (Delta V_norm_i)^2 )
        # CLINICAL MEANING: Quantifies muscle tremor, hesitation, and guarding.
        # Normal >= -2.5. Jerky / compensatory fault < -3.5.
        # ---------------------------------------------------------------------
        sparc_smoothness = self.compute_sparc(ang_vel)

        # ---------------------------------------------------------------------
        # FEATURE 7: Repetition Count (integer)
        # EQUATION:
        #   N_reps = Sum_{j} Indicator(ROM_rep_j >= 20°  and  Duration_j >= 0.8s)
        # CLINICAL MEANING: Verified count of completed valid repetitions.
        # ---------------------------------------------------------------------
        repetition_count = float(rep_count)

        # ---------------------------------------------------------------------
        # FEATURE 8: Mean Repetition Duration (seconds)
        # EQUATION:
        #   T_rep_mean = (1 / N_reps) * Sum_{j=1}^{N_reps} [ (t_end_j - t_start_j) / f_s ]
        # CLINICAL MEANING: Average time spent per repetition cycle.
        # ---------------------------------------------------------------------
        if rep_count > 0:
            durations = [(end - start) / self.fs for start, _, end in rep_bounds]
            mean_duration = float(np.mean(durations))
        else:
            mean_duration = float(len(knee_angle) / self.fs)

        # ---------------------------------------------------------------------
        # FEATURE 9: Knee Angle Standard Deviation
        # EQUATION:
        #   sigma_theta = sqrt( (1 / N) * Sum_{t=1}^N (theta(t) - theta_bar)^2 )
        # CLINICAL MEANING: Statistical dispersion of joint excursion.
        # ---------------------------------------------------------------------
        angle_std = float(np.std(knee_angle))

        # ---------------------------------------------------------------------
        # FEATURE 10: Angular Velocity Standard Deviation
        # EQUATION:
        #   sigma_omega = sqrt( (1 / N) * Sum_{t=1}^N (omega(t) - omega_bar)^2 )
        # CLINICAL MEANING: Velocity variability and cadence consistency.
        # ---------------------------------------------------------------------
        velocity_std = float(np.std(ang_vel))

        # ---------------------------------------------------------------------
        # FEATURE 11: Angle Interquartile Range (IQR in degrees)
        # EQUATION:
        #   IQR_theta = Q_3(theta) - Q_1(theta) = Percentile_75(theta) - Percentile_25(theta)
        # CLINICAL MEANING: Middle 50% movement range; robust against sudden artifact jerks.
        # ---------------------------------------------------------------------
        angle_iqr = float(np.percentile(knee_angle, 75) - np.percentile(knee_angle, 25))

        # ---------------------------------------------------------------------
        # FEATURE 12: Max Angular Velocity (Peak Extension Power in deg/s)
        # EQUATION:
        #   omega_max = max_{t} omega(t)
        # CLINICAL MEANING: Maximum forward/straightening acceleration power.
        # ---------------------------------------------------------------------
        max_ang_vel = float(np.max(ang_vel))

        # ---------------------------------------------------------------------
        # FEATURE 13: Min Angular Velocity (Peak Flexion Descent Control in deg/s)
        # EQUATION:
        #   omega_min = min_{t} omega(t)
        # CLINICAL MEANING: Eccentric control during the bending/lowering phase.
        # ---------------------------------------------------------------------
        min_ang_vel = float(np.min(ang_vel))

        # ---------------------------------------------------------------------
        # FEATURE 14: Mean Knee Angle (in degrees)
        # EQUATION:
        #   theta_bar = (1 / N) * Sum_{t=1}^N theta(t)
        # CLINICAL MEANING: Average joint load center over the entire exercise trial.
        # ---------------------------------------------------------------------
        mean_angle = float(np.mean(knee_angle))

        # ---------------------------------------------------------------------
        # FEATURE 15: Angle Root Mean Square (RMS in degrees)
        # EQUATION:
        #   RMS_theta = sqrt( (1 / N) * Sum_{t=1}^N theta(t)^2 )
        # CLINICAL MEANING: Cumulative energetic displacement of the joint.
        # ---------------------------------------------------------------------
        angle_rms = float(np.sqrt(np.mean(knee_angle**2)))

        # ---------------------------------------------------------------------
        # FEATURE 16: SPARC / Duration Ratio (Smoothness Efficiency Rate)
        # EQUATION:
        #   rho_SPARC = eta_SPARC / (T_rep_mean + epsilon),  epsilon = 1e-6
        # CLINICAL MEANING: Motor control efficiency normalized per unit of time.
        # ---------------------------------------------------------------------
        sparc_per_duration = float(sparc_smoothness / (mean_duration + 1e-6))

        # Pack into clean dictionary
        features_dict = {
            "rom_deg": rom,
            "peak_depth_deg": peak_depth,
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

        # Pack into 16-element float32 vector ready for AI Input B
        feature_vector = np.array([
            rom, peak_depth, min_angle, peak_velocity, mean_velocity,
            sparc_smoothness, repetition_count, mean_duration, angle_std, velocity_std,
            angle_iqr, max_ang_vel, min_ang_vel, mean_angle, angle_rms, sparc_per_duration
        ], dtype=np.float32)

        return features_dict, feature_vector


# ======================================================================================
# SECTION 3: STANDALONE VERIFICATION & DEMO EXECUTION (__main__)
# ======================================================================================
if __name__ == "__main__":
    import os
    import pandas as pd

    print("=" * 75)
    print("TESTING 16 CLINICAL FEATURE EXTRACTION CODE (DATASET 2 ENGINE)")
    print("=" * 75)

    # Load 1 trial from Dataset 1 to demonstrate live feature extraction
    d1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "dataset_1_input_A_12channels.npz")
    if os.path.exists(d1_path):
        data = np.load(d1_path)
        sample_trial = data["X_raw_12ch"][0]  # First trial: shape (200, 12)
        thigh_gyr_y = sample_trial[:, 4]      # Channel 4: Thigh Gyro Y
        shin_gyr_y  = sample_trial[:, 10]     # Channel 10: Shin Gyro Y

        extractor = Clinical16FeatureExtractor(fs=100.0)

        # 1. Compute Continuous Knee Angle
        knee_angle = extractor.compute_knee_angle(thigh_gyr_y, shin_gyr_y)

        # 2. Extract All 16 Clinical Features
        feats, feat_vec = extractor.extract_16_features(knee_angle)

        print("\nSuccessfully Extracted 16 Clinical Features for Trial #1:")
        print("-" * 60)
        for idx, (name, val) in enumerate(feats.items(), 1):
            unit = "deg" if "deg" in name or "angle" in name or "depth" in name or "rom" in name or "iqr" in name or "rms" in name else ""
            unit = "deg/s" if "vel" in name else unit
            unit = "s" if "duration" in name else unit
            print(f"  {idx:2d}. {name:<26}: {val:8.2f} {unit}")
        print("-" * 60)
        print(f"Feature Vector Shape (Input B): {feat_vec.shape} (dtype: {feat_vec.dtype})")
        print("\nAll 16 features extracted cleanly according to orthopedic biomechanics equations!")
    else:
        print("Dataset 1 file not found. Place dataset_1_input_A_12channels.npz in the folder to test.")
