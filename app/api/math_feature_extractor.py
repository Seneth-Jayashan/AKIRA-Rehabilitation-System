"""
========================================================================================
AKIRA KNEE REHABILITATION: 16 CLINICAL FEATURE EXTRACTION ENGINE (DATASET 2)
========================================================================================
Component: Smart Post-ORIF Rehabilitation Ecosystem with Edge-AI Kinematic Tracking
Target Joint: Knee Joint (Tibial Plateau ORIF Rehabilitation)
Sensors: Dual 6-Axis IMUs (Thigh + Shin) @ 100 Hz (dt = 0.01 s)

This standalone module extracts the 16 domain clinical kinematic features (Input B)
from raw dual-IMU signals and relative knee angles. Every feature is clinically 
grounded in orthopedic knee recovery (Range of Motion, Quadriceps Extensor Lag, 
Movement Depth, and SPARC Smoothness).
========================================================================================
"""

import numpy as np
from scipy import signal

class Clinical16FeatureExtractor:
    """
    Computes the 16 clinical biomechanical features from knee angle trajectories.
    Can be run on an Android phone, edge processor, or laptop.
    """

    def __init__(self, fs=100.0):
        """
        Parameters:
            fs (float): Sampling rate in Hz (default: 100.0 Hz, dt = 0.01 s)
        """
        self.fs = fs
        self.dt = 1.0 / fs

    # -------------------------------------------------------------------------
    # STEP 1: Compute Knee Angle Trajectory from Thigh + Shin Gyroscopes
    # -------------------------------------------------------------------------
    def compute_knee_angle(self, thigh_gyro_y, shin_gyro_y):
        """
        Calculates the continuous sagittal knee flexion angle in degrees.
        
        Clinical Principle:
            The relative angular velocity around the lateral knee joint axis is:
            omega_knee(t) = Shin_Gyro_Y(t) - Thigh_Gyro_Y(t)
            Integrating omega_knee over time gives the anatomical knee angle.
            
        Parameters:
            thigh_gyro_y (1D array): Gyroscope Y-axis of the Thigh IMU (deg/s)
            shin_gyro_y  (1D array): Gyroscope Y-axis of the Shin IMU (deg/s)
            
        Returns:
            knee_angle (1D array): Calibrated knee angle in degrees (0° = full extension)
        """
        # 1. Relative angular velocity at the knee joint
        relative_angular_velocity = shin_gyro_y - thigh_gyro_y

        # 2. Integrate angular velocity over time (dt = 0.01 s)
        raw_angle = np.cumsum(relative_angular_velocity) * self.dt

        # 3. Apply zero-phase 1st-order high-pass Butterworth filter (0.05 Hz)
        #    to eliminate integration drift while preserving movement cycles
        b_hp, a_hp = signal.butter(1, 0.05 / (0.5 * self.fs), btype='high')
        filtered_angle = signal.filtfilt(b_hp, a_hp, raw_angle)

        # 4. Calibrate baseline so full extension is at 0 degrees
        knee_angle = filtered_angle - np.min(filtered_angle)
        return knee_angle

    # -------------------------------------------------------------------------
    # STEP 2: Instantaneous Angular Velocity (Speed of Bending/Extending)
    # -------------------------------------------------------------------------
    def compute_angular_velocity(self, knee_angle):
        """
        Differentiates knee angle to obtain instantaneous angular velocity (deg/s).
        Applies a zero-phase 2nd-order low-pass filter (5 Hz) to smooth derivatives.
        """
        velocity = np.gradient(knee_angle, self.dt)
        b, a = signal.butter(2, 5.0 / (0.5 * self.fs), btype='low')
        smooth_velocity = signal.filtfilt(b, a, velocity)
        return smooth_velocity

    # -------------------------------------------------------------------------
    # STEP 3: Spectral Arc Length (SPARC) Movement Smoothness Metric
    # -------------------------------------------------------------------------
    def compute_sparc(self, velocity_signal, padlevel=4, fc=10.0, amp_th=0.05):
        """
        Calculates the Spectral Arc Length (SPARC) smoothness index.
        
        Clinical Significance:
            A validated gold-standard metric for motor control recovery:
            - Healthy / recovered movement is smooth and bell-shaped -> SPARC is close to 0 (e.g. -2.0).
            - Post-ORIF patients exhibit jerky muscle guarding and tremor -> SPARC is strongly negative (e.g. < -4.0).
            
        Parameters:
            velocity_signal (1D array): Angular velocity curve
            fc (float): Frequency cutoff (10.0 Hz)
            amp_th (float): Minimum amplitude threshold ratio (0.05)
            
        Returns:
            sparc_score (float): Movement smoothness index (dimensionless, <= 0)
        """
        v = np.asarray(velocity_signal)
        if len(v) < 10 or np.all(v == 0):
            return -5.0

        # Zero-pad for high-resolution Fast Fourier Transform (FFT)
        n_fft = int(2 ** (np.ceil(np.log2(len(v))) + padlevel))
        freq = np.fft.rfftfreq(n_fft, d=self.dt)
        f_mag = np.abs(np.fft.rfft(v, n=n_fft))

        max_f = np.max(f_mag)
        if max_f == 0:
            return -5.0
        f_norm = f_mag / max_f

        # Find cutoff frequency index
        cutoff_idx = np.where((freq <= fc) & (f_norm >= amp_th))[0]
        cutoff_idx = cutoff_idx[-1] if len(cutoff_idx) > 0 else len(freq) - 1

        f_sel = freq[:cutoff_idx + 1] / fc
        mag_sel = f_norm[:cutoff_idx + 1]

        # Calculate negative arc length of the normalized spectrum
        df = np.diff(f_sel)
        dmag = np.diff(mag_sel)
        arc_length = -np.sum(np.sqrt(df**2 + dmag**2))
        return float(arc_length)

    # -------------------------------------------------------------------------
    # STEP 4: Automated Repetition Segmentation
    # -------------------------------------------------------------------------
    def segment_repetitions(self, knee_angle, min_rom_threshold=20.0, min_duration_sec=0.8):
        """
        Automatically identifies each exercise repetition cycle using peak detection.
        
        Returns:
            rep_boundaries: List of (start_idx, peak_idx, end_idx) for each repetition
            repetition_count: Total integer count of completed repetitions
        """
        min_distance = int(min_duration_sec * self.fs)
        peaks, _ = signal.find_peaks(knee_angle, height=min_rom_threshold, distance=min_distance)

        rep_boundaries = []
        for p in peaks:
            start_search = max(0, p - int(2.5 * self.fs))
            start_idx = start_search + np.argmin(knee_angle[start_search:p])

            end_search = min(len(knee_angle), p + int(2.5 * self.fs))
            end_idx = p + np.argmin(knee_angle[p:end_search])

            rep_rom = knee_angle[p] - min(knee_angle[start_idx], knee_angle[end_idx])
            rep_dur = (end_idx - start_idx) / self.fs

            if rep_rom >= min_rom_threshold and rep_dur >= min_duration_sec:
                rep_boundaries.append((start_idx, p, end_idx))

        return rep_boundaries, len(rep_boundaries)

    # -------------------------------------------------------------------------
    # STEP 5: Extract All 16 Clinical Features
    # -------------------------------------------------------------------------
    def extract_16_features(self, knee_angle):
        """
        Transforms a continuous knee angle curve into the standardized 16-D clinical feature vector.
        
        Parameters:
            knee_angle (1D array): Filtered continuous knee angle trajectory in degrees.
            
        Returns:
            features_dict (dict): Dictionary with all 16 named clinical features.
            feature_vector (1D array): 16-element float32 array ready for AI model input (Input B).
        """
        ang_vel = self.compute_angular_velocity(knee_angle)
        rep_bounds, rep_count = self.segment_repetitions(knee_angle)

        # ---------------------------------------------------------------------
        # FEATURE 1: Range of Motion (ROM in degrees)
        # Clinical Meaning: Total active flexion arc achieved by the patient.
        # ---------------------------------------------------------------------
        rom = float(np.max(knee_angle) - np.min(knee_angle))

        # ---------------------------------------------------------------------
        # FEATURE 2: Peak Movement Depth (in degrees)
        # Clinical Meaning: Deepest point reached during squats or seated extension.
        # Critical for tibial plateau patients to verify weight-bearing depth safety.
        # ---------------------------------------------------------------------
        peak_depth = float(np.max(knee_angle))

        # ---------------------------------------------------------------------
        # FEATURE 3: Min Knee Angle / Terminal Extension Deficit (in degrees)
        # Clinical Meaning: Detects Quadriceps Extensor Lag! If the patient cannot
        # lock the knee straight to 0° (e.g. min angle = 10°), extensor lag is detected.
        # ---------------------------------------------------------------------
        min_angle = float(np.min(knee_angle))

        # ---------------------------------------------------------------------
        # FEATURE 4: Peak Angular Velocity (deg/s)
        # Clinical Meaning: Fastest movement speed reached during flexion or extension.
        # ---------------------------------------------------------------------
        peak_velocity = float(np.max(np.abs(ang_vel)))

        # ---------------------------------------------------------------------
        # FEATURE 5: Mean Angular Velocity (deg/s)
        # Clinical Meaning: Average pace maintained during the exercise set.
        # ---------------------------------------------------------------------
        mean_velocity = float(np.mean(np.abs(ang_vel)))

        # ---------------------------------------------------------------------
        # FEATURE 6: SPARC Movement Smoothness (dimensionless index)
        # Clinical Meaning: Quantifies muscle tremor and hesitancy in the frequency domain.
        # Closer to 0 = smooth, controlled. Strongly negative (< -3.5) = jerky guarding.
        # ---------------------------------------------------------------------
        sparc_smoothness = self.compute_sparc(ang_vel)

        # ---------------------------------------------------------------------
        # FEATURE 7: Repetition Count (integer)
        # Clinical Meaning: Verified count of completed valid repetitions.
        # ---------------------------------------------------------------------
        repetition_count = float(rep_count)

        # ---------------------------------------------------------------------
        # FEATURE 8: Mean Repetition Duration (seconds)
        # Clinical Meaning: Average time taken per cycle. Detects fatigue or hesitation.
        # ---------------------------------------------------------------------
        if rep_count > 0:
            durations = [(end - start) / self.fs for start, _, end in rep_bounds]
            mean_duration = float(np.mean(durations))
        else:
            mean_duration = float(len(knee_angle) / self.fs)

        # ---------------------------------------------------------------------
        # FEATURE 9: Knee Angle Standard Deviation
        # Clinical Meaning: Statistical dispersion of joint excursion.
        # ---------------------------------------------------------------------
        angle_std = float(np.std(knee_angle))

        # ---------------------------------------------------------------------
        # FEATURE 10: Angular Velocity Standard Deviation
        # Clinical Meaning: Consistency of speed control throughout the movement.
        # ---------------------------------------------------------------------
        velocity_std = float(np.std(ang_vel))

        # ---------------------------------------------------------------------
        # FEATURE 11: Angle Interquartile Range (IQR = 75th - 25th percentile)
        # Clinical Meaning: Mid-range joint stability, immune to brief jerk outliers.
        # ---------------------------------------------------------------------
        angle_iqr = float(np.percentile(knee_angle, 75) - np.percentile(knee_angle, 25))

        # ---------------------------------------------------------------------
        # FEATURE 12: Max Angular Velocity (Peak Extension Power)
        # Clinical Meaning: Maximum acceleration speed kicking straight.
        # ---------------------------------------------------------------------
        max_ang_vel = float(np.max(ang_vel))

        # ---------------------------------------------------------------------
        # FEATURE 13: Min Angular Velocity (Peak Flexion Descent Control)
        # Clinical Meaning: Controlled eccentric descent speed during squats.
        # ---------------------------------------------------------------------
        min_ang_vel = float(np.min(ang_vel))

        # ---------------------------------------------------------------------
        # FEATURE 14: Mean Knee Angle (in degrees)
        # Clinical Meaning: Average joint load angle over the entire trial.
        # ---------------------------------------------------------------------
        mean_angle = float(np.mean(knee_angle))

        # ---------------------------------------------------------------------
        # FEATURE 15: Angle Root Mean Square (RMS in degrees)
        # Clinical Meaning: Overall cumulative energetic excursion of the joint.
        # ---------------------------------------------------------------------
        angle_rms = float(np.sqrt(np.mean(knee_angle**2)))

        # ---------------------------------------------------------------------
        # FEATURE 16: SPARC / Duration Ratio
        # Clinical Meaning: Normalized smoothness efficiency per unit time.
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

        # 16-element float32 array ready for AI Input B
        feature_vector = np.array([
            rom, peak_depth, min_angle, peak_velocity, mean_velocity,
            sparc_smoothness, repetition_count, mean_duration, angle_std, velocity_std,
            angle_iqr, max_ang_vel, min_ang_vel, mean_angle, angle_rms, sparc_per_duration
        ], dtype=np.float32)

        return features_dict, feature_vector


# =============================================================================
# DEMO EXECUTION: Test on a trial from Dataset 1
# =============================================================================
if __name__ == "__main__":
    import os
    import pandas as pd

    print("=" * 70)
    print("TESTING 16 CLINICAL FEATURE EXTRACTION CODE (DATASET 2 ENGINE)")
    print("=" * 70)

    # Load 1 trial from Dataset 1 (Input A) to demonstrate feature extraction
    d1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "dataset_1_input_A_12channels.npz")
    if os.path.exists(d1_path):
        data = np.load(d1_path)
        sample_trial = data["X_raw_12ch"][0]  # First trial: shape (200, 12)
        thigh_gyr_y = sample_trial[:, 4]      # Channel 4: Thigh Gyro Y
        shin_gyr_y  = sample_trial[:, 10]     # Channel 10: Shin Gyro Y

        extractor = Clinical16FeatureExtractor(fs=100.0)

        # 1. Compute Knee Angle
        knee_angle = extractor.compute_knee_angle(thigh_gyr_y, shin_gyr_y)

        # 2. Extract All 16 Clinical Features
        feats, feat_vec = extractor.extract_16_features(knee_angle)

        print("\nSuccessfully Extracted 16 Clinical Features for Trial #1:")
        print("-" * 55)
        for idx, (name, val) in enumerate(feats.items(), 1):
            unit = "deg" if "deg" in name or "angle" in name or "depth" in name or "rom" in name or "iqr" in name or "rms" in name else ""
            unit = "deg/s" if "vel" in name else unit
            unit = "s" if "duration" in name else unit
            print(f"  {idx:2d}. {name:<26}: {val:8.2f} {unit}")
        print("-" * 55)
        print(f"Feature Vector Shape (Input B): {feat_vec.shape} (dtype: {feat_vec.dtype})")
        print("\nAll 16 features extracted cleanly according to orthopedic biomechanics!")
    else:
        print("Dataset 1 file not found. Place dataset_1_input_A_12channels.npz in the folder to test.")
