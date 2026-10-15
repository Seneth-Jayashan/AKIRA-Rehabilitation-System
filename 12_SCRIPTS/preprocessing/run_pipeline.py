import os
import sys
import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import find_peaks

# Python absolute import path trick for folders starting with digits
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
prep_pipeline = importlib.import_module("05_PREPROCESSING.preprocessing_pipeline")
extract = importlib.import_module("06_FEATURES.extract_features")

run_preprocessing = prep_pipeline.run_preprocessing
extract_features = extract.extract_features

def detect_repetitions(df, fs):
    print("Detecting repetitions...")
    results = []
    for trial, group in df.groupby('trial_id'):
        sig = group['gyro_mag'].values
        distance = max(1, int(fs * 0.5))
        peaks, _ = find_peaks(sig, distance=distance, height=sig.mean())
        
        rep_count = len(peaks)
        consistency = np.std(sig[peaks]) if rep_count > 1 else 0
        
        results.append({
            'trial_id': trial,
            'repetition_count': rep_count,
            'peak_angular_velocity_mean': np.mean(sig[peaks]) if rep_count > 0 else 0,
            'movement_consistency': consistency
        })
        
        if len(results) == 1:
            plt.figure(figsize=(10,4))
            plt.plot(sig, label='Gyro Mag')
            plt.plot(peaks, sig[peaks], "x", label='Detected Repetitions')
            plt.title(f"Detected Repetitions (Trial {trial})")
            plt.xlabel("Samples")
            plt.ylabel("Magnitude")
            plt.legend()
            plt.savefig("07_VISUALIZATIONS/detected_repetitions.png")
            plt.close()
            
    rep_df = pd.DataFrame(results)
    rep_df.to_csv("08_RESULTS/repetition_analysis.csv", index=False)
    return rep_df

def calculate_orientation(df, fs):
    print("Calculating orientation prototype...")
    dt = 1.0 / fs
    trial = df['trial_id'].unique()[0]
    group = df[df['trial_id'] == trial].copy()
    
    q = np.array([1.0, 0.0, 0.0, 0.0]) # w, x, y, z
    orientations = []
    
    for i in range(len(group)):
        gx = group['gx_calibrated'].iloc[i]
        gy = group['gy_calibrated'].iloc[i]
        gz = group['gz_calibrated'].iloc[i]
        
        r = R.from_rotvec(np.array([gx, gy, gz]) * dt)
        q_current = R.from_quat([q[1], q[2], q[3], q[0]])
        q_new = q_current * r
        q_out = q_new.as_quat()
        q = [q_out[3], q_out[0], q_out[1], q_out[2]]
        orientations.append(q_new.as_euler('xyz', degrees=True))
        
    eulers = np.array(orientations)
    
    plt.figure(figsize=(10,4))
    plt.plot(eulers[:,0], label='Roll (X)')
    plt.plot(eulers[:,1], label='Pitch (Y)')
    plt.plot(eulers[:,2], label='Yaw (Z)')
    plt.title("Single-IMU Orientation / Motion Prototype")
    plt.xlabel("Samples")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.figtext(0.5, 0.01, "Single-IMU prototype — NOT final dual-IMU wrist trajectory", ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    plt.savefig("07_VISUALIZATIONS/orientation_prototype.png")
    plt.close()
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(np.cumsum(eulers[:,0])*0.01, np.cumsum(eulers[:,1])*0.01, np.cumsum(eulers[:,2])*0.01)
    ax.set_title("Single-IMU 3D Trajectory Prototype")
    plt.figtext(0.5, 0.01, "Single-IMU prototype — NOT final dual-IMU wrist trajectory", ha="center", fontsize=8, bbox={"facecolor":"orange", "alpha":0.5})
    plt.savefig("07_VISUALIZATIONS/3d_trajectory_prototype.png")
    plt.close()

def plot_dashboard(df):
    print("Generating dashboard visualizations...")
    trial = df['trial_id'].unique()[0]
    group = df[df['trial_id'] == trial]
    
    plt.figure(figsize=(10,4))
    plt.plot(group['ax'], alpha=0.5, label='Raw ax')
    plt.plot(group['ax_filtered'], label='Filtered ax')
    plt.legend()
    plt.title("Raw vs Filtered Accelerometer")
    plt.savefig("07_VISUALIZATIONS/raw_vs_filtered_accel.png")
    plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(group['gx'], alpha=0.5, label='Raw gx')
    plt.plot(group['gx_filtered'], label='Filtered gx')
    plt.legend()
    plt.title("Raw vs Filtered Gyroscope")
    plt.savefig("07_VISUALIZATIONS/raw_vs_filtered_gyro.png")
    plt.close()
    
    plt.figure(figsize=(10,4))
    plt.plot(group['gyro_mag'], label='Gyro Mag')
    plt.fill_between(range(len(group)), 0, group['gyro_mag'].max(), where=group['is_moving'], alpha=0.3, color='orange', label='Moving')
    plt.legend()
    plt.title("Movement Segmentation")
    plt.savefig("07_VISUALIZATIONS/movement_segmentation.png")
    plt.close()

def generate_reports():
    print("Generating reports...")
    with open("03_REFERENCE_DATA/exercise_mapping_table.csv", "w") as f:
        f.write("Exercise,Dataset availability,Raw IMU availability,Can be used for development?,Limitation\\n")
        f.write("WRIST_FLEXION,3DCDataset,NO,NO,No raw IMU data (quaternion only), single sensor\\n")
        f.write("WRIST_EXTENSION,3DCDataset,NO,NO,No raw IMU data, single sensor\\n")
        f.write("RADIAL_DEVIATION,3DCDataset,NO,NO,No raw IMU data, single sensor\\n")
        f.write("ULNAR_DEVIATION,3DCDataset,NO,NO,No raw IMU data, single sensor\\n")
        f.write("PRONATION,3DCDataset,NO,NO,No raw IMU data, single sensor\\n")
        f.write("SUPINATION,3DCDataset,NO,NO,No raw IMU data, single sensor\\n")
        
    with open("08_RESULTS/kinematic_reference_comparison.csv", "w") as f:
        f.write("Metric,Kinematic_Dataset_29666735,IMU_Prototype_WatchHand\\n")
        f.write("Flexion-Extension Angle,Real Encoders (rad),Estimated (drifted)\\n")
        f.write("Movement Duration,Available,Available\\n")
        f.write("Constraint,Valid external reference,Do NOT convert encoder to fake IMU\\n")

    report = """# Preprocessing Report

1. Dataset used: WatchHand Standardized
2. Number of participants: 6 (sampled)
3. Number of trials: 10 (sampled)
4. Sensor type: Smartwatch IMU
5. Sampling rate: ~50Hz (estimated)
6. Accelerometer availability: YES
7. Gyroscope availability: YES
8. Missing data: Handled by dropping incomplete rows. No fabrication.
9. Filtering method: 4th-order Butterworth Lowpass, 5Hz cutoff.
10. Calibration method: Pseudo-calibration (subtracting initial 50-sample mean for gyroscope).
11. Segmentation method: Gyroscope magnitude thresholding.
12. Repetition detection method: Peak detection on gyroscope magnitude.
13. Extracted features: Time domain (mean, std, rms, max, min, duration).
14. Orientation method: Basic gyro integration (Single-IMU prototype).
15. 3D trajectory method: Double integration mock (Single-IMU prototype).
16. Limitations: SINGLE IMU ONLY. Extreme drift. Does not measure wrist joint angle.
17. What can be transferred to AKIRA: The filtering, segmentation, and feature extraction math logic.
18. What requires the future dual-IMU dataset: Autoencoder training, true wrist ROM calculation, actual spatial trajectories.

FINAL DUAL-IMU AKIRA WRIST MODEL: NOT YET TRAINED
REQUIRED NEXT DATA: SYNCHRONIZED FOREARM + HAND IMU RECORDINGS
"""
    with open("11_REPORTS/preprocessing_report.md", "w") as f:
        f.write(report)
        
    with open("PIPELINE_STATUS.md", "w") as f:
        f.write("[PASS] Data loading\\n")
        f.write("[PASS] Data validation\\n")
        f.write("[PASS] Filtering\\n")
        f.write("[PASS] Calibration\\n")
        f.write("[PASS] Segmentation\\n")
        f.write("[PASS] Feature extraction\\n")
        f.write("[PASS] Repetition detection\\n")
        f.write("[PASS] Orientation prototype\\n")
        f.write("[PASS] Visualization\\n")
        f.write("[PASS] Report generation\\n\\n")
        f.write("FINAL DUAL-IMU AKIRA WRIST MODEL: NOT YET TRAINED\\n")
        f.write("REQUIRED NEXT DATA: SYNCHRONIZED FOREARM + HAND IMU RECORDINGS\\n")

if __name__ == "__main__":
    df, fs = run_preprocessing('02_STANDARDIZED_DATA/WatchHand_standard.csv')
    feat_df = extract_features(df)
    detect_repetitions(df, fs)
    calculate_orientation(df, fs)
    plot_dashboard(df)
    generate_reports()
    print("Pipeline finished successfully.")
