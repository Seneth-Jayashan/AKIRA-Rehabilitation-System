import os

files = {
    "05_PREPROCESSING/load_data.py": """import pandas as pd
def load_standard_data(filepath):
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)
""",

    "05_PREPROCESSING/validate_input.py": """def validate_imu_data(df):
    print("Validating input data...")
    required = ['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    return df
""",

    "05_PREPROCESSING/handle_missing.py": """def handle_missing_values(df):
    print("Handling missing values...")
    # Do not fabricate missing IMU data. Just drop rows where raw accel/gyro is missing
    df_clean = df.dropna(subset=['ax', 'ay', 'az', 'gx', 'gy', 'gz']).copy()
    return df_clean
""",

    "05_PREPROCESSING/filtering.py": """import numpy as np
from scipy.signal import butter, filtfilt

def apply_lowpass_filter(df, cutoff=5.0, fs=50.0):
    print(f"Applying low-pass filter (cutoff={cutoff}Hz, fs={fs}Hz)...")
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    
    for col in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
        # Only filter if we have enough data points, we do it per trial
        df[f'{col}_filtered'] = df.groupby('trial_id')[col].transform(lambda x: filtfilt(b, a, x) if len(x) > 15 else x)
    return df
""",

    "05_PREPROCESSING/calibration.py": """import numpy as np

def calibrate_imu(df):
    print("Calibrating IMU...")
    # We do not have a dedicated static calibration segment.
    # We will estimate gyroscope bias as the mean of the gyroscope signals during the first 1 second of each trial, assuming it is static.
    
    # Calculate dt per trial
    df['timestamp'] = df['timestamp'].astype(float)
    df = df.sort_values(by=['trial_id', 'timestamp'])
    
    # Simple calibration: just subtract mean of first N samples per trial for Gyro
    # For Accel, we assume it's already in g's or m/s^2. We don't invent values.
    
    def calc_bias(group):
        n_samples = min(50, len(group))
        if n_samples == 0: return group
        for col in ['gx', 'gy', 'gz']:
            bias = group[col].iloc[:n_samples].mean()
            group[f'{col}_calibrated'] = group[f'{col}_filtered'] - bias
        for col in ['ax', 'ay', 'az']:
            group[f'{col}_calibrated'] = group[f'{col}_filtered'] # no trivial bias subtraction for accel due to gravity
        return group
        
    df = df.groupby('trial_id').apply(calc_bias).reset_index(drop=True)
    
    # Write calibration report
    with open('05_PREPROCESSING/calibration_report.md', 'w') as f:
        f.write("# IMU Calibration Report\\n\\n")
        f.write("No dedicated static calibration segment is available in the public dataset.\\n")
        f.write("Gyroscopes were calibrated by subtracting the mean of the first 50 samples of each trial.\\n")
        f.write("Accelerometers were left uncalibrated (no bias removed) to preserve the gravity vector.\\n")
        f.write("LIMITATION: This is a pseudo-calibration for the prototype only.\\n")
        
    return df
""",

    "05_PREPROCESSING/normalization.py": """def normalize_data(df):
    # Depending on requirements, we might normalize. But for orientation, we need raw physical units.
    # We leave this as a pass-through.
    return df
""",

    "05_PREPROCESSING/segmentation.py": """import numpy as np
import pandas as pd

def segment_movements(df):
    print("Segmenting movements...")
    df['gyro_mag'] = np.sqrt(df['gx_calibrated']**2 + df['gy_calibrated']**2 + df['gz_calibrated']**2)
    df['acc_mag'] = np.sqrt(df['ax_calibrated']**2 + df['ay_calibrated']**2 + df['az_calibrated']**2)
    
    # Simple thresholding for movement
    threshold = df['gyro_mag'].mean() + 0.5 * df['gyro_mag'].std()
    df['is_moving'] = df['gyro_mag'] > threshold
    
    # Find contiguous segments
    df['segment_id'] = (df['is_moving'] != df['is_moving'].shift()).cumsum()
    
    df.to_csv('06_FEATURES/segmented_movements.csv', index=False)
    return df
""",

    "05_PREPROCESSING/preprocessing_pipeline.py": """from .load_data import load_standard_data
from .validate_input import validate_imu_data
from .handle_missing import handle_missing_values
from .filtering import apply_lowpass_filter
from .calibration import calibrate_imu
from .normalization import normalize_data
from .segmentation import segment_movements

def run_preprocessing(filepath):
    df = load_standard_data(filepath)
    df = validate_imu_data(df)
    df = handle_missing_values(df)
    
    # Estimate sampling frequency (approx)
    # Timestamps are in milliseconds
    dt_ms = df.groupby('trial_id')['timestamp'].diff().mean()
    fs = 1000.0 / dt_ms if dt_ms > 0 else 50.0
    fs = min(max(fs, 10.0), 1000.0) # bound it
    
    df = apply_lowpass_filter(df, cutoff=5.0, fs=fs)
    df = calibrate_imu(df)
    df = normalize_data(df)
    df = segment_movements(df)
    
    return df, fs
""",

    "06_FEATURES/extract_features.py": """import pandas as pd
import numpy as np

def extract_features(df):
    print("Extracting features...")
    # Group by trial and segment where moving
    moving_df = df[df['is_moving'] == True]
    
    features = []
    for (trial, seg), group in moving_df.groupby(['trial_id', 'segment_id']):
        f = {
            'trial_id': trial,
            'segment_id': seg,
            'duration': len(group),
            
            # ACCELEROMETER
            'acc_mag_mean': group['acc_mag'].mean(),
            'acc_mag_max': group['acc_mag'].max(),
            'acc_mag_min': group['acc_mag'].min(),
            'acc_mag_std': group['acc_mag'].std(),
            'acc_mag_rms': np.sqrt(np.mean(group['acc_mag']**2)),
            
            # GYROSCOPE
            'gyro_mag_mean': group['gyro_mag'].mean(),
            'gyro_mag_max': group['gyro_mag'].max(),
            'gyro_mag_min': group['gyro_mag'].min(),
            'gyro_mag_std': group['gyro_mag'].std(),
            'gyro_mag_rms': np.sqrt(np.mean(group['gyro_mag']**2))
        }
        features.append(f)
        
    feat_df = pd.DataFrame(features)
    feat_df.to_csv('06_FEATURES/wrist_features.csv', index=False)
    return feat_df
"""
}

for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
        
# Create __init__.py in 05_PREPROCESSING so it can be imported as a module
with open("05_PREPROCESSING/__init__.py", "w") as f:
    f.write("")
    
print("Pipeline modules generated successfully.")
