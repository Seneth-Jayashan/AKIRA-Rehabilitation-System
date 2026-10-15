from .load_data import load_standard_data
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
