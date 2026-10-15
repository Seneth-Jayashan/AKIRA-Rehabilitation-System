import numpy as np
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
