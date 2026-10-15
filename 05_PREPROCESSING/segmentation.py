import numpy as np
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
