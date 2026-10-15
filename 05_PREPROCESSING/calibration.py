import numpy as np

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
        
    # Ensure trial_id is kept
    trial_ids = df['trial_id'].copy()
    df = df.groupby('trial_id').apply(calc_bias).reset_index(drop=True)
    if 'trial_id' not in df.columns:
        df['trial_id'] = trial_ids.values

    
    # Write calibration report
    with open('05_PREPROCESSING/calibration_report.md', 'w') as f:
        f.write("# IMU Calibration Report\n\n")
        f.write("No dedicated static calibration segment is available in the public dataset.\n")
        f.write("Gyroscopes were calibrated by subtracting the mean of the first 50 samples of each trial.\n")
        f.write("Accelerometers were left uncalibrated (no bias removed) to preserve the gravity vector.\n")
        f.write("LIMITATION: This is a pseudo-calibration for the prototype only.\n")
        
    return df
