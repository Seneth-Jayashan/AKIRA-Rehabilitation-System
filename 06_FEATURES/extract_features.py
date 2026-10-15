import pandas as pd
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
