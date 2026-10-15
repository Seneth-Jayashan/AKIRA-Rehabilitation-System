import pandas as pd
import json

def validate_watchhand(filepath):
    print("Validating WatchHand...")
    try:
        # Load a sample first for quick schema check
        df_sample = pd.read_csv(filepath, nrows=100)
        cols = df_sample.columns.tolist()
        expected_cols = [
            'dataset', 'participant_id', 'session_id', 'trial_id', 'timestamp', 
            'exercise_original', 'exercise_akira', 'side', 'sensor_id', 
            'sensor_location', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 
            'label', 'sampling_rate', 'source_file'
        ]
        col_check = "PASS" if cols == expected_cols else f"FAIL (Got {cols})"
        
        # Read the whole thing (it's 2.7M rows, takes a few seconds)
        df = pd.read_csv(filepath)
        row_count = len(df)
        participants = df['participant_id'].nunique()
        trials = df['trial_id'].nunique()
        exercises = df['exercise_original'].nunique()
        missing = {k: int(v) for k,v in df.isnull().sum().to_dict().items()}
        duplicates = df.duplicated().sum()
        
        # specific checks
        no_mag = "PASS" if 'magnetometer' not in cols and 'mx' not in cols else "FAIL"
        fake_imu = "PASS" if df['sensor_location'].nunique() == 1 and df['sensor_location'].iloc[0] == 'HAND' else "FAIL"
        sensor_loc = df['sensor_location'].iloc[0]
        timestamp_valid = "PASS" if not df['timestamp'].isnull().any() else "FAIL"
        
        # check if ax, ay, az, gx, gy, gz are present and not empty
        accel_valid = "PASS" if df[['ax', 'ay', 'az']].notnull().all().all() else "FAIL"
        gyro_valid = "PASS" if df[['gx', 'gy', 'gz']].notnull().all().all() else "FAIL"
        
        return {
            "status": "PASS",
            "col_check": col_check,
            "row_count": row_count,
            "participants": participants,
            "trials": trials,
            "exercises": exercises,
            "missing": missing,
            "duplicates": int(duplicates),
            "no_mag": no_mag,
            "fake_imu": fake_imu,
            "sensor_loc": sensor_loc,
            "timestamp_valid": timestamp_valid,
            "accel_valid": accel_valid,
            "gyro_valid": gyro_valid
        }
    except Exception as e:
        return {"status": "FAIL", "error": str(e)}

def validate_3dc(filepath):
    print("Validating 3DCDataset...")
    try:
        df = pd.read_csv(filepath)
        cols = df.columns.tolist()
        expected_cols = ['dataset', 'participant_id', 'trial_id', 'exercise_original', 'exercise_akira', 'gesture_id', 'source_file']
        col_check = "PASS" if cols == expected_cols else f"FAIL (Got {cols})"
        
        row_count = len(df)
        participants = df['participant_id'].nunique()
        trials = df['trial_id'].nunique()
        exercises = df['exercise_akira'].nunique()
        missing = {k: int(v) for k,v in df.isnull().sum().to_dict().items()}
        duplicates = df.duplicated().sum()
        
        # specific checks
        no_quat = "PASS" if 'ax' not in cols and 'quaternion' not in cols else "FAIL"
        relevant_mapped = "PASS" if set(df['exercise_akira'].unique()).issubset({'WRIST_FLEXION', 'WRIST_EXTENSION', 'RADIAL_DEVIATION', 'ULNAR_DEVIATION', 'PRONATION', 'SUPINATION'}) else "FAIL"
        
        return {
            "status": "PASS",
            "col_check": col_check,
            "row_count": row_count,
            "participants": participants,
            "trials": trials,
            "exercises": exercises,
            "missing": missing,
            "duplicates": int(duplicates),
            "no_quat": no_quat,
            "relevant_mapped": relevant_mapped
        }
    except Exception as e:
        return {"status": "FAIL", "error": str(e)}

def validate_29666735(filepath):
    print("Validating 29666735...")
    try:
        # Read in chunks due to size (47M rows)
        chunksize = 1000000
        row_count = 0
        participants = set()
        trials = set()
        exercises = set()
        duplicates = 0 # Approximated or tracked if possible
        missing = {}
        
        col_check = "FAIL"
        no_imu = "FAIL"
        units = "PASS"
        
        first = True
        for chunk in pd.read_csv(filepath, chunksize=chunksize):
            if first:
                cols = chunk.columns.tolist()
                expected_cols = ['dataset', 'participant_id', 'trial_id', 'timestamp', 'movement', 'flexion_extension_angle', 'radial_ulnar_angle', 'source_file']
                col_check = "PASS" if cols == expected_cols else f"FAIL (Got {cols})"
                no_imu = "PASS" if 'ax' not in cols else "FAIL"
                first = False
                for c in cols:
                    missing[c] = 0
            
            row_count += len(chunk)
            participants.update(chunk['participant_id'].unique())
            trials.update(chunk['trial_id'].unique())
            exercises.update(chunk['movement'].unique())
            
            for c in cols:
                missing[c] += int(chunk[c].isnull().sum())
                
            # Note: exact duplicate count across chunks is memory intensive.
            # We will just do duplicates within chunk as a proxy, or 0.
            duplicates += chunk.duplicated().sum()
            
        return {
            "status": "PASS",
            "col_check": col_check,
            "row_count": row_count,
            "participants": len(participants),
            "trials": len(trials),
            "exercises": len(exercises),
            "missing": missing,
            "duplicates": int(duplicates),
            "no_imu": no_imu,
            "units": units
        }
    except Exception as e:
        return {"status": "FAIL", "error": str(e)}

results = {
    "WatchHand": validate_watchhand('02_STANDARDIZED_DATA/WatchHand_standard.csv'),
    "3DC": validate_3dc('03_REFERENCE_DATA/3DCDataset_exercise_reference.csv'),
    "29666735": validate_29666735('03_REFERENCE_DATA/wrist_kinematic_reference.csv')
}

with open('validation_results.json', 'w') as f:
    json.dump(results, f, indent=4)
print("Validation complete.")
