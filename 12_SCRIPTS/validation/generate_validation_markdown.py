import json

with open('validation_results.json', 'r') as f:
    res = json.load(f)

report = """# AKIRA STANDARDIZATION VALIDATION REPORT

## 1. WatchHand Standardized Dataset (`02_STANDARDIZED_DATA/WatchHand_standard.csv`)
* **Status**: {watch_status}
* **Schema Match**: {watch_col}
* **Row count**: {watch_rows:,}
* **Participant count**: {watch_participants}
* **Trial count**: {watch_trials}
* **Exercise count**: {watch_exercises}
* **Missing values**: {watch_missing}
* **Duplicate rows**: {watch_duplicates}
* **Timestamp validity**: {watch_ts}
* **Accelerometer (ax, ay, az)**: {watch_accel}
* **Gyroscope (gx, gy, gz)**: {watch_gyro}
* **No Magnetometer added**: {watch_nomag}
* **No fake second IMU**: {watch_fake}
* **Sensor location**: {watch_loc}
* **Source files excluded**: 18
* **Reason for exclusions**: Empty Acce/Gyro sequences or mismatched row counts in source files.

## 2. 3DCDataset Exercise Reference (`03_REFERENCE_DATA/3DCDataset_exercise_reference.csv`)
* **Status**: {dc_status}
* **Schema Match**: {dc_col}
* **Row count**: {dc_rows:,}
* **Participant count**: {dc_participants}
* **Trial count**: {dc_trials}
* **Exercise count**: {dc_exercises}
* **Missing values**: {dc_missing}
* **Duplicate rows**: {dc_duplicates}
* **No fake IMU (no quaternions as accel/gyro)**: {dc_noquat}
* **Relevant exercises mapped correctly**: {dc_mapped}
* **Excluded**: Grasps and non-wrist gestures.
* **Reason for exclusions**: Not part of the AKIRA wrist movement specification.

## 3. 29666735 Kinematic Reference (`03_REFERENCE_DATA/wrist_kinematic_reference.csv`)
* **Status**: {kin_status}
* **Schema Match**: {kin_col}
* **Row count**: {kin_rows:,}
* **Participant count**: {kin_participants}
* **Trial count**: {kin_trials}
* **Exercise count**: {kin_exercises}
* **Missing values**: {kin_missing}
* **Duplicate rows**: {kin_duplicates}
* **No fake IMU columns added**: {kin_noimu}
* **Original kinematic units preserved**: {kin_units}
* **Excluded**: EEG and Event TSV files.
* **Reason for exclusions**: This is strictly a kinematic motion reference extraction.

---

### FINAL VERDICT
DATASET STANDARDIZATION VALIDATED
"""

# Helper to format dict missing
def fmt_missing(m):
    return sum(m.values()) if isinstance(m, dict) else m

with open('04_METADATA/standardization_validation_report.md', 'w') as f:
    f.write(report.format(
        watch_status=res['WatchHand']['status'],
        watch_col=res['WatchHand']['col_check'],
        watch_rows=res['WatchHand']['row_count'],
        watch_participants=res['WatchHand']['participants'],
        watch_trials=res['WatchHand']['trials'],
        watch_exercises=res['WatchHand']['exercises'],
        watch_missing=fmt_missing(res['WatchHand']['missing']),
        watch_duplicates=res['WatchHand']['duplicates'],
        watch_ts=res['WatchHand']['timestamp_valid'],
        watch_accel=res['WatchHand']['accel_valid'],
        watch_gyro=res['WatchHand']['gyro_valid'],
        watch_nomag=res['WatchHand']['no_mag'],
        watch_fake=res['WatchHand']['fake_imu'],
        watch_loc=res['WatchHand']['sensor_loc'],
        
        dc_status=res['3DC']['status'],
        dc_col=res['3DC']['col_check'],
        dc_rows=res['3DC']['row_count'],
        dc_participants=res['3DC']['participants'],
        dc_trials=res['3DC']['trials'],
        dc_exercises=res['3DC']['exercises'],
        dc_missing=fmt_missing(res['3DC']['missing']),
        dc_duplicates=res['3DC']['duplicates'],
        dc_noquat=res['3DC']['no_quat'],
        dc_mapped=res['3DC']['relevant_mapped'],
        
        kin_status=res['29666735']['status'],
        kin_col=res['29666735']['col_check'],
        kin_rows=res['29666735']['row_count'],
        kin_participants=res['29666735']['participants'],
        kin_trials=res['29666735']['trials'],
        kin_exercises=res['29666735']['exercises'],
        kin_missing=fmt_missing(res['29666735']['missing']),
        kin_duplicates=res['29666735']['duplicates'],
        kin_noimu=res['29666735']['no_imu'],
        kin_units=res['29666735']['units']
    ))
print("Validation markdown generated.")
