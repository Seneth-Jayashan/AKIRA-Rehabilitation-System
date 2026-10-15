import os
import csv
import glob
import pandas as pd

output_csv = '02_STANDARDIZED_DATA/WatchHand_standard.csv'
header = [
    'dataset', 'participant_id', 'session_id', 'trial_id', 'timestamp', 
    'exercise_original', 'exercise_akira', 'side', 'sensor_id', 
    'sensor_location', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 
    'label', 'sampling_rate', 'source_file'
]

rows_created = 0
excluded_files = []

with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    
    csv_files = glob.glob('01_DATASETS/EXTRACTED/WatchHand/**/*.csv', recursive=True)
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                participant_id = str(row.get('subNum', 'Unknown'))
                session_id = str(row.get('blockNum', 'Unknown'))
                trial_id = str(row.get('trialNum', 'Unknown'))
                pose = str(row.get('pose', 'Unknown'))
                
                # We determine side from the path (e.g. sub10_samsung_left_video)
                side = 'L' if 'left' in file.lower() else ('R' if 'right' in file.lower() else 'Unknown')
                
                acce = str(row.get('Acce', '')).split()
                gyro = str(row.get('Gyro', '')).split()
                
                # Check if we have matching number of points
                if len(acce) > 0 and len(acce) == len(gyro):
                    points = len(acce) // 4
                    for i in range(points):
                        idx = i * 4
                        ax, ay, az, ts_a = acce[idx], acce[idx+1], acce[idx+2], acce[idx+3]
                        gx, gy, gz, ts_g = gyro[idx], gyro[idx+1], gyro[idx+2], gyro[idx+3]
                        
                        # Timestamps match, so we write the row
                        writer.writerow([
                            'WatchHand',
                            participant_id,
                            session_id,
                            trial_id,
                            ts_a,
                            pose, # original exercise
                            'N/A', # not mapped to AKIRA wrist exercises
                            side,
                            'SamsungWatch',
                            'HAND', # sensor location is hand/wrist
                            ax, ay, az,
                            gx, gy, gz,
                            'pose_ID_' + pose,
                            'NOT AVAILABLE', # freq not explicitly stored per row, could be derived from diff
                            file
                        ])
                        rows_created += 1
                else:
                    excluded_files.append((file, "Mismatch or empty Acce/Gyro lengths"))
        except Exception as e:
            excluded_files.append((file, f"Error processing: {str(e)}"))

print(f"WatchHand standardization created with {rows_created} rows.")
with open('04_METADATA/watchhand_excluded.txt', 'w') as ef:
    for ex in excluded_files:
        ef.write(f"{ex[0]} - {ex[1]}\n")
