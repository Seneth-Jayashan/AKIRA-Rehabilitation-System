import os
import csv
import glob

# Map based on README
gesture_map = {
    '1': ('radial deviation', 'RADIAL_DEVIATION'),
    '2': ('wrist flexion', 'WRIST_FLEXION'),
    '3': ('ulnar deviation', 'ULNAR_DEVIATION'),
    '4': ('wrist extension', 'WRIST_EXTENSION'),
    '5': ('supination', 'SUPINATION'),
    '6': ('pronation', 'PRONATION')
}

output_csv = '03_REFERENCE_DATA/3DCDataset_exercise_reference.csv'
header = ['dataset', 'participant_id', 'trial_id', 'exercise_original', 'exercise_akira', 'gesture_id', 'source_file']
rows_created = 0

with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    
    # Find all IMU files to get the trials
    imu_files = glob.glob('01_DATASETS/EXTRACTED/3DCDataset/**/IMU/*.txt', recursive=True)
    for file in imu_files:
        # Example: Participant1/test/IMU/3dc_IMU_gesture_2_0.txt
        basename = os.path.basename(file)
        parts = basename.replace('.txt', '').split('_')
        if len(parts) >= 4:
            gesture_id = parts[-2]
            trial_id = parts[-1]
            
            # We only map valid wrist movements
            if gesture_id in gesture_map:
                participant_id = file.split(os.sep)[-4] if len(file.split(os.sep)) >= 4 else "Unknown"
                orig, akira = gesture_map[gesture_id]
                
                writer.writerow([
                    '3DCDataset',
                    participant_id,
                    trial_id,
                    orig,
                    akira,
                    gesture_id,
                    file
                ])
                rows_created += 1

print(f"3DCDataset reference created with {rows_created} rows.")
