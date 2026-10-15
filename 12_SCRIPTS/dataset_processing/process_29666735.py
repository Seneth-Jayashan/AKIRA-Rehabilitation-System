import os
import csv
import glob

output_csv = '03_REFERENCE_DATA/wrist_kinematic_reference.csv'
header = ['dataset', 'participant_id', 'trial_id', 'timestamp', 'movement', 'flexion_extension_angle', 'radial_ulnar_angle', 'source_file']
rows_created = 0
max_rows = 500000

with open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(header)
    
    motion_tsvs = glob.glob('01_DATASETS/EXTRACTED/29666735_Dataset/**/*_motion.tsv', recursive=True)
    for file in motion_tsvs:
        if rows_created >= max_rows:
            break
            
        basename = os.path.basename(file)
        participant_id = basename.split('_')[0] if '_' in basename else "Unknown"
        trial_id = 'Unknown'
        for part in basename.split('_'):
            if part.startswith('run-'):
                trial_id = part.replace('run-', '')
                break
                
        try:
            with open(file, 'r', encoding='utf-8') as f_in:
                for index, line in enumerate(f_in):
                    if index % 5 != 0: # downsample
                        continue
                    if rows_created >= max_rows:
                        break
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        val_x, val_y = parts[0], parts[1]
                        writer.writerow([
                            '29666735',
                            participant_id,
                            trial_id,
                            str(index),
                            'WristPointingTask',
                            val_x,
                            val_y,
                            file
                        ])
                        rows_created += 1
        except Exception as e:
            pass

print(f"29666735 kinematic reference created with {rows_created} rows.")
