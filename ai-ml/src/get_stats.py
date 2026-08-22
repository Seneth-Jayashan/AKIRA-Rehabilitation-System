import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = "../data/processed"
WINDOW_SIZE = 100

all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

total_files = len(all_files)
total_windows = 0
valid_files = 0
too_short_files = 0

datasets = {}
activities = {}
unused_files = 0

for file in all_files:
    try:
        df = pd.read_csv(file)
        ds_name = df['dataset_name'].values[0]
        if ds_name not in datasets:
            datasets[ds_name] = 0
        datasets[ds_name] += 1
        
        if len(df) < WINDOW_SIZE:
            too_short_files += 1
            unused_files += 1
            continue
            
        valid_files += 1
        
        # Count windows
        windows = len(range(0, len(df) - WINDOW_SIZE + 1, 50))
        total_windows += windows
        
    except Exception as e:
        unused_files += 1
        pass

print(f"Total processed files: {total_files}")
print(f"Valid files (>= {WINDOW_SIZE} rows): {valid_files}")
print(f"Unused files (too short): {too_short_files}")
print(f"Percentage Unused: {too_short_files/total_files*100:.2f}%")
print(f"Datasets present in processed: {datasets}")
print(f"Total windows generated: {total_windows}")

# For Train/Test splits (usually 80% train, 20% test of subjects)
# Since subjects are split, the actual window count for test is roughly 20%, but we can do a quick calculation
subjects = []
for file in all_files:
    try:
        df = pd.read_csv(file)
        if len(df) >= WINDOW_SIZE:
            subjects.append(df['subject_id'].values[0])
    except:
        pass
        
unique_subs = len(set(subjects))
print(f"Total Unique Subjects: {unique_subs}")
print(f"Approx Test Subjects: {int(unique_subs * 0.2)}")
