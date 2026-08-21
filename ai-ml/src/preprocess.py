import os
import glob
import zipfile
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import re

OUTPUT_DIR = "../data/processed"
TARGET_FS = 50.0  # Hz
TARGET_DT = 1.0 / TARGET_FS

# Hardcoded age dictionary from KneE-PAD metadata.txt
KNEEPAD_AGES = {
    'Subject_1': 27, 'Subject_2': 47, 'Subject_3': 56, 'Subject_4': 48,
    'Subject_5': 31, 'Subject_6': 54, 'Subject_7': 63, 'Subject_8': 43,
    'Subject_9': 55, 'Subject_10': 51, 'Subject_11': 49, 'Subject_12': 50,
    'Subject_13': 40, 'Subject_14': 62, 'Subject_15': 38, 'Subject_16': 55,
    'Subject_17': 58, 'Subject_18': 33, 'Subject_19': 18, 'Subject_20': 51,
    'Subject_21': 27, 'Subject_22': 41, 'Subject_23': 51, 'Subject_24': 51,
    'Subject_25': 29, 'Subject_26': 42, 'Subject_27': 68, 'Subject_28': 53,
    'Subject_29': 41, 'Subject_30': 47, 'Subject_31': 56
}

def resample_data(time, data, target_dt):
    if len(time) < 2:
        return time, data
    # Create new time vector from 0 to max_time with step target_dt
    max_time = time[-1] - time[0]
    new_time = np.arange(0, max_time, target_dt)
    
    # Interpolate each column
    interpolator = interp1d(time - time[0], data, axis=0, fill_value="extrapolate")
    new_data = interpolator(new_time)
    return new_time, new_data

def save_to_csv(dataset_name, subject_id, age, activity_label, correct_label, new_time, new_data, output_file):
    df = pd.DataFrame(new_data, columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
    df['timestamp'] = new_time
    df['dataset_name'] = dataset_name
    df['subject_id'] = subject_id
    df['age'] = age
    df['activity_label'] = activity_label
    df['correct_label'] = correct_label
    
    # Reorder columns
    cols = ['timestamp', 'dataset_name', 'subject_id', 'age', 'activity_label', 'correct_label', 
            'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    df = df[cols]
    
    df.to_csv(output_file, index=False)

def process_kneepad():
    print("Processing KneE-PAD...")
    zip_path = "../data/raw/12112951/dataset.zip"
    if not os.path.exists(zip_path):
        print(f"Skipping KneE-PAD, {zip_path} not found.")
        return
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        npy_files = [f for f in z.namelist() if f.endswith('imu.npy')]
        for i, npy_file in enumerate(npy_files):
            parts = npy_file.split('/')
            if len(parts) < 4: continue
            subject_id = parts[1]
            label = parts[2] # 0, 1, etc.
            trial = parts[3]
            
            age = KNEEPAD_AGES.get(subject_id, -1)
            
            with z.open(npy_file) as f:
                data = np.load(f)
                
            # Extract first 6 rows, transpose to (num_samples, 6)
            imu_data = data[:6, :].T
            
            fs = 148.14
            time = np.arange(imu_data.shape[0]) / fs
            
            new_time, new_data = resample_data(time, imu_data, TARGET_DT)
            
            # Map label to activity/correct
            try:
                label_int = int(label)
                if label_int in [0, 1, 2]:
                    activity = "Squat"
                elif label_int in [3, 4, 5]:
                    activity = "Leg Extension"
                elif label_int in [6, 7, 8]:
                    activity = "Walking"
                else:
                    activity = "Unknown"
                    
                # Detailed compensation mapping
                if label_int in [0, 3, 6]:
                    correct = "Correct"
                elif label_int == 1:
                    correct = "Squat_Weight_Transfer"
                elif label_int == 2:
                    correct = "Squat_Leg_Front"
                elif label_int == 4:
                    correct = "Leg_Ext_No_ROM"
                elif label_int == 5:
                    correct = "Leg_Ext_Lift"
                elif label_int == 7:
                    correct = "Walk_No_Ext"
                elif label_int == 8:
                    correct = "Walk_Hip_Abduction"
                else:
                    correct = "Incorrect"
            except ValueError:
                activity = "Unknown"
                correct = "Unknown"
            
            out_file = os.path.join(OUTPUT_DIR, f"kneepad_{subject_id}_{activity}_{trial}.csv")
            save_to_csv("KneE-PAD", subject_id, age, activity, correct, new_time, new_data, out_file)

def process_phytmo():
    print("Processing PHYTMO...")
    base_dir = "../data/raw/PHYTMO/inertial/lower"
    if not os.path.exists(base_dir):
        print(f"Skipping PHYTMO, {base_dir} not found.")
        return
        
    csv_files = glob.glob(os.path.join(base_dir, "**/*.csv"), recursive=True)
    for i, file_path in enumerate(csv_files):
        try:
            df = pd.read_csv(file_path)
            if 'Time (s)' not in df.columns:
                continue
            time = df['Time (s)'].values
            
            cols = ['Accelerometer X (g)', 'Accelerometer Y (g)', 'Accelerometer Z (g)',
                    'Gyroscope X (deg/s)', 'Gyroscope Y (deg/s)', 'Gyroscope Z (deg/s)']
            
            if not all(c in df.columns for c in cols):
                continue
                
            imu_data = df[cols].values
            new_time, new_data = resample_data(time, imu_data, TARGET_DT)
            
            filename = os.path.basename(file_path)
            subject_id = "PHYTMO_" + filename.split('_')[0]
            age = -1 # Age not explicitly provided in file paths
            activity = "Rehab/Gait"
            correct = "Unknown"
            
            out_file = os.path.join(OUTPUT_DIR, f"phytmo_{i}_{filename}")
            save_to_csv("PHYTMO", subject_id, age, activity, correct, new_time, new_data, out_file)
        except Exception as e:
            pass

def process_pt_dataset():
    print("Processing Physical Therapy Exercises Dataset...")
    base_dir = "../data/raw/physical+therapy+exercises+dataset"
    if not os.path.exists(base_dir):
        print(f"Skipping PT dataset, {base_dir} not found.")
        return
        
    txt_files = glob.glob(os.path.join(base_dir, "s*/e*/u*/*.txt"))
    for file_path in txt_files:
        if "template_times" in file_path:
            continue
            
        try:
            df = pd.read_csv(file_path, sep=';')
            if 'time index' not in df.columns:
                continue
                
            fs = 25.0
            time = df['time index'].values / fs
            
            cols = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
            if not all(c in df.columns for c in cols):
                continue
                
            imu_data = df[cols].values
            # Standardize: Acc to g (divide by 9.81), Gyro to deg/s (multiply by 180/pi)
            imu_data[:, 0:3] = imu_data[:, 0:3] / 9.81
            imu_data[:, 3:6] = imu_data[:, 3:6] * (180.0 / np.pi)
            
            new_time, new_data = resample_data(time, imu_data, TARGET_DT)
            
            parts = file_path.replace('\\', '/').split('/')
            subject_id = parts[-4] # s1
            age = -1 # Exact ages not explicitly provided
            activity = parts[-3]   # e1
            correct = "Correct/Test" if "test.txt" in os.path.basename(file_path) else "Template"
            
            out_file = os.path.join(OUTPUT_DIR, f"pt_{subject_id}_{activity}_{parts[-2]}_{os.path.basename(file_path).replace('.txt', '.csv')}")
            save_to_csv("Physical_Therapy", subject_id, age, activity, correct, new_time, new_data, out_file)
        except Exception as e:
            pass

def process_10m_walk():
    print("Processing 10m Walk Test Dataset...")
    base_dir = "../data/raw/Dataset related to Lower limb movement in 10 Meter Walk Test"
    if not os.path.exists(base_dir):
        print(f"Skipping 10m Walk, {base_dir} not found.")
        return
        
    subject_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for subj in subject_dirs:
        subj_dir = os.path.join(base_dir, subj)
        
        # Read YAML to find age
        age = -1
        yaml_file = os.path.join(subj_dir, "other_data.yml")
        if os.path.exists(yaml_file):
            try:
                with open(yaml_file, 'r') as yf:
                    yaml_content = yf.read()
                    age_match = re.search(r'age:\s*(\d+)', yaml_content)
                    if age_match:
                        age = int(age_match.group(1))
            except Exception:
                pass
        
        # We look for TILE1 or TILE3
        for tile in ['TILE1', 'TILE2', 'TILE3']:
            acc_file = os.path.join(subj_dir, f"{tile}_Accelerometer.csv")
            gyro_file = os.path.join(subj_dir, f"{tile}_Gyroscope.csv")
            
            if os.path.exists(acc_file) and os.path.exists(gyro_file):
                try:
                    df_acc = pd.read_csv(acc_file)
                    df_gyro = pd.read_csv(gyro_file)
                    
                    df_acc = df_acc.rename(columns={'X': 'acc_x', 'Y': 'acc_y', 'Z': 'acc_z'})
                    df_gyro = df_gyro.rename(columns={'X': 'gyro_x', 'Y': 'gyro_y', 'Z': 'gyro_z'})
                    
                    t_acc = (df_acc['timestamp'].values - df_acc['timestamp'].values[0]) / 1000.0
                    t_gyro = (df_gyro['timestamp'].values - df_gyro['timestamp'].values[0]) / 1000.0
                    
                    max_t = min(t_acc[-1], t_gyro[-1])
                    if max_t <= 0:
                        continue
                        
                    new_time = np.arange(0, max_t, TARGET_DT)
                    
                    acc_interp = interp1d(t_acc, df_acc[['acc_x', 'acc_y', 'acc_z']].values, axis=0, fill_value="extrapolate")
                    gyro_interp = interp1d(t_gyro, df_gyro[['gyro_x', 'gyro_y', 'gyro_z']].values, axis=0, fill_value="extrapolate")
                    
                    new_acc = acc_interp(new_time)
                    new_gyro = gyro_interp(new_time)
                    
                    new_data = np.hstack((new_acc, new_gyro))
                    
                    activity = "10m_Walk"
                    correct = "N/A"
                    
                    out_file = os.path.join(OUTPUT_DIR, f"10mwalk_{subj}_{tile}.csv")
                    save_to_csv("10m_Walk", subj, age, activity, correct, new_time, new_data, out_file)
                except Exception as e:
                    pass

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_kneepad()
    process_phytmo()
    process_pt_dataset()
    process_10m_walk()
    print("Done!")

if __name__ == "__main__":
    main()
