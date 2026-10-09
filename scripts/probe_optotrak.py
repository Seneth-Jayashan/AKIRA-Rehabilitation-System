import os
import glob
import pandas as pd
import scipy.io as sio

subj_dir = r"D:\me\project\datasets\raw\Optotrak vs DOT\Optotrak vs DOT\Data\subject 1"
gt_path = os.path.join(subj_dir, "30. Results", "H_Joint_Angles_Aligned_FINAL")
mat = sio.loadmat(gt_path)
j_aligned = mat['j_aligned_eu']

# Let's see the lengths of the Elbow arrays for 'opto'
opto = j_aligned['opto'][0,0]
elbow_arrays = opto['Elbow'][0,0]

print(f"Number of Elbow arrays in opto: {elbow_arrays.shape[1]}")
for i in range(elbow_arrays.shape[1]):
    print(f"  opto Elbow array {i} shape: {elbow_arrays[0,i].shape}")

# Also check other alignment methods to see if they have more arrays
print(f"Fields in j_aligned: {j_aligned.dtype.names}")
for f in j_aligned.dtype.names:
    field_data = j_aligned[f][0,0]
    elbow_data = field_data['Elbow'][0,0]
    print(f"  Field {f} Elbow has {elbow_data.shape[1]} arrays")
    for i in range(min(3, elbow_data.shape[1])): # print first 3
        print(f"    Array {i}: {elbow_data[0,i].shape}")

# Now list lengths of RUA and RLA IMU files for all tasks
print("\nIMU CSV File Lengths:")
task_folders = [d for d in os.listdir(subj_dir) if os.path.isdir(os.path.join(subj_dir, d))]
for folder in task_folders:
    folder_path = os.path.join(subj_dir, folder)
    rua_files = glob.glob(os.path.join(folder_path, "3RUA*.csv"))
    rla_files = glob.glob(os.path.join(folder_path, "4RLA*.csv"))
    
    # Check if there are RUA/RLA inside the folder (they might only be in 0. raw data/DOT)
    rua_len = -1
    rla_len = -1
    
    if rua_files:
        df = pd.read_csv(rua_files[0], skiprows=1)
        rua_len = len(df)
    if rla_files:
        df = pd.read_csv(rla_files[0], skiprows=1)
        rla_len = len(df)
        
    if rua_len > 0 or rla_len > 0:
        print(f"  {folder}: RUA={rua_len}, RLA={rla_len}")
    
# Check raw data folder as well
raw_dot = os.path.join(subj_dir, "0. raw data", "DOT")
if os.path.exists(raw_dot):
    print("\nRaw DOT folder CSV lengths:")
    rua_raw = sorted(glob.glob(os.path.join(raw_dot, "3RUA*.csv")))
    rla_raw = sorted(glob.glob(os.path.join(raw_dot, "4RLA*.csv")))
    
    for rua, rla in zip(rua_raw, rla_raw):
        l1 = len(pd.read_csv(rua, skiprows=1))
        l2 = len(pd.read_csv(rla, skiprows=1))
        print(f"  {os.path.basename(rua)}: RUA={l1}, RLA={l2}")

