import os
import pandas as pd
import numpy as np
import scipy.io as sio

subj_dir = r"D:\me\project\datasets\raw\Optotrak vs DOT\Optotrak vs DOT\Data\subject 1"
raw_csv = os.path.join(subj_dir, r"0. raw data\DOT\3RUA_0A8BB2DFBE36_20230110_155009.csv")
mat_file = os.path.join(subj_dir, r"2. ElbowFlexCali\RUA-eFlex")

df = pd.read_csv(raw_csv, skiprows=1)
mat = sio.loadmat(mat_file)
rua = mat['rua']

print("CSV snippet (Acc_X, Acc_Y, Acc_Z):")
print(df[['Acc_X', 'Acc_Y', 'Acc_Z']].head())

print("CSV snippet (Gyr_X, Gyr_Y, Gyr_Z):")
print(df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].head())

print("\nMAT snippet (first 3 rows, all cols):")
for i in range(18):
    print(f"Col {i}: {rua[:3, i]}")
