import os
import scipy.io as sio

subj_dir = r"D:\me\project\datasets\raw\Optotrak vs DOT\Optotrak vs DOT\Data\subject 1"
rua_eflex = os.path.join(subj_dir, r"2. ElbowFlexCali\RUA-eFlex")
mat = sio.loadmat(rua_eflex)
IM = mat['IM'][0,0]
print("Acc shape in RUA-eFlex:", IM['Acc'].shape)
