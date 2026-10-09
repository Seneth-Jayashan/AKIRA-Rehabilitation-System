import scipy.io as sio

gt_path = r"D:\me\project\datasets\raw\Optotrak vs DOT\Optotrak vs DOT\Data\subject 1\30. Results\H_Joint_Angles_Aligned_FINAL"
mat = sio.loadmat(gt_path)
j_aligned = mat['j_aligned_eu']

for f in j_aligned.dtype.names:
    opto = j_aligned[f][0,0]
    elbow = opto['Elbow'][0,0]
    print(f"Alignment method '{f}' Elbow shape: {elbow.shape}")
