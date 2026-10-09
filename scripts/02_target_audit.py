import numpy as np
npz_path = r"D:\me\project\data\processed\windows\dataset.npz"
data = np.load(npz_path)
Y_reg = np.concatenate([data['Y_train_reg'], data['Y_val_reg'], data['Y_test_reg']], axis=0)

forearm = Y_reg[:, :, 0]
diff_forearm = np.diff(forearm, axis=1)

mask = np.abs(diff_forearm) > 300
win_idx, frame_idx = np.where(mask)
for i in range(min(5, len(win_idx))):
    w = win_idx[i]
    f = frame_idx[i]
    print(f"Forearm Wrap in window {w} at frame {f}: {forearm[w, f]:.2f} -> {forearm[w, f+1]:.2f} (Diff: {diff_forearm[w,f]:.2f})")
