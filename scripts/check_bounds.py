import scipy.io as sio

split_mat = sio.loadmat(r"D:\me\project\datasets\raw\Optotrak vs DOT\Optotrak vs DOT\Data\subject 1\30. Results\I_tasks_split")
mov = split_mat['mov']
print("mov:")
for i in range(mov.shape[1]):
    print(f"  mov {i}: {mov[0,i]}")
