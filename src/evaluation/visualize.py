import matplotlib.pyplot as plt
import numpy as np
import torch
from src.training.metrics import reconstruct_angle_degrees

def plot_predictions(y_f_pred, y_f_true, y_e_pred, y_e_true, num_samples=3, save_dir='reports'):
    # y_f_pred is [N, 240, 2] in sin/cos
    # y_f_true is [N, 240] in deg
    sin_p = y_f_pred[..., 0]
    cos_p = y_f_pred[..., 1]
    y_f_pred_deg = reconstruct_angle_degrees(sin_p, cos_p).numpy()
    y_f_true = y_f_true.numpy()
    
    y_e_pred = y_e_pred.numpy().squeeze(-1)
    y_e_true = y_e_true.numpy().squeeze(-1)
    
    for i in range(num_samples):
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        # Forearm
        axs[0].plot(y_f_true[i], label='True', color='blue')
        axs[0].plot(y_f_pred_deg[i], label='Predicted', color='orange', linestyle='--')
        axs[0].set_title(f"Forearm Rotation (Sample {i})")
        axs[0].set_ylabel("Degrees")
        axs[0].legend()
        
        # Elbow
        axs[1].plot(y_e_true[i], label='True', color='green')
        axs[1].plot(y_e_pred[i], label='Predicted', color='red', linestyle='--')
        axs[1].set_title(f"Elbow Flexion (Sample {i})")
        axs[1].set_ylabel("Degrees")
        axs[1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/pred_sample_{i}.png")
        plt.close()
