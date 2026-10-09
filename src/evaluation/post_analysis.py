import os
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, savgol_filter
from sklearn.metrics import confusion_matrix, r2_score, accuracy_score, precision_recall_fscore_support

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.tcn import MultiTaskTCN
from src.models.baselines import LSTMBaseline, LinearBaseline
from src.training.train_tcn import IMUDataset
from src.training.metrics import reconstruct_angle_degrees, angular_error

def setup_env():
    os.makedirs('reports/plots', exist_ok=True)
    os.makedirs('results/post_analysis', exist_ok=True)

def hash_model(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def get_predictions(model, loader, device='cpu'):
    model.eval()
    f_p, f_t, e_p, e_t, c_p, c_t, masks, metas = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for i in range(len(loader.dataset)):
            X, y_f_sc, y_f_deg, y_e_deg, mask_e, y_c = loader.dataset[i]
            meta = loader.dataset.meta[i]
            X = X.unsqueeze(0).to(device)
            p_f, p_e, p_c = model(X)
            
            f_p.append(p_f.cpu().squeeze(0))
            f_t.append(y_f_deg.cpu())
            e_p.append(p_e.cpu().squeeze(0))
            e_t.append(y_e_deg.cpu())
            c_p.append(torch.argmax(p_c.cpu(), dim=1).item())
            c_t.append(y_c.item())
            masks.append(mask_e.cpu())
            metas.append(meta)
            
    return (torch.stack(f_p), torch.stack(f_t), torch.stack(e_p), 
            torch.stack(e_t), torch.tensor(c_p), torch.tensor(c_t), 
            torch.stack(masks), metas)

def run_post_analysis():
    setup_env()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Freeze
    model_path = "models/final/best_tcn.pth"
    model_hash = hash_model(model_path)
    
    with open('results/official_config.json', 'r') as f:
        config = json.load(f)
        
    freeze_meta = {
        'model_hash': model_hash,
        'architecture': 'MultiTaskTCN',
        'checkpoint': model_path,
        'config': config
    }
    with open('results/post_analysis/frozen_metadata.json', 'w') as f:
        json.dump(freeze_meta, f, indent=4)
        
    # Load model
    tcn = MultiTaskTCN()
    tcn.load_state_dict(torch.load(model_path, map_location=device))
    tcn.to(device)
    
    # Data
    npz_path = r"D:\me\project\data\processed\windows\dataset.npz"
    test_ds = IMUDataset(npz_path, 'test')
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    with open(r'D:\me\project\data\processed\class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    print("Computing TCN predictions...")
    f_p, f_t, e_p, e_t, c_p, c_t, masks, metas = get_predictions(tcn, test_loader, device)
    
    # Reconstruct Forearm Angle
    sin_p = f_p[..., 0]
    cos_p = f_p[..., 1]
    f_p_deg = reconstruct_angle_degrees(sin_p, cos_p)
    f_err = angular_error(f_p_deg, f_t) # [N, 240]
    
    e_p_squeeze = e_p.squeeze(-1)
    e_t_squeeze = e_t.squeeze(-1)
    masks_squeeze = masks.squeeze(-1)
    e_err = torch.abs(e_p_squeeze - e_t_squeeze) * masks_squeeze
    
    # 2. Per-Subject Analysis
    metas = [{"subject": m.split("::")[0], "exercise": m.split("::")[1]} for m in metas]; subjects = [m["subject"] for m in metas]
    df_meta = pd.DataFrame(metas)
    df_meta['subject'] = subjects
    df_meta['c_p'] = c_p.numpy()
    df_meta['c_t'] = c_t.numpy()
    df_meta['f_mae'] = f_err.mean(dim=1).numpy()
    df_meta['f_rmse'] = torch.sqrt((f_err**2).mean(dim=1)).numpy()
    
    # Masked elbow MAE
    e_mae_list = []
    e_rmse_list = []
    for i in range(len(e_err)):
        valid = masks_squeeze[i] > 0
        if valid.sum() > 0:
            e_mae_list.append(e_err[i][valid].mean().item())
            e_rmse_list.append(torch.sqrt((e_err[i][valid]**2).mean()).item())
        else:
            e_mae_list.append(0)
            e_rmse_list.append(0)
    df_meta['e_mae'] = e_mae_list
    df_meta['e_rmse'] = e_rmse_list
    
    subj_results = []
    for subj in df_meta['subject'].unique():
        sub_df = df_meta[df_meta['subject'] == subj]
        acc = accuracy_score(sub_df['c_t'], sub_df['c_p'])
        f1 = precision_recall_fscore_support(sub_df['c_t'], sub_df['c_p'], average='macro', zero_division=0)[2]
        
        # Flatten valid elbow errors
        sub_idx = sub_df.index
        valid_e_errs = e_err[sub_idx][masks_squeeze[sub_idx] > 0]
        
        subj_results.append({
            'subject': subj,
            'elbow_mae': sub_df['e_mae'].mean(),
            'elbow_rmse': sub_df['e_rmse'].mean(),
            'forearm_mae': sub_df['f_mae'].mean(),
            'forearm_rmse': sub_df['f_rmse'].mean(),
            'cls_acc': acc,
            'cls_f1_macro': f1
        })
        
    pd.DataFrame(subj_results).to_csv('results/post_analysis/per_subject_results.csv', index=False)
    
    # 3. Per-Exercise Analysis
    ex_results = []
    for ex_id in df_meta['c_t'].unique():
        ex_name = idx_to_class[ex_id]
        ex_df = df_meta[df_meta['c_t'] == ex_id]
        
        prec, rec, f1, _ = precision_recall_fscore_support(
            (df_meta['c_t'] == ex_id).astype(int), 
            (df_meta['c_p'] == ex_id).astype(int), 
            average='binary', zero_division=0)
            
        ex_results.append({
            'exercise': ex_name,
            'windows': len(ex_df),
            'subjects': ex_df['subject'].nunique(),
            'elbow_mae': ex_df['e_mae'].mean(),
            'forearm_mae': ex_df['f_mae'].mean(),
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
        
    pd.DataFrame(ex_results).to_csv('results/post_analysis/per_exercise_results.csv', index=False)
    
    # 4. Error vs Angle
    def plot_binned_errors(true_vals, err_vals, bins, title, filename, circular=False):
        t_flat = true_vals.flatten()
        e_flat = err_vals.flatten()
        
        df_bins = pd.DataFrame({'true': t_flat, 'err': e_flat})
        df_bins['bin'] = pd.cut(df_bins['true'], bins=bins)
        mean_errs = df_bins.groupby('bin')['err'].mean()
        
        plt.figure(figsize=(10, 6))
        mean_errs.plot(kind='bar', color='skyblue')
        plt.title(title)
        plt.ylabel("Mean Absolute Error (degrees)")
        plt.xlabel("Angle Range (degrees)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'reports/plots/{filename}')
        plt.close()
        
    # Valid elbow masks only
    valid_e_mask_flat = masks_squeeze.flatten() > 0
    plot_binned_errors(e_t_squeeze.flatten()[valid_e_mask_flat].numpy(), 
                       e_err.flatten()[valid_e_mask_flat].numpy(),
                       bins=np.arange(0, 181, 30),
                       title="Elbow Error vs Angle",
                       filename="elbow_error_vs_angle.png")
                       
    plot_binned_errors(f_t.flatten().numpy(), 
                       f_err.flatten().numpy(),
                       bins=np.arange(-180, 181, 45),
                       title="Forearm Error vs Angle",
                       filename="forearm_error_vs_angle.png", circular=True)
                       
    # 5. Temporal Trajectory Analysis
    # Best, Median, Worst based on joint MAE
    joint_mae = (df_meta['e_mae'] + df_meta['f_mae']) / 2
    best_idx = joint_mae.idxmin()
    worst_idx = joint_mae.idxmax()
    median_idx = (joint_mae - joint_mae.median()).abs().idxmin()
    
    def plot_traj(idx, name):
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].plot(f_t[idx].numpy(), label='True', color='blue')
        axs[0].plot(f_p_deg[idx].numpy(), label='TCN', color='orange', linestyle='--')
        axs[0].set_title(f"Forearm Rotation ({name} - Window {idx})")
        axs[0].set_ylabel("Degrees")
        axs[0].legend()
        
        axs[1].plot(e_t_squeeze[idx].numpy(), label='True', color='green')
        axs[1].plot(e_p_squeeze[idx].numpy(), label='TCN', color='red', linestyle='--')
        axs[1].set_title(f"Elbow Flexion ({name} - Window {idx})")
        axs[1].set_ylabel("Degrees")
        axs[1].legend()
        plt.tight_layout()
        plt.savefig(f'reports/plots/traj_{name}.png')
        plt.close()
        
    plot_traj(best_idx, 'best')
    plot_traj(median_idx, 'median')
    plot_traj(worst_idx, 'worst')
    
    # 8-11. Rehabilitation Metrics
    rehab_metrics = []
    
    # Sampling rate is 10 Hz according to prior data preparation
    dt = 0.1
    
    for i in range(len(test_ds)):
        # Calculate velocity using Savitzky-Golay for smoothing derivative
        f_true_np = f_t[i].numpy()
        f_pred_np = f_p_deg[i].numpy()
        e_true_np = e_t_squeeze[i].numpy()
        e_pred_np = e_p_squeeze[i].numpy()
        
        # Velocity (deg/s)
        # Using np.gradient for simplicity
        f_v_t = np.gradient(f_true_np, dt)
        f_v_p = np.gradient(f_pred_np, dt)
        e_v_t = np.gradient(e_true_np, dt)
        e_v_p = np.gradient(e_pred_np, dt)
        
        # ROM
        f_rom_t = np.max(f_true_np) - np.min(f_true_np)
        f_rom_p = np.max(f_pred_np) - np.min(f_pred_np)
        e_rom_t = np.max(e_true_np) - np.min(e_true_np)
        e_rom_p = np.max(e_pred_np) - np.min(e_pred_np)
        
        # Repetitions (peak finding)
        # We find peaks with prominence at least 15% of ROM to avoid noise
        peaks_e_t, _ = find_peaks(e_true_np, prominence=max(10, e_rom_t*0.15))
        peaks_e_p, _ = find_peaks(e_pred_np, prominence=max(10, e_rom_p*0.15))
        
        rehab_metrics.append({
            'window_idx': i,
            'exercise': idx_to_class[c_t[i].item()],
            'e_rom_true': e_rom_t,
            'e_rom_pred': e_rom_p,
            'f_rom_true': f_rom_t,
            'f_rom_pred': f_rom_p,
            'e_v_mae': np.abs(e_v_t - e_v_p).mean(),
            'f_v_mae': np.abs(f_v_t - f_v_p).mean(),
            'e_reps_true': len(peaks_e_t),
            'e_reps_pred': len(peaks_e_p)
        })
        
    df_rehab = pd.DataFrame(rehab_metrics)
    df_rehab.to_csv('results/post_analysis/rehabilitation_metrics.csv', index=False)
    
    # Confusion Matrix
    cm = confusion_matrix(df_meta['c_t'], df_meta['c_p'])
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[idx_to_class[i] for i in range(20)],
                yticklabels=[idx_to_class[i] for i in range(20)])
    plt.title('Exercise Confusion Matrix (TCN)')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig('reports/plots/confusion_matrix.png')
    plt.close()
    
    print("Post-analysis script generated artifacts successfully.")

if __name__ == "__main__":
    run_post_analysis()
