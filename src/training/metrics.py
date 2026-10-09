import torch
import numpy as np

def reconstruct_angle_degrees(sin_pred, cos_pred):
    # Stabilize vector
    norm = torch.sqrt(sin_pred**2 + cos_pred**2 + 1e-8)
    sin_pred = sin_pred / norm
    cos_pred = cos_pred / norm
    angle_rad = torch.atan2(sin_pred, cos_pred)
    return torch.rad2deg(angle_rad)

def angular_error(pred_deg, target_deg):
    diff = torch.abs(pred_deg - target_deg) % 360
    return torch.min(diff, 360 - diff)

def calculate_metrics(pred_forearm_sin_cos, target_forearm_deg, pred_elbow_deg, target_elbow_deg, mask_elbow):
    # Forearm
    sin_p = pred_forearm_sin_cos[..., 0]
    cos_p = pred_forearm_sin_cos[..., 1]
    pred_f_deg = reconstruct_angle_degrees(sin_p, cos_p)
    
    err_f = angular_error(pred_f_deg, target_forearm_deg)
    mae_f = torch.mean(err_f).item()
    rmse_f = torch.sqrt(torch.mean(err_f**2)).item()
    median_f = torch.median(err_f).item()
    p95_f = torch.quantile(err_f.float(), 0.95).item()
    
    # Elbow (Masked)
    err_e = torch.abs(pred_elbow_deg - target_elbow_deg)
    err_e_masked = err_e[mask_elbow > 0]
    
    if err_e_masked.numel() > 0:
        mae_e = torch.mean(err_e_masked).item()
        rmse_e = torch.sqrt(torch.mean(err_e_masked**2)).item()
    else:
        mae_e = rmse_e = 0.0
        
    return {
        'forearm_mae': mae_f,
        'forearm_rmse': rmse_f,
        'forearm_median': median_f,
        'forearm_p95': p95_f,
        'elbow_mae': mae_e,
        'elbow_rmse': rmse_e
    }
