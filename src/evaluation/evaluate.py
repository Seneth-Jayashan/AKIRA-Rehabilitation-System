import torch
import numpy as np
from sklearn.metrics import r2_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
import json

from src.training.metrics import calculate_metrics, reconstruct_angle_degrees

def evaluate_model_full(model, dataloader, device='cpu', prefix='test', meta_list=None):
    model.eval()
    all_f_pred, all_f_true = [], []
    all_e_pred, all_e_true = [], []
    all_e_mask = []
    all_c_pred, all_c_true = [], []
    
    with torch.no_grad():
        for X, y_f_sc, y_f_deg, y_e_deg, mask_e, y_c in dataloader:
            X = X.to(device)
            p_f, p_e, p_c = model(X)
            
            all_f_pred.append(p_f.cpu())
            all_f_true.append(y_f_deg.cpu())
            all_e_pred.append(p_e.cpu())
            all_e_true.append(y_e_deg.cpu())
            all_e_mask.append(mask_e.cpu())
            all_c_pred.append(torch.argmax(p_c.cpu(), dim=1))
            all_c_true.append(y_c.cpu())
            
    all_f_pred = torch.cat(all_f_pred, dim=0)
    all_f_true = torch.cat(all_f_true, dim=0)
    all_e_pred = torch.cat(all_e_pred, dim=0)
    all_e_true = torch.cat(all_e_true, dim=0)
    all_e_mask = torch.cat(all_e_mask, dim=0)
    all_c_pred = torch.cat(all_c_pred, dim=0)
    all_c_true = torch.cat(all_c_true, dim=0)
    
    # Base regression metrics
    metrics = calculate_metrics(all_f_pred, all_f_true, all_e_pred, all_e_true, all_e_mask)
    
    # Elbow R2 and correlation
    e_pred_flat = all_e_pred[all_e_mask > 0].numpy()
    e_true_flat = all_e_true[all_e_mask > 0].numpy()
    if len(e_true_flat) > 0:
        metrics['elbow_r2'] = r2_score(e_true_flat, e_pred_flat)
        metrics['elbow_corr'] = np.corrcoef(e_true_flat, e_pred_flat)[0, 1]
    
    # Classification metrics
    c_true_np = all_c_true.numpy()
    c_pred_np = all_c_pred.numpy()
    
    acc = accuracy_score(c_true_np, c_pred_np)
    prec_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(c_true_np, c_pred_np, average='macro', zero_division=0)
    prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(c_true_np, c_pred_np, average='weighted', zero_division=0)
    
    metrics.update({
        'cls_acc': acc,
        'cls_f1_macro': f1_mac,
        'cls_f1_weighted': f1_wt,
        'cls_precision': prec_mac,
        'cls_recall': rec_mac
    })
    
    # Return raw arrays for further plotting/analysis
    return metrics, (all_f_pred, all_f_true, all_e_pred, all_e_true, all_c_pred, all_c_true, all_e_mask)
