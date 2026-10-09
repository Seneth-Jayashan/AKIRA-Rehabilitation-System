import os
import time
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from copy import deepcopy

from src.models.tcn import MultiTaskTCN
from src.models.baselines import LinearBaseline, LSTMBaseline
from src.training.train_tcn import IMUDataset
from src.training.losses import MaskedHuberLoss, MultiTaskLoss
from src.evaluation.evaluate import evaluate_model_full

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(model_name, model, train_loader, val_loader, config):
    device = config['device']
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    huber = MaskedHuberLoss(delta=1.0)
    cce = torch.nn.CrossEntropyLoss()
    mtl_loss = MultiTaskLoss()
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = []
    
    print(f"\n--- Training {model_name} ---")
    
    for epoch in range(config['epochs']):
        start_time = time.time()
        model.train()
        train_loss = train_f = train_e = train_c = 0.0
        
        for X, y_f_sc, y_f_deg, y_e_deg, mask_e, y_c in train_loader:
            X, y_f_sc, y_e_deg, mask_e, y_c = X.to(device), y_f_sc.to(device), y_e_deg.to(device), mask_e.to(device), y_c.to(device)
            
            optimizer.zero_grad()
            p_f, p_e, p_c = model(X)
            
            l_f = F.mse_loss(p_f, y_f_sc)
            l_e = huber(p_e, y_e_deg, mask_e)
            l_c = cce(p_c, y_c)
            
            loss, weights = mtl_loss(l_f, l_e, l_c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_f += l_f.item()
            train_e += l_e.item()
            train_c += l_c.item()
            
        N_train = len(train_loader)
        mtl_loss.update_weights([train_f/N_train, train_e/N_train, train_c/N_train])
        
        # Validation
        model.eval()
        val_loss = val_f = val_e = val_c = 0.0
        with torch.no_grad():
            for X, y_f_sc, y_f_deg, y_e_deg, mask_e, y_c in val_loader:
                X, y_f_sc, y_e_deg, mask_e, y_c = X.to(device), y_f_sc.to(device), y_e_deg.to(device), mask_e.to(device), y_c.to(device)
                p_f, p_e, p_c = model(X)
                
                l_f = F.mse_loss(p_f, y_f_sc)
                l_e = huber(p_e, y_e_deg, mask_e)
                l_c = cce(p_c, y_c)
                
                # Use current weights for val loss
                v_loss = (weights[0]*l_f + weights[1]*l_e + weights[2]*l_c)
                
                val_loss += v_loss.item()
                val_f += l_f.item()
                val_e += l_e.item()
                val_c += l_c.item()
                
        N_val = len(val_loader)
        v_loss_avg = val_loss/N_val
        scheduler.step(v_loss_avg)
        
        dur = time.time() - start_time
        
        # Logging
        hist_entry = {
            'epoch': epoch, 'dur': dur, 'lr': optimizer.param_groups[0]['lr'],
            't_loss': train_loss/N_train, 't_f': train_f/N_train, 't_e': train_e/N_train, 't_c': train_c/N_train,
            'v_loss': v_loss_avg, 'v_f': val_f/N_val, 'v_e': val_e/N_val, 'v_c': val_c/N_val,
            'w_f': weights[0], 'w_e': weights[1], 'w_c': weights[2]
        }
        history.append(hist_entry)
        
        if v_loss_avg < best_val_loss:
            best_val_loss = v_loss_avg
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if epoch % 10 == 0 or patience_counter >= config['patience']:
            print(f"Ep {epoch:3d} | Val Loss: {v_loss_avg:.4f} | Weights: {weights} | LR: {hist_entry['lr']:.5f}")
            
        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch}")
            break
            
    model.load_state_dict(best_state)
    torch.save(best_state, f"models/final/best_{model_name}.pth")
    pd.DataFrame(history).to_csv(f"results/{model_name}_training_history.csv", index=False)
    
    return model

def run_official_experiment():
    config = {
        'seed': 42,
        'batch_size': 64,
        'lr': 1e-3,
        'epochs': 150,
        'patience': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    set_seed(config['seed'])
    
    with open("results/official_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    npz_path = r"D:\me\project\data\processed\windows\dataset.npz"
    train_ds = IMUDataset(npz_path, 'train')
    val_ds = IMUDataset(npz_path, 'val')
    test_ds = IMUDataset(npz_path, 'test')
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)
    
    models_to_train = {
        'tcn': MultiTaskTCN(),
        'lstm': LSTMBaseline(),
        'linear': LinearBaseline()
    }
    
    best_models = {}
    
    for name, model in models_to_train.items():
        best_models[name] = train_model(name, model, train_loader, val_loader, config)
        
    # Official Test Evaluation
    print("\n--- Official Test Evaluation ---")
    test_results = {}
    for name, model in best_models.items():
        metrics, _ = evaluate_model_full(model, test_loader, device=config['device'], prefix='test')
        test_results[name] = metrics
        print(f"{name} Test MAE Elbow: {metrics['elbow_mae']:.2f}, Forearm MAE: {metrics['forearm_mae']:.2f}, Cls Acc: {metrics['cls_acc']:.4f}")
        
    with open("results/test_metrics.json", "w") as f:
        json.dump(test_results, f, indent=4)
        
if __name__ == "__main__":
    run_official_experiment()
