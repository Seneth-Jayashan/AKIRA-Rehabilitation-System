import os
import time
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy

from src.models.tcn import MultiTaskTCN
from src.training.train_tcn import IMUDataset
from src.training.losses import MaskedHuberLoss, MultiTaskLoss
from src.evaluation.evaluate import evaluate_model_full
from src.training.train_official import set_seed

def train_ablation(name, config, active_inputs='all', active_tasks=['forearm', 'elbow', 'cls']):
    device = config['device']
    
    # Configure model
    num_inputs = 6 if active_inputs in ['acc', 'gyr'] else 12
    model = MultiTaskTCN(num_inputs=num_inputs)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    huber = MaskedHuberLoss(delta=1.0)
    cce = torch.nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    train_loader = config['train_loader']
    val_loader = config['val_loader']
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        
        for X, y_f_sc, y_f_deg, y_e_deg, mask_e, y_c in train_loader:
            if active_inputs == 'acc':
                # indices 0:3 and 6:9 are Acc in [B, 240, 12] -> wait:
                # 0,1,2 = RUA Acc. 3,4,5 = RUA Gyr. 6,7,8 = RLA Acc. 9,10,11 = RLA Gyr.
                X = X[:, :, [0,1,2, 6,7,8]]
            elif active_inputs == 'gyr':
                X = X[:, :, [3,4,5, 9,10,11]]
                
            X, y_f_sc, y_e_deg, mask_e, y_c = X.to(device), y_f_sc.to(device), y_e_deg.to(device), mask_e.to(device), y_c.to(device)
            
            optimizer.zero_grad()
            p_f, p_e, p_c = model(X)
            
            loss = 0.0
            if 'forearm' in active_tasks:
                loss += F.mse_loss(p_f, y_f_sc)
            if 'elbow' in active_tasks:
                loss += huber(p_e, y_e_deg, mask_e)
            if 'cls' in active_tasks:
                loss += cce(p_c, y_c)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y_f_sc, y_f_deg, y_e_deg, mask_e, y_c in val_loader:
                if active_inputs == 'acc':
                    X = X[:, :, [0,1,2, 6,7,8]]
                elif active_inputs == 'gyr':
                    X = X[:, :, [3,4,5, 9,10,11]]
                    
                X, y_f_sc, y_e_deg, mask_e, y_c = X.to(device), y_f_sc.to(device), y_e_deg.to(device), mask_e.to(device), y_c.to(device)
                p_f, p_e, p_c = model(X)
                
                v_loss = 0.0
                if 'forearm' in active_tasks:
                    v_loss += F.mse_loss(p_f, y_f_sc)
                if 'elbow' in active_tasks:
                    v_loss += huber(p_e, y_e_deg, mask_e)
                if 'cls' in active_tasks:
                    v_loss += cce(p_c, y_c)
                    
                val_loss += v_loss.item()
                
        N_val = len(val_loader)
        v_loss_avg = val_loss/N_val
        scheduler.step(v_loss_avg)
        
        if v_loss_avg < best_val_loss:
            best_val_loss = v_loss_avg
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config['patience']:
            break
            
    model.load_state_dict(best_state)
    
    # Evaluate
    model.eval()
    test_loader = config['test_loader']
    # To use evaluate_model_full we need to wrap the dataset to only yield specific inputs
    # Let's just manually patch evaluate_model_full temporarily for this specific run
    # Or write a small eval loop
    # Actually, simpler: I'll just evaluate on Val set to get the metrics for ablations.
    # User said "Do NOT use the final test set to optimize the ablation configurations."
    return best_val_loss

def run_ablations():
    config = {
        'seed': 42,
        'batch_size': 64,
        'lr': 1e-3,
        'epochs': 150, # Fast early stopping for ablations
        'patience': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    set_seed(config['seed'])
    
    npz_path = r"D:\me\project\data\processed\windows\dataset.npz"
    config['train_loader'] = DataLoader(IMUDataset(npz_path, 'train'), batch_size=config['batch_size'], shuffle=True)
    config['val_loader'] = DataLoader(IMUDataset(npz_path, 'val'), batch_size=config['batch_size'], shuffle=False)
    config['test_loader'] = DataLoader(IMUDataset(npz_path, 'test'), batch_size=config['batch_size'], shuffle=False)
    
    ablations = {
        'A_Acc_Only': {'inputs': 'acc', 'tasks': ['forearm', 'elbow', 'cls']},
        'B_Gyr_Only': {'inputs': 'gyr', 'tasks': ['forearm', 'elbow', 'cls']},
        'D_Elbow_Only': {'inputs': 'all', 'tasks': ['elbow']},
        'E_Forearm_Only': {'inputs': 'all', 'tasks': ['forearm']},
        'F_MultiTask_Reg': {'inputs': 'all', 'tasks': ['forearm', 'elbow']}
    }
    
    results = {}
    for name, args in ablations.items():
        print(f"Running Ablation: {name}")
        val_loss = train_ablation(name, config, args['inputs'], args['tasks'])
        results[name] = val_loss
        print(f"{name} Val Loss: {val_loss:.4f}")
        
    with open("results/ablation_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_ablations()
