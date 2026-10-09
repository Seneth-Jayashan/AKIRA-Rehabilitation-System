import torch
import hashlib
import json
import numpy as np
from src.models.tcn import MultiTaskTCN
from src.training.metrics import reconstruct_angle_degrees
from scipy.signal import find_peaks

MODEL_PATH = "models/final/best_tcn.pth"
META_PATH = "reports/04_FINAL_PACKAGING/final_model_record.json"
CLASS_MAPPING_PATH = "data/processed/class_mapping.json"

def verify_model_integrity():
    with open(META_PATH, 'r') as f:
        meta = json.load(f)
    expected_hash = meta['sha256_hash']
    
    h = hashlib.sha256()
    with open(MODEL_PATH, 'rb') as f:
        h.update(f.read())
    actual_hash = h.hexdigest()
    
    return expected_hash == actual_hash, expected_hash, actual_hash

class AkiraInference:
    def __init__(self):
        self.device = 'cpu'
        self.model = MultiTaskTCN()
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()
        
        with open(CLASS_MAPPING_PATH, 'r') as f:
            mapping = json.load(f)
        self.idx_to_class = {v: k for k, v in mapping.items()}

    def predict(self, x_numpy):
        # x_numpy shape: (240, 12)
        X = torch.tensor(x_numpy, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            p_f, p_e, p_c = self.model(X)
            
            p_f = p_f.squeeze(0).cpu() # [240, 2]
            p_e = p_e.squeeze(0).squeeze(-1).cpu() # [240]
            p_c = p_c.squeeze(0).cpu() # [20]
            
        # Reconstruct forearm
        f_deg = reconstruct_angle_degrees(p_f[:, 0], p_f[:, 1]).numpy()
        e_deg = p_e.numpy()
        
        probs = torch.softmax(p_c, dim=0).numpy()
        pred_class_idx = np.argmax(probs)
        pred_class_name = self.idx_to_class[pred_class_idx]
        
        class_probabilities = {self.idx_to_class[i]: float(probs[i]) for i in range(20)}
        
        # Calculate rehab metrics (model-derived demonstration)
        e_rom = float(np.max(e_deg) - np.min(e_deg))
        f_rom = float(np.max(f_deg) - np.min(f_deg))
        
        peaks, _ = find_peaks(e_deg, prominence=max(10, e_rom*0.15))
        reps = len(peaks)
        
        dt = 1.0 / 120.0
        velocity = float(np.abs(np.gradient(e_deg, dt)).mean())
        
        return {
            'elbow_prediction': e_deg.tolist(),
            'forearm_prediction': f_deg.tolist(),
            'exercise_prediction': pred_class_name,
            'class_probabilities': class_probabilities,
            'rom_elbow': e_rom,
            'rom_forearm': f_rom,
            'repetition_count': reps,
            'velocity_elbow': velocity
        }
