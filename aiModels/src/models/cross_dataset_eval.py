import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def create_out_of_distribution_dataset(X, random_seed=42):
    """
    Simulates a cross-dataset (e.g., PHYTMO vs SDALLE) by applying 
    calibration shifts, sensor noise, and scaling differences commonly 
    found when using a different physical IMU hardware.
    """
    np.random.seed(random_seed)
    X_ood = X.copy()
    
    # 1. Calibration Shift (IMU might be mounted slightly differently)
    for col in X_ood.columns:
        if 'mean' in col:
            # Shift means by up to 10%
            X_ood[col] += np.random.normal(0, X_ood[col].std() * 0.1, size=len(X_ood))
            
    # 2. Sensitivity Scaling (Different accelerometer range/sensitivity)
    # Accelerations might read 5% lower or higher uniformly
    scale_factor = np.random.uniform(0.90, 1.10)
    for col in X_ood.columns:
        if 'var' in col or 'std' in col or 'range' in col:
            X_ood[col] *= scale_factor
            
    # 3. Add White Noise
    noise = np.random.normal(0, 0.05, size=X_ood.shape)
    X_ood = X_ood + noise
    
    return X_ood

def run_cross_dataset_evaluation():
    print("=== Phase 7: Cross-Dataset Generalization & Transfer Learning ===")
    
    # 1. Load Original Data and Model
    X_test_path = "aiModels/datasets/ml_ready/X_test.csv"
    y_test_path = "aiModels/datasets/ml_ready/y_test.csv"
    model_path = "aiModels/models/optimized/best_xgboost.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}. Ensure Phase 5 is completed.")
        
    X_test = pd.read_csv(X_test_path).fillna(0)
    y_test_raw = pd.read_csv(y_test_path).values.ravel()
    
    # Encode labels mathematically
    le = LabelEncoder()
    y_test = le.fit_transform(y_test_raw)
    
    model = joblib.load(model_path)
    
    # 2. Baseline (In-Distribution) Evaluation
    y_pred_base = model.predict(X_test)
    base_acc = accuracy_score(y_test, y_pred_base)
    print(f"\n[Baseline] Original Test Accuracy (In-Distribution): {base_acc*100:.2f}%")
    
    # 3. Create OOD (Out-Of-Distribution) Dataset to simulate SDALLE/GAITEX
    print("\nSimulating cross-dataset domain shift (SDALLE hardware characteristics)...")
    X_ood = create_out_of_distribution_dataset(X_test)
    
    # 4. Zero-Shot Evaluation (Testing on OOD data without any re-training)
    y_pred_zero_shot = model.predict(X_ood)
    zero_shot_acc = accuracy_score(y_test, y_pred_zero_shot)
    print(f"[Zero-Shot] Accuracy on NEW Dataset (Out-of-Distribution): {zero_shot_acc*100:.2f}%")
    print(f"   -> Accuracy Drop: {(base_acc - zero_shot_acc)*100:.2f}% due to domain shift.")
    
    # 5. Few-Shot Transfer Learning
    # We take a very small fraction (10%) of the OOD data to "calibrate" or "fine-tune"
    print("\nApplying Few-Shot Transfer Learning...")
    X_ood_train, X_ood_test, y_ood_train, y_ood_test = train_test_split(
        X_ood, y_test, test_size=0.90, random_state=42, stratify=y_test
    )
    
    print(f"Fine-tuning on just {len(X_ood_train)} samples from the new dataset...")
    
    # XGBoost allows incremental training via the `xgb_model` parameter.
    # We use the existing model as the starting point.
    transfer_model = XGBClassifier(
        n_estimators=50, # add 50 new trees to adapt to the shift
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    # Fit with the old model as a base
    transfer_model.fit(X_ood_train, y_ood_train, xgb_model=model)
    
    # 6. Evaluation after Transfer Learning
    y_pred_transfer = transfer_model.predict(X_ood_test)
    transfer_acc = accuracy_score(y_ood_test, y_pred_transfer)
    
    print(f"[Few-Shot] Accuracy on NEW Dataset (After Transfer Learning): {transfer_acc*100:.2f}%")
    print(f"   -> Accuracy Recovered: {(transfer_acc - zero_shot_acc)*100:.2f}%!")
    
    # Save the transfer-learned model
    transfer_model_path = "aiModels/models/optimized/transfer_learned_xgboost.pkl"
    joblib.dump(transfer_model, transfer_model_path)
    print(f"\nTransfer-learned model saved to: {transfer_model_path}")

if __name__ == "__main__":
    run_cross_dataset_evaluation()
