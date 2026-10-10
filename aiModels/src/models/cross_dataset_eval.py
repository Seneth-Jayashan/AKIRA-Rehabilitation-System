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
    
    # ==========================================
    # 7. GENUINE Cross-Dataset Evaluation (PHYTMO -> SDALLE)
    # ==========================================
    print("\n=== GENUINE CROSS-DATASET EVALUATION (PHYTMO -> SDALLE) ===")
    sdalle_path = "aiModels/datasets/processed/sdalle_features.csv"
    if os.path.exists(sdalle_path):
        sdalle_df = pd.read_csv(sdalle_path).fillna(0)
        
        # In SDALLE, some activity names might differ slightly, but we map them to our taxonomy
        # PHYTMO labels: 'Sit', 'Stand', 'Walk', 'Turn' (or similar based on earlier taxonomy)
        # We need the true labels from SDALLE. Let's see what they are in sdalle_df['activity'].
        if 'activity' in sdalle_df.columns:
            # We must drop metadata cols just like in train_test split
            meta_cols = ["dataset_name", "subject_id", "session_id", "trial_id", "activity", "sensor_id", "timestamp"]
            X_genuine = sdalle_df.drop(columns=[c for c in meta_cols if c in sdalle_df.columns], errors='ignore')
            
            # Align columns to X_test (in case of missing/extra features)
            for col in X_test.columns:
                if col not in X_genuine.columns:
                    X_genuine[col] = 0.0
            X_genuine = X_genuine[X_test.columns] # Enforce order
            
            y_genuine_raw = sdalle_df['activity'].astype(str).str.lower()
            
            # The model was trained on PHYTMO labels. We map SDALLE labels to match.
            # E.g. "walking" -> "walk", "jogging" -> "run", "stairs_up" -> "stairs_up" (etc)
            # Since we don't know the exact PHYTMO classes the model expects, we can transform using `le`
            # But we must only evaluate on classes the model knows about.
            # Create lower-case to original-case mapping for known classes
            known_classes_lower = {str(c).lower(): c for c in le.classes_}
            
            # Map SDALLE labels to PHYTMO taxonomy where possible
            mapped_y = []
            valid_indices = []
            
            for idx, label in enumerate(y_genuine_raw):
                label_clean = label.strip()
                mapped = None
                
                if 'walk' in label_clean:
                    if 'walking' in known_classes_lower:
                        mapped = known_classes_lower['walking']
                    elif 'walk' in known_classes_lower:
                        mapped = known_classes_lower['walk']
                
                if mapped:
                    mapped_y.append(mapped)
                    valid_indices.append(idx)
                    
            if len(valid_indices) > 0:
                X_genuine_valid = X_genuine.iloc[valid_indices]
                y_genuine_valid = le.transform(mapped_y)
                
                # Zero-Shot on Genuine Data
                y_pred_genuine = model.predict(X_genuine_valid)
                genuine_acc = accuracy_score(y_genuine_valid, y_pred_genuine)
                print(f"[Genuine Zero-Shot] Accuracy on SDALLE Dataset: {genuine_acc*100:.2f}%")
                
                # Few-Shot on Genuine Data
                # If we have enough data, let's fine-tune on 10%
                if len(X_genuine_valid) > 10:
                    X_g_train, X_g_test, y_g_train, y_g_test = train_test_split(
                        X_genuine_valid, y_genuine_valid, test_size=0.90, random_state=42, stratify=y_genuine_valid
                    )
                    
                    if len(np.unique(y_g_train)) > 1:
                        genuine_transfer_model = XGBClassifier(n_estimators=50, learning_rate=0.05, max_depth=4, random_state=42)
                        genuine_transfer_model.fit(X_g_train, y_g_train, xgb_model=model)
                        
                        y_pred_g_transfer = genuine_transfer_model.predict(X_g_test)
                        g_transfer_acc = accuracy_score(y_g_test, y_pred_g_transfer)
                        print(f"[Genuine Few-Shot] Accuracy on SDALLE (After Transfer Learning): {g_transfer_acc*100:.2f}%")
                    else:
                        print("[Genuine Few-Shot] Skipping transfer learning because the genuine mapped dataset only contains 1 unique class.")
            else:
                print("Could not map any SDALLE activities to the PHYTMO taxonomy for evaluation.")
    else:
        print("SDALLE features not found. Skipping Genuine Cross-Dataset test.")

if __name__ == "__main__":
    run_cross_dataset_evaluation()
