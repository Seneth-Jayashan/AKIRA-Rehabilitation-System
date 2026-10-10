import os
import pandas as pd
import numpy as np
import time
import joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def run_audit():
    print("=== Phase 4 & 5 Audit: Model Validation ===")
    
    data_dir = "aiModels/datasets/ml_ready"
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv")).fillna(0)
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()
    meta_test = pd.read_csv(os.path.join(data_dir, "meta_test.csv"))
    
    model_path = "aiModels/models/optimized/best_xgboost.pkl"
    le_path = "aiModels/models/label_encoder.pkl"
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}. Run hyperparameter tuning first.")
        return
        
    model = joblib.load(model_path)
    le = joblib.load(le_path)
    
    y_test_encoded = le.transform(y_test)
    class_names = le.classes_
    
    # 1. Inference Latency and Model Size
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\nModel Size: {model_size_mb:.2f} MB")
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    avg_latency = (inference_time / len(X_test)) * 1000 # in ms
    print(f"Inference Latency: {avg_latency:.4f} ms per sample")
    
    # 2. Overall Metrics
    acc = accuracy_score(y_test_encoded, y_pred)
    balanced_acc = balanced_accuracy_score(y_test_encoded, y_pred)
    
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test_encoded, y_pred, average='macro')
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test_encoded, y_pred, average='weighted')
    
    print("\n--- Overall Performance ---")
    print(f"Accuracy:          {acc*100:.2f}%")
    print(f"Balanced Accuracy: {balanced_acc*100:.2f}%")
    print(f"Macro-F1:          {f1_macro*100:.2f}%")
    print(f"Weighted-F1:       {f1_weighted*100:.2f}%")
    
    # 3. Per-class metrics
    print("\n--- Per-Class Precision & Recall ---")
    prec_none, rec_none, _, _ = precision_recall_fscore_support(y_test_encoded, y_pred, average=None)
    for i, cls in enumerate(class_names):
        print(f"{cls:20s}: Precision = {prec_none[i]*100:.2f}%, Recall = {rec_none[i]*100:.2f}%")
        
    print("\nDetailed Classification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=class_names))
    
    # 4. Subject-level Variance
    print("\n--- Subject-Level Variance ---")
    meta_test["unique_subject_id"] = meta_test["dataset_name"] + "_" + meta_test["subject_id"].astype(str)
    subjects = meta_test["unique_subject_id"].unique()
    
    subject_accs = []
    subject_f1s = []
    for subj in subjects:
        idx = meta_test.index[meta_test["unique_subject_id"] == subj].tolist()
        if len(idx) > 0:
            subj_y_test = y_test_encoded[idx]
            subj_y_pred = y_pred[idx]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                subject_accs.append(accuracy_score(subj_y_test, subj_y_pred))
                _, _, f1, _ = precision_recall_fscore_support(subj_y_test, subj_y_pred, average='macro')
                subject_f1s.append(f1)
                
    mean_acc = np.mean(subject_accs)
    std_acc = np.std(subject_accs)
    mean_f1 = np.mean(subject_f1s)
    std_f1 = np.std(subject_f1s)
    
    print(f"Mean Subject Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"Mean Subject Macro-F1: {mean_f1*100:.2f}% ± {std_f1*100:.2f}%")
    
    # 5. Confusion Matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - XGBoost (Audited)')
    plt.tight_layout()
    
    out_dir = "aiModels/documentation/audit_results"
    os.makedirs(out_dir, exist_ok=True)
    cm_path = os.path.join(out_dir, "confusion_matrix_audited.png")
    plt.savefig(cm_path)
    plt.close()
    
    print(f"\nAudit complete. Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    run_audit()
