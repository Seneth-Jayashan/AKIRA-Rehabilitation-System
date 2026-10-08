import os
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib

def load_data(data_dir="datasets/ml_ready"):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()
    
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()
    
    return X_train, y_train, X_test, y_test

def evaluate_model(model, name, X_train, y_train, X_test, y_test):
    print(f"\n{'='*40}")
    print(f"Evaluating Model: {name}")
    print(f"{'='*40}")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    
    print(f"Training Time  : {train_time:.2f} seconds")
    print(f"Inference Time : {inference_time:.4f} seconds (Total)")
    print(f"Accuracy       : {acc*100:.2f}%")
    print(f"Macro F1-Score : {f1*100:.2f}%")
    print(f"Precision      : {precision*100:.2f}%")
    print(f"Recall         : {recall*100:.2f}%")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, acc, f1

def run_track_a():
    """Track A: Feature-Based Models"""
    X_train, y_train, X_test, y_test = load_data()
    
    # Handle NaNs from FFT features (if any division by zero occurred)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    models = {
        "Random Forest (Baseline)": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM (Linear)": SVC(kernel='linear', random_state=42)
    }
    
    best_f1 = 0
    best_model = None
    best_name = ""
    
    for name, model in models.items():
        trained_model, acc, f1 = evaluate_model(model, name, X_train, y_train, X_test, y_test)
        if f1 > best_f1:
            best_f1 = f1
            best_model = trained_model
            best_name = name
            
    print(f"\n🏆 Best Track A Model: {best_name} with Macro-F1: {best_f1*100:.2f}%")
    
    # Save the best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, f"models/track_a_best_{best_name.replace(' ', '_').lower()}.pkl")

if __name__ == "__main__":
    run_track_a()
