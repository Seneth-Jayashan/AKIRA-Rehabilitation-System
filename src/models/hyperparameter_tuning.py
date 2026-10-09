import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score

def load_data(data_dir="datasets/ml_ready"):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()
    
    # Fill NAs
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    return X_train, y_train, X_test, y_test

def tune_random_forest(X_train, y_train, X_test, y_test):
    print("\n" + "="*40)
    print("Tuning Model: Random Forest")
    print("="*40)
    
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    # Note: class_weight='balanced' explicitly forces the model to prioritize minority classes!
    rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1)
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    random_search = RandomizedSearchCV(
        rf, 
        param_distributions=param_dist, 
        n_iter=15, 
        scoring='f1_macro', 
        cv=cv, 
        verbose=1, 
        random_state=42, 
        n_jobs=-1
    )
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"\nBest Parameters: {random_search.best_params_}")
    print(f"Best CV Macro-F1: {random_search.best_score_:.4f}")
    
    best_rf = random_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Test Accuracy  : {acc*100:.2f}%")
    print(f"Test Macro-F1  : {f1*100:.2f}%")
    print(f"Tuning Time    : {end_time - start_time:.2f} seconds")
    
    return best_rf, acc, f1, random_search.best_params_

def tune_xgboost(X_train, y_train_encoded, X_test, y_test_encoded, class_names):
    print("\n" + "="*40)
    print("Tuning Model: XGBoost")
    print("="*40)
    
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    # Compute sample weights to heavily penalize misclassification on minority classes
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_encoded)
    
    xgb = XGBClassifier(random_state=42, n_jobs=1, objective='multi:softmax')
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    random_search = RandomizedSearchCV(
        xgb, 
        param_distributions=param_dist, 
        n_iter=15, 
        scoring='f1_macro', 
        cv=cv, 
        verbose=1, 
        random_state=42, 
        n_jobs=-1
    )
    
    # Fit with sample_weight explicitly passed to force balance
    start_time = time.time()
    random_search.fit(X_train, y_train_encoded, sample_weight=sample_weights)
    end_time = time.time()
    
    print(f"\nBest Parameters: {random_search.best_params_}")
    print(f"Best CV Macro-F1: {random_search.best_score_:.4f}")
    
    best_xgb = random_search.best_estimator_
    y_pred = best_xgb.predict(X_test)
    
    acc = accuracy_score(y_test_encoded, y_pred)
    f1 = f1_score(y_test_encoded, y_pred, average='macro')
    
    print(f"Test Accuracy  : {acc*100:.2f}%")
    print(f"Test Macro-F1  : {f1*100:.2f}%")
    print(f"Tuning Time    : {end_time - start_time:.2f} seconds")
    
    print("\nXGBoost Detailed Classification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=class_names))
    
    return best_xgb, acc, f1, random_search.best_params_

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    
    # Encode labels for XGBoost
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    class_names = le.classes_
    
    best_rf, rf_acc, rf_f1, rf_params = tune_random_forest(X_train, y_train, X_test, y_test)
    best_xgb, xgb_acc, xgb_f1, xgb_params = tune_xgboost(X_train, y_train_encoded, X_test, y_test_encoded, class_names)
    
    # Save the optimized models
    os.makedirs("models/optimized", exist_ok=True)
    joblib.dump(best_rf, "models/optimized/best_random_forest.pkl")
    joblib.dump(best_xgb, "models/optimized/best_xgboost.pkl")
    joblib.dump(le, "models/optimized/label_encoder.pkl")
    
    print("\nOptimized models saved to models/optimized/ directory.")
