import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

def load_optimized_model():
    print("Loading optimized XGBoost model...")
    model_path = "models/optimized/best_xgboost.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Optimized model not found at {model_path}. Did you run hyperparameter tuning?")
    return joblib.load(model_path)

def load_dataset(data_dir="datasets/ml_ready"):
    print("Loading dataset for feature extraction analysis...")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()
    
    # Fill NAs identically to the training process
    X_train = X_train.fillna(0)
    
    return X_train, y_train

def generate_importance_charts():
    model = load_optimized_model()
    X_train, y_train = load_dataset()
    
    os.makedirs("results", exist_ok=True)
    
    # 1. Native XGBoost Feature Importance (Weight / Gain)
    print("Generating XGBoost native feature importance plot...")
    plt.figure(figsize=(12, 10))
    xgb.plot_importance(model, max_num_features=20, importance_type='gain', show_values=False)
    plt.title("XGBoost Native Feature Importance (Top 20 by Gain)")
    plt.tight_layout()
    plt.savefig("results/feature_importance_xgb.png")
    plt.close()
    
    # 2. SHAP (SHapley Additive exPlanations)
    print("Calculating SHAP values for global interpretability (this might take a moment)...")
    
    # SHAP requires the booster, not the sklearn wrapper, for TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # For speed on large datasets, we can compute SHAP on a background sample
    # Here we use 1000 random samples to keep it fast but representative
    X_sample = X_train.sample(n=min(1000, len(X_train)), random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    # For multiclass, shap_values is a list of arrays (one per class). 
    # The summary_plot handles this beautifully by stacking the bars per class.
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance across all Exercise Classes")
    plt.tight_layout()
    plt.savefig("results/feature_importance_shap.png")
    plt.close()
    
    print("\nFeature Importance analysis complete!")
    print("Charts saved to:")
    print("- results/feature_importance_xgb.png")
    print("- results/feature_importance_shap.png")

if __name__ == "__main__":
    generate_importance_charts()
