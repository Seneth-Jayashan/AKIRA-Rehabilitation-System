import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_stability_regressor(data_dir="aiModels/datasets/ml_ready"):
    """
    Trains regression models to predict the clinical Stability Index (0-100) 
    using only IMU features (Accelerometer/Gyroscope) and compares them.
    """
    print("Loading IMU feature dataset...")
    X_train_path = os.path.join(data_dir, "X_train.csv")
    
    if not os.path.exists(X_train_path):
        raise FileNotFoundError(f"Cannot find {X_train_path}. Make sure to run from repository root.")
        
    X = pd.read_csv(X_train_path).fillna(0)
    
    # Simulate a target "Stability Index" (0-100) based on variance/tremor features
    # If the user is shaking a lot (high az_var, gx_var), stability is lower.
    print("Mapping IMU signatures to Stability Indices...")
    tremor_proxy = X['az_var'] + X['gx_var']
    max_tremor = tremor_proxy.quantile(0.95)
    y = 100 - (tremor_proxy / max_tremor * 100)
    y = np.clip(y, 0, 100)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "XGBoost Regressor": XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "Ridge (Linear) Regression": Ridge(alpha=1.0)
    }
    
    best_model = None
    best_r2 = -float('inf')
    best_name = ""
    
    print("\n--- Model Comparison for Stability Regression ---")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name
            
    print(f"\nBest Model Selected: {best_name} with R² = {best_r2:.4f}")
    
    # Save the best model
    os.makedirs("aiModels/models/optimized", exist_ok=True)
    model_path = f"aiModels/models/optimized/best_stability_regressor.pkl"
    joblib.dump(best_model, model_path)
    print(f"\nBest Regressor saved to: {model_path}")

if __name__ == "__main__":
    train_stability_regressor(data_dir="aiModels/datasets/ml_ready")
