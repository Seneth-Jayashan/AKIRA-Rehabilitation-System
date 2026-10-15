import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

def run_prediction():
    print("Loading model and preprocessing artifacts...")
    
    # 1. Load the pre-trained autoencoder
    model_path = '09_MODELS/autoencoder.keras'
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}. Run autoencoder training first.")
        return
    model = tf.keras.models.load_model(model_path)
    
    # 2. Load the exact feature list used during training
    features_path = '09_MODELS/selected_features.json'
    if not os.path.exists(features_path):
        print(f"Error: Could not find {features_path}.")
        return
    with open(features_path, 'r') as f:
        feature_cols = json.load(f)
        
    # 3. Load the fitted scaler
    scaler_path = '09_MODELS/feature_scaler.pkl'
    if not os.path.exists(scaler_path):
        print(f"Error: Could not find {scaler_path}.")
        return
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # 4. Load threshold
    threshold_path = '08_RESULTS/anomaly_threshold_analysis.csv'
    if not os.path.exists(threshold_path):
        print(f"Error: Could not find {threshold_path}.")
        return
    threshold_df = pd.read_csv(threshold_path)
    # Extract the 95th percentile threshold
    threshold = threshold_df[threshold_df['Method'] == '95th_percentile']['Threshold'].values[0]
        
    print("Artifacts loaded successfully.")
    
    # 5. Load data to predict on (using the existing features file for demo)
    data_path = '06_FEATURES/wrist_features.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Merge metadata if available for readable output
    try:
        segments = pd.read_csv('06_FEATURES/segmented_movements.csv', usecols=['trial_id', 'participant_id', 'exercise_original'])
        meta = segments.drop_duplicates(subset=['trial_id'])
        df = pd.merge(df, meta, on='trial_id', how='left')
    except Exception:
        df['participant_id'] = 'Unknown'
        df['exercise_original'] = 'Unknown'
        
    # Schema validation
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in input data: {missing_cols}")
        return
        
    # Handle missing values (fill with 0 as done in training)
    df.fillna(0, inplace=True)
    
    # 6. Preprocess using the loaded scaler (DO NOT FIT!)
    print("Scaling features...")
    X = df[feature_cols].copy()
    X_scaled = scaler.transform(X)
    
    # 7. Predict and calculate reconstruction error
    print("Generating predictions...")
    reconstructed = model.predict(X_scaled, verbose=1)
    mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
    
    # 8. Classify anomalies based on the pre-calculated threshold
    df['reconstruction_error'] = mse
    df['anomaly_status'] = np.where(mse > threshold, 'ANOMALOUS', 'NORMAL')
    
    # 9. Save results
    output_path = '08_RESULTS/predictions_demo.csv'
    output_cols = ['participant_id', 'trial_id', 'exercise_original', 'reconstruction_error', 'anomaly_status']
    # Include output cols if they exist in the dataframe
    output_cols = [c for c in output_cols if c in df.columns]
    
    df[output_cols].to_csv(output_path, index=False)
    print(f"Predictions successfully saved to {output_path}")
    print(f"Total records processed: {len(df)}")
    print(f"Anomalies detected: {len(df[df['anomaly_status'] == 'ANOMALOUS'])}")
    
if __name__ == '__main__':
    run_prediction()
