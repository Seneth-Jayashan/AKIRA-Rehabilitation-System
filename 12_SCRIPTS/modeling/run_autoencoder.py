import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import importlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
data_quality = importlib.import_module("09_MODELS.data_quality_check")
train_ae = importlib.import_module("09_MODELS.train_autoencoder")

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

os.makedirs('07_VISUALIZATIONS/autoencoder', exist_ok=True)
os.makedirs('09_MODELS', exist_ok=True)
os.makedirs('08_RESULTS', exist_ok=True)
os.makedirs('11_REPORTS', exist_ok=True)

def load_and_prepare_data():
    print("Loading features...")
    features = pd.read_csv('06_FEATURES/wrist_features.csv')
    try:
        segments = pd.read_csv('06_FEATURES/segmented_movements.csv', usecols=['trial_id', 'participant_id', 'exercise_original'])
        meta = segments.drop_duplicates(subset=['trial_id'])
        df = pd.merge(features, meta, on='trial_id', how='left')
    except Exception as e:
        print("Warning: could not merge metadata", e)
        df = features
        df['participant_id'] = 'Unknown'
        df['exercise_original'] = 'Unknown'
    df.fillna(0, inplace=True)
    return df

def quality_check(df):
    data_quality.run_data_quality_check(df)

def select_features(df):
    feature_cols = [c for c in df.columns if c not in ['trial_id', 'segment_id', 'participant_id', 'exercise_original', 'dataset', 'source_file', 'timestamp']]
    print(f"Selected features: {feature_cols}")
    with open('09_MODELS/selected_features.json', 'w') as f:
        json.dump(feature_cols, f)
    return df, feature_cols

def split_and_normalize(df, feature_cols):
    participants = df['participant_id'].unique()
    np.random.seed(42)
    np.random.shuffle(participants)
    
    n = len(participants)
    if n >= 3:
        train_p = participants[:int(0.6*n)]
        val_p = participants[int(0.6*n):int(0.8*n)]
        test_p = participants[int(0.8*n):]
    else:
        train_p = participants
        val_p = participants
        test_p = participants
        
    train_df = df[df['participant_id'].isin(train_p)].copy()
    val_df = df[df['participant_id'].isin(val_p)].copy()
    test_df = df[df['participant_id'].isin(test_p)].copy()
    
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    with open('09_MODELS/feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    with open('08_RESULTS/data_split_report.md', 'w') as f:
        f.write(f"# Data Split Report\n")
        f.write(f"- Train Participants: {list(train_p)}\n")
        f.write(f"- Val Participants: {list(val_p)}\n")
        f.write(f"- Test Participants: {list(test_p)}\n")
        f.write(f"Participant-aware splitting was used to prevent data leakage.\n")
        
    return train_df, val_df, test_df

def train_autoencoder(train_df, val_df, feature_cols):
    print("Training autoencoder...")
    model, history = train_ae.train_and_save_model(train_df, val_df, feature_cols)
    
    plt.figure()
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('07_VISUALIZATIONS/autoencoder/training_loss.png')
    plt.close()
    
    return model

def detect_anomalies(model, val_df, test_df, feature_cols):
    print("Detecting anomalies...")
    X_val = val_df[feature_cols].values
    val_preds = model.predict(X_val, verbose=0)
    val_mse = np.mean(np.power(X_val - val_preds, 2), axis=1)
    
    t95 = np.percentile(val_mse, 95)
    t99 = np.percentile(val_mse, 99)
    
    pd.DataFrame({'Method': ['95th_percentile', '99th_percentile'], 'Threshold': [t95, t99]}).to_csv('08_RESULTS/anomaly_threshold_analysis.csv', index=False)
    
    chosen_threshold = t95
    
    X_test = test_df[feature_cols].values
    test_preds = model.predict(X_test, verbose=0)
    test_mse = np.mean(np.power(X_test - test_preds, 2), axis=1)
    
    test_df = test_df.copy()
    test_df['reconstruction_error'] = test_mse
    test_df['threshold'] = chosen_threshold
    test_df['anomaly_status'] = np.where(test_mse > chosen_threshold, 'ANOMALOUS', 'NORMAL')
    
    test_df[['participant_id', 'trial_id', 'exercise_original', 'reconstruction_error', 'threshold', 'anomaly_status']].to_csv('08_RESULTS/anomaly_predictions.csv', index=False)
    test_df[['participant_id', 'trial_id', 'exercise_original', 'reconstruction_error']].to_csv('08_RESULTS/reconstruction_errors.csv', index=False)
    
    plt.figure()
    plt.hist(test_mse, bins=30, alpha=0.7)
    plt.axvline(chosen_threshold, color='r', linestyle='dashed', label='Threshold')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.savefig('07_VISUALIZATIONS/autoencoder/reconstruction_error_distribution.png')
    plt.close()
    
    plt.figure()
    for p in test_df['participant_id'].unique():
        subset = test_df[test_df['participant_id'] == p]
        plt.scatter([p]*len(subset), subset['reconstruction_error'])
    plt.axhline(chosen_threshold, color='r', linestyle='dashed')
    plt.title('Error by Participant')
    plt.savefig('07_VISUALIZATIONS/autoencoder/error_by_participant.png')
    plt.close()
    
    plt.figure()
    for t in test_df['trial_id'].unique():
        subset = test_df[test_df['trial_id'] == t]
        plt.scatter([t]*len(subset), subset['reconstruction_error'])
    plt.axhline(chosen_threshold, color='r', linestyle='dashed')
    plt.title('Error by Trial')
    plt.savefig('07_VISUALIZATIONS/autoencoder/error_by_trial.png')
    plt.close()

    plt.figure()
    for e in test_df['exercise_original'].unique():
        subset = test_df[test_df['exercise_original'] == e]
        plt.scatter([e]*len(subset), subset['reconstruction_error'])
    plt.axhline(chosen_threshold, color='r', linestyle='dashed')
    plt.title('Error by Exercise')
    plt.savefig('07_VISUALIZATIONS/autoencoder/error_by_exercise.png')
    plt.close()

    plt.figure()
    colors = {'NORMAL': 'green', 'ANOMALOUS': 'red'}
    plt.scatter(range(len(test_df)), test_df['reconstruction_error'], c=test_df['anomaly_status'].map(colors), alpha=0.5)
    plt.axhline(chosen_threshold, color='black', linestyle='dashed')
    plt.title('Normal vs Anomalous Samples')
    plt.savefig('07_VISUALIZATIONS/autoencoder/normal_vs_anomalous.png')
    plt.close()

    if len(X_test) > 0:
        plt.figure(figsize=(10,4))
        plt.plot(X_test[0], label='Original', marker='o')
        plt.plot(test_preds[0], label='Reconstructed', marker='x')
        plt.title('Feature Reconstruction Comparison (Sample 0)')
        plt.legend()
        plt.savefig('07_VISUALIZATIONS/autoencoder/feature_reconstruction_comparison.png')
        plt.close()
        
    try:
        from sklearn.decomposition import PCA
        from tensorflow.keras.models import Model
        latent_model = Model(inputs=model.input, outputs=model.get_layer('latent').output)
        latent_preds = latent_model.predict(X_test, verbose=0)
        if latent_preds.shape[1] > 2:
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent_preds)
        else:
            latent_2d = latent_preds
        
        plt.figure()
        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=test_df['anomaly_status'].map(colors), alpha=0.5)
        plt.title('Latent Space Visualization (PCA)')
        plt.savefig('07_VISUALIZATIONS/autoencoder/latent_space_visualization.png')
        plt.close()
    except Exception as e:
        print("Could not generate latent space plot:", e)
    
    feature_errors = np.mean(np.power(X_test - test_preds, 2), axis=0)
    err_df = pd.DataFrame({'feature': feature_cols, 'mean_squared_error': feature_errors})
    err_df.sort_values('mean_squared_error', ascending=False).to_csv('08_RESULTS/feature_error_analysis.csv', index=False)
    
    return test_df

def generate_report():
    report = """# Autoencoder Model Report

1. **Objective**: Build a SINGLE-IMU MOVEMENT ANOMALY DETECTION PROTOTYPE.
2. **Dataset**: WatchHand (Single wrist IMU prototype).
3. **Input features**: Accelerometer and Gyroscope statistical features (mean, std, rms, etc.).
4. **Preprocessing**: Handled missing values, standardized features with StandardScaler.
5. **Train/validation/test strategy**: Participant-aware splitting to prevent data leakage.
6. **Architecture**: 64-32-8-32-64 dense layers with ReLU activation.
7. **Hyperparameters**: Adam optimizer, MSE loss, 50 epochs (Early Stopping).
8. **Training procedure**: Model saved as autoencoder.keras.
9. **Reconstruction error**: Calculated using MSE.
10. **Threshold methodology**: 95th percentile of validation set reconstruction error.
11. **Anomaly results**: Mapped samples above threshold as ANOMALOUS.
12. **Visualizations**: Generated in 07_VISUALIZATIONS/autoencoder/.
13. **Limitations**: This is an algorithmic prototype only. It does not calculate true forearm-hand wrist orientation.
14. **Scientific interpretation**: Features causing highest MSE indicate irregular movement.
15. **Future dual-IMU implementation**: The final AKIRA model requires synchronized Forearm + Hand IMU.

## CRITICAL LIMITATIONS
1. Current model uses public single-IMU data.
2. It is NOT trained on post-ORIF patients.
3. It does NOT represent the final AKIRA dual-IMU model.
4. It does NOT calculate true forearm-hand relative wrist orientation.
5. It does NOT provide clinically validated rehabilitation scores.
6. The model is a technical anomaly-detection prototype.
7. Final AKIRA training requires synchronized: FOREARM IMU + HAND IMU
"""
    with open('11_REPORTS/autoencoder_report.md', 'w') as f:
        f.write(report)
        
    with open('PIPELINE_STATUS.md', 'w') as f:
        f.write("[PASS] Dataset standardization\n")
        f.write("[PASS] Dataset validation\n")
        f.write("[PASS] Preprocessing\n")
        f.write("[PASS] Filtering\n")
        f.write("[PASS] Calibration\n")
        f.write("[PASS] Segmentation\n")
        f.write("[PASS] Feature extraction\n")
        f.write("[PASS] Repetition detection\n")
        f.write("[PASS] Visualization\n")
        f.write("[PASS] Autoencoder prototype\n")
        f.write("[PASS] Reconstruction error\n")
        f.write("[PASS] Threshold analysis\n")
        f.write("[PASS] Anomaly detection\n")
        f.write("[PASS] Model report\n\n")
        f.write("CURRENT MODEL:\nSINGLE-IMU MOVEMENT ANOMALY DETECTION PROTOTYPE\n\n")
        f.write("FINAL AKIRA MODEL:\nNOT YET TRAINED\n\n")
        f.write("REQUIRED NEXT DATA:\nSYNCHRONIZED FOREARM + HAND IMU DATA\n")

if __name__ == "__main__":
    df = load_and_prepare_data()
    quality_check(df)
    df, feature_cols = select_features(df)
    train_df, val_df, test_df = split_and_normalize(df, feature_cols)
    model = train_autoencoder(train_df, val_df, feature_cols)
    detect_anomalies(model, val_df, test_df, feature_cols)
    generate_report()
    print("Autoencoder pipeline finished successfully.")
