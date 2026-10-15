import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

def generate_rich_demo_data():
    print("Loading model and artifacts...")
    model = tf.keras.models.load_model('09_MODELS/autoencoder.keras')
    with open('09_MODELS/selected_features.json', 'r') as f:
        feature_cols = json.load(f)
    with open('09_MODELS/feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    threshold_df = pd.read_csv('08_RESULTS/anomaly_threshold_analysis.csv')
    threshold = threshold_df[threshold_df['Method'] == '95th_percentile']['Threshold'].values[0]
    
    # We will pick 50 continuous samples from the features file
    df = pd.read_csv('06_FEATURES/wrist_features.csv')
    try:
        segments = pd.read_csv('06_FEATURES/segmented_movements.csv', usecols=['trial_id', 'participant_id', 'exercise_original'])
        meta = segments.drop_duplicates(subset=['trial_id'])
        df = pd.merge(df, meta, on='trial_id', how='left')
    except Exception:
        df['participant_id'] = 'Unknown'
        df['exercise_original'] = 'Unknown'
        
    df.fillna(0, inplace=True)
    
    # Make sure we grab some anomalous ones too
    # First, let's predict everything to find anomalies
    X_all = scaler.transform(df[feature_cols])
    preds_all = model.predict(X_all, verbose=0)
    mse_all = np.mean(np.power(X_all - preds_all, 2), axis=1)
    df['reconstruction_error'] = mse_all
    df['anomaly_status'] = np.where(mse_all > threshold, 'ANOMALOUS', 'NORMAL')
    
    # Grab 25 normal and 25 anomalous
    normal_df = df[df['anomaly_status'] == 'NORMAL'].head(25)
    anom_df = df[df['anomaly_status'] == 'ANOMALOUS'].head(25)
    demo_df = pd.concat([normal_df, anom_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    X_demo = scaler.transform(demo_df[feature_cols])
    preds_demo = model.predict(X_demo, verbose=0)
    
    rich_predictions = []
    for i in range(len(demo_df)):
        row = demo_df.iloc[i]
        # Calculate feature-level squared errors
        f_errors = np.power(X_demo[i] - preds_demo[i], 2)
        
        # Zip features with errors, sort descending
        f_err_list = [{'feature': f, 'error': float(e), 'input': float(X_demo[i][j]), 'reconstructed': float(preds_demo[i][j])} 
                      for j, (f, e) in enumerate(zip(feature_cols, f_errors))]
        f_err_list = sorted(f_err_list, key=lambda x: x['error'], reverse=True)
        
        rich_predictions.append({
            'trial_id': str(row.get('trial_id', f'Trial-{i}')),
            'participant_id': str(row.get('participant_id', 'Unknown')),
            'exercise_original': str(row.get('exercise_original', 'Unknown')),
            'reconstruction_error': float(row['reconstruction_error']),
            'threshold': float(threshold),
            'anomaly_status': str(row['anomaly_status']),
            'top_feature_errors': f_err_list[:10]  # Store top 10 contributing features
        })
        
    # Get timeseries for the FIRST trial in the list to display in the TS chart
    ts_data = []
    if len(rich_predictions) > 0:
        first_trial = rich_predictions[0]['trial_id']
        ts_iter = pd.read_csv('02_STANDARDIZED_DATA/WatchHand_standard.csv', chunksize=50000)
        for chunk in ts_iter:
            match = chunk[chunk['trial_id'] == first_trial]
            if not match.empty:
                # Take 150 points max for smooth playback animation
                ts_data = match.head(150).fillna('').to_dict(orient='records')
                break

    data = {
        'threshold': float(threshold),
        'predictions': rich_predictions,
        'timeseries': ts_data
    }
    
    with open('frontend-demo/src/data/demo_data.json', 'w') as f:
        json.dump(data, f)
        
    print(f"Generated rich demo data with {len(rich_predictions)} predictions and {len(ts_data)} timeseries rows.")

if __name__ == '__main__':
    generate_rich_demo_data()
