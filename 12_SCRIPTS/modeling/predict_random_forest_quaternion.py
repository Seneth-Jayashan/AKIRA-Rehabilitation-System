import os
import json
import pandas as pd
import joblib
import sys

def predict():
    model_path = '09_MODELS/random_forest_quaternion_classifier.joblib'
    features_path = '09_MODELS/random_forest_quaternion_features.json'
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        print("Error: Model or feature configuration not found. Run training script first.")
        sys.exit(1)
        
    rf = joblib.load(model_path)
    with open(features_path, 'r') as f:
        required_features = json.load(f)
        
    input_file = '06_FEATURES/random_forest_quaternion_features.csv'
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)
        
    df = pd.read_csv(input_file)
    
    # Validate columns
    missing = [col for col in required_features if col not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        sys.exit(1)
        
    X = df[required_features].values
    
    predictions = rf.predict(X)
    
    df_out = pd.DataFrame({
        'recording_id': df['recording_id'] if 'recording_id' in df.columns else df.index,
        'participant_id': df['participant_id'] if 'participant_id' in df.columns else 'Unknown',
        'predicted_exercise': predictions
    })
    
    out_path = '08_RESULTS/random_forest_quaternion_inference_results.csv'
    df_out.to_csv(out_path, index=False)
    print(f"Predictions successfully saved to {out_path}")

if __name__ == '__main__':
    predict()
