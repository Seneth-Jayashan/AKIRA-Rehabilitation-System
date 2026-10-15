import os
import json
import numpy as np
import tensorflow as tf
import zipfile
import ast
import sys

def predict_cnn():
    model_path = '09_MODELS/1d_cnn_quaternion_classifier.keras'
    config_path = '09_MODELS/1d_cnn_quaternion_config.json'
    
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print("Error: 1D CNN model or config not found.")
        sys.exit(1)
        
    model = tf.keras.models.load_model(model_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # Pick a random sequence to test
    zip_path = '01_DATASETS/RAW/3DCDataset-main.zip'
    with zipfile.ZipFile(zip_path, 'r') as z:
        imu_files = [f for f in z.namelist() if 'IMU' in f and f.endswith('.txt')]
        test_file = imu_files[-1] # Pick the last one for inference test
        print(f"Testing inference on: {test_file}")
        
        with z.open(test_file) as file:
            content = file.read().decode('utf-8').strip().split('\n')
            
    parsed_quats = []
    for line in content:
        if not line: continue
        try:
            q = ast.literal_eval(line)
            parsed_quats.append([q[0], q[1], q[2], q[3]])
        except:
            pass
            
    q_arr = np.array(parsed_quats)
    seq_len = len(q_arr)
    max_len = config['max_time_steps']
    
    if seq_len >= max_len:
        q_arr = q_arr[:max_len, :]
    else:
        pad = np.zeros((max_len - seq_len, 4))
        q_arr = np.vstack([q_arr, pad])
        
    X_input = np.expand_dims(q_arr, axis=0) # shape (1, 250, 4)
    
    probs = model.predict(X_input, verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_label = config['gesture_map'][str(pred_idx)]
    
    print("\n--- Inference Result ---")
    print(f"Input Shape: {X_input.shape}")
    print(f"Predicted Class: {pred_label}")
    print("Probabilities:")
    for i, p in enumerate(probs):
        print(f"  {config['gesture_map'][str(i)]}: {p*100:.2f}%")

if __name__ == '__main__':
    predict_cnn()
