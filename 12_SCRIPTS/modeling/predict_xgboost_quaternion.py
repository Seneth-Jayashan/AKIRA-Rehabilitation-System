import os
import ast
import zipfile
import joblib
import numpy as np

def run_prediction():
    model_path = '09_MODELS/xgboost_quaternion_classifier.joblib'
    if not os.path.exists(model_path):
        print("Model not found!")
        return

    model_data = joblib.load(model_path)
    model = model_data['model']
    le = model_data['label_encoder']
    features = model_data['features']

    # Read a sequence from the test set participants (e.g. Participant9)
    zip_path = '01_DATASETS/RAW/3DCDataset-main.zip'
    sequence_id = '3DCDataset-main/Participant9/train/IMU/3dc_IMU_gesture_3_9.txt'
    
    parsed_quats = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(sequence_id) as f:
            content = f.read().decode('utf-8').strip().split('\n')
            for line in content:
                if not line: continue
                try:
                    q = ast.literal_eval(line)
                    parsed_quats.append(q)
                except:
                    pass
                    
    q_arr = np.array(parsed_quats)
    means = np.mean(q_arr, axis=0)
    stds = np.std(q_arr, axis=0)
    mins = np.min(q_arr, axis=0)
    maxs = np.max(q_arr, axis=0)
    ranges = maxs - mins
    
    feat_dict = {}
    for i, comp in enumerate(['W', 'X', 'Y', 'Z']):
        feat_dict[f'q_{comp}_mean'] = means[i]
        feat_dict[f'q_{comp}_std'] = stds[i]
        feat_dict[f'q_{comp}_min'] = mins[i]
        feat_dict[f'q_{comp}_max'] = maxs[i]
        feat_dict[f'q_{comp}_range'] = ranges[i]
        
    X_pred = np.array([feat_dict[f] for f in features]).reshape(1, -1)
    
    pred_encoded = model.predict(X_pred)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    
    probs = model.predict_proba(X_pred)[0]
    prob_dict = {le.classes_[i]: p for i, p in enumerate(probs)}
    
    print(f"Testing on sequence: {sequence_id}")
    print(f"Predicted Class: {pred_label}")
    print("Probabilities:")
    for k, v in prob_dict.items():
        print(f"  {k}: {v*100:.2f}%")

if __name__ == '__main__':
    run_prediction()
