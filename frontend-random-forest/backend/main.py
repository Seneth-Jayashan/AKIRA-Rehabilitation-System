from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
import os
import zipfile
import ast
import tensorflow as tf
import pickle

app = FastAPI(title="AKIRA Classifiers & Autoencoder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODELS = {
    "rf": {"model": None, "features": None, "metrics": None},
    "cnn": {"model": None, "config": None, "metrics": None},
    "xgb": {"model": None, "label_encoder": None, "features": None, "metrics": None},
    "ae": {"model": None, "scaler": None, "features": None, "threshold": None}
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ZIP_PATH = os.path.join(BASE_DIR, '01_DATASETS', 'RAW', '3DCDataset-main.zip')

@app.on_event("startup")
def load_artifacts():
    # Load RF
    rf_model_path = os.path.join(BASE_DIR, '09_MODELS', 'random_forest_quaternion_classifier.joblib')
    rf_features_path = os.path.join(BASE_DIR, '09_MODELS', 'random_forest_quaternion_features.json')
    rf_metrics_path = os.path.join(BASE_DIR, '08_RESULTS', 'random_forest_quaternion_metrics.json')
    if os.path.exists(rf_model_path) and os.path.exists(rf_features_path):
        MODELS["rf"]["model"] = joblib.load(rf_model_path)
        with open(rf_features_path, 'r') as f: MODELS["rf"]["features"] = json.load(f)
    if os.path.exists(rf_metrics_path):
        with open(rf_metrics_path, 'r') as f: MODELS["rf"]["metrics"] = json.load(f)

    # Load CNN
    cnn_model_path = os.path.join(BASE_DIR, '09_MODELS', '1d_cnn_quaternion_classifier.keras')
    cnn_config_path = os.path.join(BASE_DIR, '09_MODELS', '1d_cnn_quaternion_config.json')
    cnn_metrics_path = os.path.join(BASE_DIR, '08_RESULTS', '1d_cnn_quaternion_metrics.json')
    if os.path.exists(cnn_model_path) and os.path.exists(cnn_config_path):
        MODELS["cnn"]["model"] = tf.keras.models.load_model(cnn_model_path)
        with open(cnn_config_path, 'r') as f: MODELS["cnn"]["config"] = json.load(f)
    if os.path.exists(cnn_metrics_path):
        with open(cnn_metrics_path, 'r') as f: MODELS["cnn"]["metrics"] = json.load(f)
        
    # Load XGBoost
    xgb_model_path = os.path.join(BASE_DIR, '09_MODELS', 'xgboost_quaternion_classifier.joblib')
    xgb_metrics_path = os.path.join(BASE_DIR, '08_RESULTS', 'xgboost_quaternion_metrics.json')
    if os.path.exists(xgb_model_path):
        xgb_data = joblib.load(xgb_model_path)
        MODELS["xgb"]["model"] = xgb_data['model']
        MODELS["xgb"]["label_encoder"] = xgb_data['label_encoder']
        MODELS["xgb"]["features"] = xgb_data['features']
    if os.path.exists(xgb_metrics_path):
        with open(xgb_metrics_path, 'r') as f: MODELS["xgb"]["metrics"] = json.load(f)

    # Load AE
    ae_model_path = os.path.join(BASE_DIR, '09_MODELS', 'autoencoder.keras')
    ae_scaler_path = os.path.join(BASE_DIR, '09_MODELS', 'feature_scaler.pkl')
    ae_features_path = os.path.join(BASE_DIR, '09_MODELS', 'selected_features.json')
    ae_threshold_path = os.path.join(BASE_DIR, '08_RESULTS', 'anomaly_threshold_analysis.csv')
    if os.path.exists(ae_model_path) and os.path.exists(ae_scaler_path) and os.path.exists(ae_features_path):
        MODELS["ae"]["model"] = tf.keras.models.load_model(ae_model_path)
        with open(ae_scaler_path, 'rb') as f: MODELS["ae"]["scaler"] = pickle.load(f)
        with open(ae_features_path, 'r') as f: MODELS["ae"]["features"] = json.load(f)
        if os.path.exists(ae_threshold_path):
            try:
                thresh_df = pd.read_csv(ae_threshold_path)
                row = thresh_df[thresh_df['Method'] == '95th_percentile']
                if not row.empty:
                    MODELS["ae"]["threshold"] = float(row.iloc[0]['Threshold'])
            except:
                pass

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "rf_loaded": MODELS["rf"]["model"] is not None,
        "cnn_loaded": MODELS["cnn"]["model"] is not None,
        "xgb_loaded": MODELS["xgb"]["model"] is not None,
        "ae_loaded": MODELS["ae"]["model"] is not None
    }

@app.get("/api/model-info")
def model_info():
    info = {}
    if MODELS["rf"]["model"]:
        info["rf"] = {"model_type": "Random Forest", "classes": list(MODELS["rf"]["model"].classes_), "metrics": MODELS["rf"]["metrics"]}
    if MODELS["cnn"]["model"]:
        info["cnn"] = {"model_type": "1D CNN", "classes": list(MODELS["cnn"]["config"]["gesture_map"].values()), "metrics": MODELS["cnn"]["metrics"]}
    if MODELS["xgb"]["model"]:
        info["xgb"] = {"model_type": "XGBoost", "classes": list(MODELS["xgb"]["label_encoder"].classes_), "metrics": MODELS["xgb"]["metrics"]}
    if MODELS["ae"]["model"]:
        info["ae"] = {"model_type": "Autoencoder", "threshold": MODELS["ae"]["threshold"]}
    return info

DEMO_SEQUENCES = []
@app.on_event("startup")
def load_demo_sequences():
    if os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            imu_files = [f for f in z.namelist() if 'IMU' in f and f.endswith('.txt') and 'Participant1/' in f]
            for f in imu_files[:15]:
                basename = f.split('/')[-1]
                DEMO_SEQUENCES.append({
                    "id": f,
                    "filename": basename,
                    "participant": "Participant1",
                    "description": f"Quaternion Sequence ({basename})"
                })

@app.get("/api/datasets")
def get_datasets():
    return DEMO_SEQUENCES

@app.get("/api/sequence/{sequence_id:path}")
def get_sequence(sequence_id: str):
    if not os.path.exists(ZIP_PATH):
        raise HTTPException(status_code=404, detail="Dataset zip not found")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            if sequence_id not in z.namelist():
                raise HTTPException(status_code=404, detail="Sequence not found in dataset zip")
            with z.open(sequence_id) as f:
                content = f.read().decode('utf-8').strip().split('\n')
                
        parsed_quats = []
        for line in content:
            if not line: continue
            try:
                q = ast.literal_eval(line)
                parsed_quats.append({"w": q[0], "x": q[1], "y": q[2], "z": q[3]})
            except:
                pass
        return {"id": sequence_id, "data": parsed_quats}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PredictRequest(BaseModel):
    sequence: list
    model: str = "all"  # "rf", "cnn", "xgb", or "all"

@app.post("/api/predict")
def predict(req: PredictRequest):
    if not req.sequence:
        raise HTTPException(status_code=400, detail="Empty sequence")
        
    results = {}
    q_arr = np.array([[s['w'], s['x'], s['y'], s['z']] for s in req.sequence])
    
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
    
    # RF Inference
    if req.model in ["rf", "all"] and MODELS["rf"]["model"]:
        try:
            X_rf = [feat_dict[f] for f in MODELS["rf"]["features"]]
            rf_model = MODELS["rf"]["model"]
            pred = rf_model.predict([X_rf])[0]
            proba = None
            if hasattr(rf_model, "predict_proba"):
                proba_arr = rf_model.predict_proba([X_rf])[0]
                proba = {str(c): float(p) for c, p in zip(rf_model.classes_, proba_arr)}
            results["rf"] = {"prediction": str(pred), "probabilities": proba}
        except Exception as e:
            results["rf"] = {"error": str(e)}

    # XGB Inference
    if req.model in ["xgb", "all"] and MODELS["xgb"]["model"]:
        try:
            X_xgb = [feat_dict[f] for f in MODELS["xgb"]["features"]]
            xgb_model = MODELS["xgb"]["model"]
            le = MODELS["xgb"]["label_encoder"]
            
            X_xgb_arr = np.array(X_xgb).reshape(1, -1)
            pred_encoded = xgb_model.predict(X_xgb_arr)[0]
            pred_label = le.inverse_transform([pred_encoded])[0]
            
            probs = xgb_model.predict_proba(X_xgb_arr)[0]
            proba = {str(le.classes_[i]): float(p) for i, p in enumerate(probs)}
            results["xgb"] = {"prediction": str(pred_label), "probabilities": proba}
        except Exception as e:
            results["xgb"] = {"error": str(e)}

    # CNN Inference
    if req.model in ["cnn", "all"] and MODELS["cnn"]["model"]:
        try:
            config = MODELS["cnn"]["config"]
            max_len = config['max_time_steps']
            seq_len = len(q_arr)
            if seq_len >= max_len:
                q_arr_cnn = q_arr[:max_len, :]
            else:
                pad = np.zeros((max_len - seq_len, 4))
                q_arr_cnn = np.vstack([q_arr, pad])
                
            X_cnn = np.expand_dims(q_arr_cnn, axis=0)
            cnn_model = MODELS["cnn"]["model"]
            
            probs = cnn_model.predict(X_cnn, verbose=0)[0]
            pred_idx = np.argmax(probs)
            pred_label = config['gesture_map'][str(pred_idx)]
            proba = {config['gesture_map'][str(i)]: float(p) for i, p in enumerate(probs)}
            results["cnn"] = {"prediction": pred_label, "probabilities": proba}
        except Exception as e:
            results["cnn"] = {"error": str(e)}

    if not results:
        raise HTTPException(status_code=500, detail="Requested models are not loaded.")

    return results

# Anomaly detection logic
AE_DEMO_DATA = []
@app.on_event("startup")
def load_ae_demo():
    csv_path = os.path.join(BASE_DIR, '06_FEATURES', 'wrist_features.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path).head(15).replace({float('nan'): None})
        for _, row in df.iterrows():
            AE_DEMO_DATA.append(row.to_dict())

@app.get("/api/ae-datasets")
def get_ae_datasets():
    return [{"id": str(i), "description": f"Sample {i} - {d.get('activity_type', 'Unknown')}", "data": d} for i, d in enumerate(AE_DEMO_DATA)]

class AnomalyRequest(BaseModel):
    features: dict

@app.post("/api/predict-anomaly")
def predict_anomaly(req: AnomalyRequest):
    if not MODELS["ae"]["model"]:
        raise HTTPException(status_code=500, detail="Autoencoder not loaded")
    
    feats = MODELS["ae"]["features"]
    input_df = pd.DataFrame([req.features])
    
    missing = [f for f in feats if f not in input_df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")
            
    X = input_df[feats]
    X_scaled = MODELS["ae"]["scaler"].transform(X)
    
    reconstruction = MODELS["ae"]["model"].predict(X_scaled, verbose=0)
    mse = np.mean(np.power(X_scaled - reconstruction, 2), axis=1)[0]
    
    thresh = MODELS["ae"]["threshold"]
    
    result = {"mse_score": float(mse)}
    if thresh is not None:
        result["is_anomaly"] = bool(mse > thresh)
        result["threshold_used"] = thresh
    else:
        result["is_anomaly"] = None
        result["warning"] = "Threshold unavailable, returning raw MSE only."
    
    return result

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
