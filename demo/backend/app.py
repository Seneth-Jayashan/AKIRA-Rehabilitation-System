from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import io

from demo.backend.inference import AkiraInference, verify_model_integrity
from demo.backend.schemas import PredictionResponse

app = FastAPI(title="AKIRA Local Inference Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

integrity_passed, exp_hash, act_hash = verify_model_integrity()
inference_engine = AkiraInference() if integrity_passed else None

@app.get("/api/health")
def health_check():
    return {"status": "ok", "integrity": integrity_passed}

@app.get("/api/model-info")
def model_info():
    return {
        "integrity_passed": integrity_passed,
        "expected_hash": exp_hash,
        "actual_hash": act_hash
    }

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_sequence(file: UploadFile = File(...)):
    if not integrity_passed:
        raise HTTPException(status_code=500, detail="Model integrity failed. Inference disabled.")
        
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid CSV format.")
        
    if df.shape != (240, 12):
        raise HTTPException(status_code=400, detail=f"Invalid input. Expected: 240 rows × 12 IMU channels. Received: {df.shape[0]} rows × {df.shape[1]} channels.")
        
    if df.isnull().values.any():
        raise HTTPException(status_code=400, detail="Invalid input. CSV contains missing values.")
        
    # Process
    try:
        x_np = df.values.astype(np.float32)
        res = inference_engine.predict(x_np)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/api/predict-demo", response_model=PredictionResponse)
def predict_demo(example_id: str = "error"):
    if not integrity_passed:
        raise HTTPException(status_code=500, detail="Model integrity failed. Inference disabled.")
        
    df = pd.read_csv(f"demo/demo_data/sample_{example_id}_input.csv")
    x_np = df.values.astype(np.float32)
    res = inference_engine.predict(x_np)
    
    # Attach GT
    df_gt = pd.read_csv(f"demo/demo_data/sample_{example_id}_gt.csv")
    res['gt_elbow'] = df_gt['Elbow_deg'].tolist()
    res['gt_forearm'] = df_gt['Forearm_deg'].tolist()
    
    with open(f"demo/demo_data/sample_{example_id}_meta.txt", "r") as f:
        meta = f.read()
        
    # parse GT class id
    import json
    with open("data/processed/class_mapping.json", 'r') as f:
        mapping = json.load(f)
    idx_to_class = {v: k for k, v in mapping.items()}
    c_id = int(meta.split("Class ID: ")[1].split("\n")[0])
    res['gt_exercise'] = idx_to_class[c_id]
    
    return res


demo_cache = {}

@app.get("/api/predict-stream", response_model=PredictionResponse)
def predict_stream(example_id: str, end_idx: int):
    if not integrity_passed:
        raise HTTPException(status_code=500, detail="Model integrity failed.")
        
    if example_id not in demo_cache:
        df_in = pd.read_csv(f"demo/demo_data/sample_{example_id}_input.csv")
        df_gt = pd.read_csv(f"demo/demo_data/sample_{example_id}_gt.csv")
        
        with open(f"demo/demo_data/sample_{example_id}_meta.txt", "r") as f:
            meta = f.read()
        import json
        with open("data/processed/class_mapping.json", 'r') as f:
            mapping = json.load(f)
        idx_to_class = {v: k for k, v in mapping.items()}
        c_id = int(meta.split("Class ID: ")[1].split("\n")[0])
        gt_exercise = idx_to_class[c_id]
        
        demo_cache[example_id] = {
            'in': df_in.values.astype(np.float32),
            'gt': df_gt,
            'ex': gt_exercise
        }
        
    cache = demo_cache[example_id]
    
    if end_idx < 240 or end_idx > len(cache['in']):
        raise HTTPException(status_code=400, detail="Invalid end_idx")
        
    x_np = cache['in'][end_idx-240:end_idx]
    res = inference_engine.predict(x_np)
    
    res['gt_elbow'] = cache['gt']['Elbow_deg'].iloc[end_idx-240:end_idx].tolist()
    res['gt_forearm'] = cache['gt']['Forearm_deg'].iloc[end_idx-240:end_idx].tolist()
    res['gt_exercise'] = cache['ex']
    
    return res

@app.get("/api/demo-length")
def get_demo_length(example_id: str):
    df_in = pd.read_csv(f"demo/demo_data/sample_{example_id}_input.csv")
    return {"length": len(df_in)}

app.mount("/", StaticFiles(directory="demo/frontend", html=True), name="frontend")

