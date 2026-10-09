# Demo Application Test Report

## Startup Result
- **Result:** SUCCESS
- **Detail:** Uvicorn and FastAPI initialized perfectly on `127.0.0.1:8000`. `src.models.tcn` successfully loaded the PyTorch parameters.

## Model Integrity Result
- **Result:** SUCCESS
- **Detail:** Cryptographic integrity verified at boot. 
  - Expected SHA-256: `bddadfc3999db6f1dde2868ca5942e8e2be4cb0a17d9ef9eae16a58dedabf697`
  - Actual SHA-256: `bddadfc3999db6f1dde2868ca5942e8e2be4cb0a17d9ef9eae16a58dedabf697`
  - *Integrity Passed Flag:* `True`

## API Test Results
- `/api/health`: 200 OK. Returns `{"status": "ok", "integrity": true}`
- `/api/model-info`: 200 OK. Returned full hash data.
- `/api/predict`: 200 OK (when provided 240x12 CSV).
- `/api/predict-demo`: 200 OK. Returned multi-head kinematic outputs.

## Inference Test Result
- **Result:** SUCCESS
- **Detail:** Extracted valid JSON structure matching `PredictionResponse`. Correctly reconstructed the forearm angle using `atan2` inside `inference.py`. Did NOT invoke `.backward()` or `optimizer`. Purely executed `model.eval()` within `torch.no_grad()`.

## Invalid-Input Test
- **Result:** SUCCESS
- **Detail:** Uploaded a 1-row, 3-channel CSV (`bad.csv`). 
- **Response:** HTTP 400 - `Invalid input. Expected: 240 rows × 12 IMU channels. Received: 1 rows × 3 channels.`

## Files Modified/Created
- `demo/README.md`
- `demo/requirements.txt`
- `demo/backend/app.py`
- `demo/backend/inference.py`
- `demo/backend/schemas.py`
- `demo/frontend/index.html`
- `demo/frontend/style.css`
- `demo/frontend/app.js`
- `demo/demo_data/sample_input.csv`
- `demo/demo_data/sample_gt.csv`
- `demo/demo_data/sample_meta.txt`
- `demo/test_report.md`

## Research Freeze Confirmation
**CONFIRMED**: No official experimental logs, preprocessing logic, testing methodologies, test splits, or model weights (`models/final/best_tcn.pth`) were modified, mutated, or overwritten during the creation of this demonstration package.
