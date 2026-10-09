# Structure Change Log

## Folders Created
- `reports/01_DATA_AUDIT` (and subfolders)
- `reports/02_MODEL_DEVELOPMENT` (and subfolders)
- `reports/03_FINAL_EVALUATION` (and subfolders)
- `reports/04_FINAL_PACKAGING` (and subfolders)
- `reports/05_VISUAL_RESULTS` (and subfolders)

## Files Moved
- All contents of `reports/01_DATA_AUDIT/` mapped into `01_DATA_AUDIT` (quality, preprocessing, anomalies) except architecture diagrams which went to `02_MODEL_DEVELOPMENT` and `05_VISUAL_RESULTS`.
- All unstructured markdown reports in `reports/` mapped to their respective workflow directories (e.g. `baseline_comparison.md` -> `02_MODEL_DEVELOPMENT`).
- Final delivery artifacts in `reports/final_packaging/` mapped to `04_FINAL_PACKAGING`.
- All visual plots mapped into `05_VISUAL_RESULTS` categorization.

## Intentionally Left Unchanged
- `models/final/best_tcn.pth`
- `models/final/best_lstm.pth`
- `models/final/best_linear.pth`
- All dataset files (`demo/demo_data/` and `data/`)
- All numeric metrics and models logic (`demo/frontend/` and `demo/backend/`)
- The FastAPI localhost demo remains fully functional and untouched.
