# AKIRA Frontend Merge Audit & Integration Plan

## 1. Existing Frontends Analysis

### Frontend 1: `frontend-demo`
- **Purpose:** Original Autoencoder anomaly-detection UI mockup.
- **Structure:** Modularized (components, layouts, pages, static JSON data).
- **Key Features to Preserve:**
  - Sidebar and Topbar navigation layout with rich CSS blur/glow effects.
  - Scatter plots showing MSE against thresholds (`AnomalyDetection.jsx`).
  - Recorded-data playback/Live Monitor UI mockups.
  - Feature Explorer/Data Explorer UI elements.
  - Metric cards and System Status dashboard.
- **Limitations:** Uses static `demo_data.json` mock files instead of live Python backend APIs. Arbitrary threshold hardcoding.

### Frontend 2: `frontend-random-forest`
- **Purpose:** Multi-model classification dashboard (Models 2, 3, 4).
- **Structure:** Monolithic `App.jsx` talking to FastAPI backend on `:8000`.
- **Key Features to Preserve:**
  - Live Axios fetching of sequences from `3DCDataset-main.zip`.
  - True inference of RF, 1D CNN, and XGBoost via POST `/api/predict`.
  - Live Autoencoder inference via POST `/api/predict-anomaly` utilizing proper dynamically loaded 95th-percentile thresholds (no 0-fill).
  - Side-by-side model comparison cards ("Compare All").
  - Model Performance bar chart fetching actual `metrics.json`.

## 2. Integration Plan

1. **Base Framework:** Use `frontend-random-forest` as the root workspace to retain active Vite config, Axios wiring, and live FastApi backend dependencies.
2. **Modularization:** Break `frontend-random-forest/src/App.jsx` into multiple modular components mimicking the `frontend-demo` directory structure.
3. **Design System Merge:** Port the animated background blurs, Lucide-react icons, Topbar/Sidebar layout, and responsive typography from `frontend-demo` into `frontend-random-forest`.
4. **Navigation Sections:**
   - **Overview:** Combine RF/CNN/XGB/AE system status.
   - **Autoencoder Anomaly Detection:** Port the live API call from `frontend-random-forest` into the UI of `frontend-demo`'s `AnomalyDetection` tab, adding scatter charts of historical predictions.
   - **Movement Classification & Compare Classifiers:** Enhance the live prediction tab with playback and selection tools.
   - **Sensor Data Explorer:** Re-wire `DataExplorer.jsx` to fetch live datasets from the `/api/sequence` or `/api/ae-datasets` endpoints.
   - **Model Performance:** Retain the live Recharts bar chart comparing RF/CNN/XGB.
   - **Reports/About:** Expand disclaimer and clinical limitations.
5. **Model Strictness:** 
   - No models will be retrained.
   - Threshold `0.0128194` is securely loaded via the FastAPI backend.
   - Preprocessing rules strictly preserved via backend API routes.
6. **Backend:** The `frontend-random-forest/backend/main.py` is already consolidated and fully supports all four models with distinct pipelines and 404 handlers. No python server changes needed.
