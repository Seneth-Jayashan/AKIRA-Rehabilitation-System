# AKIRA Wrist Movement Intelligence
## Random Forest Quaternion Classification Research Prototype

This is the interactive frontend for the AKIRA Random Forest model (Model 2). It allows users to run supervised classification inferences on quaternion sequences extracted from the `3DCDataset`.

### How to Run

You need two separate PowerShell terminals running simultaneously from the project root (`C:\Users\ROG STRIX\Desktop\Wrist_Component`).

#### 1. Start the Python Backend API
Open a PowerShell terminal and run:
```powershell
cd frontend-random-forest\backend
python -m uvicorn main:app --reload
```
*The backend runs on http://localhost:8000 and automatically loads the trained `.joblib` model.*

#### 2. Start the React Frontend
Open a **new** PowerShell terminal and run:
```powershell
cd frontend-random-forest
npm install
npm run dev
```
*The frontend runs on http://localhost:5174. Click the link in the terminal to open the dashboard.*

### Features
- **Overview Dashboard:** Reads live evaluation metrics straight from the pipeline's JSON reports.
- **Movement Prediction:** Allows selection of real `3DCDataset` quaternion sequences. Previews the `(W,X,Y,Z)` sequence in an interactive Recharts timeline.
- **Inference:** Sends the raw sequence to the FastAPI backend, which extracts temporal sequence statistics identically to the training pipeline, and returns the predicted label and class probabilities.
- **Strict Scientific Integrity:** Does not fabricate data. If the model or metrics are missing, the UI gracefully displays an offline/missing status.
