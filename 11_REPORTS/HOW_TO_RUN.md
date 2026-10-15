# AKIRA Wrist Component: How to Run Guide

Welcome to the AKIRA Wrist Component pipeline! This guide explains exactly how to run the preprocessing and Autoencoder scripts, how to generate new predictions without retraining, and where to find your outputs.

> **CRITICAL REMINDER**: The current prototype is a **Single-IMU Movement Anomaly Detection Prototype**. The final dual-IMU AKIRA model requires synchronized Forearm + Hand IMU data and has **NOT YET BEEN TRAINED**.

---

## 1. Environment Setup

Before running any scripts, you need to ensure your environment is configured correctly.

1. **Open VS Code:** Open the `Wrist_Component` folder in VS Code.
2. **Open Terminal:** Open a PowerShell terminal by clicking `Terminal -> New Terminal` (or pressing `Ctrl+~`). Ensure the path points to the project root (e.g., `C:\Users\ROG STRIX\Desktop\Wrist_Compenent`).
3. **Verify Python Environment:** Run the following command:
   ```powershell
   python --version
   ```
   *Note: This project currently uses the global system Python installation (`pythoncore-3.14-64`). No virtual environment (`.venv`) is required as packages were installed globally.*
4. **Verify Dependencies:** Verify that TensorFlow is installed by running:
   ```powershell
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```
   If no error appears and a version number prints, you are ready to go!

---

## 2. Running the Pipeline Scripts

All scripts have been organized into `12_SCRIPTS/`. **You must always execute them from the project root.**

### A. Preprocessing Pipeline
**Purpose:** Loads raw standardized data, applies filtering, calibration, segmentation, feature extraction, and repetition analysis.
**Command:**
```powershell
python 12_SCRIPTS/preprocessing/run_pipeline.py
```
* **Inputs:** `02_STANDARDIZED_DATA/WatchHand_standard.csv`
* **Outputs:** Generates `06_FEATURES/wrist_features.csv`, repetition reports in `08_RESULTS/`, and visualizations in `07_VISUALIZATIONS/`.
* **Note:** This will overwrite the existing feature files.

### B. Autoencoder Training
**Purpose:** Selects features, scales them, splits data by participant, and trains the prototype Autoencoder.
**Command:**
```powershell
python 12_SCRIPTS/modeling/run_autoencoder.py
```
* **Inputs:** `06_FEATURES/wrist_features.csv`
* **Outputs:** Overwrites `09_MODELS/autoencoder.keras`, the feature scaler `.pkl`, the selected features `.json`, and generates new threshold analysis reports.
* **Warning:** Do not run this unless you explicitly want to retrain the model.

### C. Dataset Standardization Scripts
**Purpose:** Converts the original raw `.zip` and extracted datasets into the standardized CSV schema.
**Commands:**
```powershell
python 12_SCRIPTS/dataset_processing/process_watchhand.py
python 12_SCRIPTS/dataset_processing/process_3dc.py
python 12_SCRIPTS/dataset_processing/process_29666735.py
```
* **Inputs:** Data inside `01_DATASETS/EXTRACTED/`
* **Outputs:** CSV files in `02_STANDARDIZED_DATA/` and `03_REFERENCE_DATA/`.

---

## 3. Safe Prediction Method (No Retraining)

If you simply want to generate anomaly predictions from new data using the *existing* trained model, you can use the newly created safe prediction script.

**Command:**
```powershell
python 12_SCRIPTS/modeling/predict_existing.py
```

* **How it works:** This script loads the pre-trained `09_MODELS/autoencoder.keras`, applies the saved `feature_scaler.pkl`, and uses the exact `selected_features.json` configuration. It evaluates the time-series summary windows from `wrist_features.csv`.
* **Safety:** It **does not** retrain the model and **does not** fit a new scaler.
* **Output:** Saves results directly to `08_RESULTS/predictions_demo.csv`. 

---

## 4. Viewing the Outputs

All generated files can be opened directly within VS Code:

* **The Trained Model:** `09_MODELS/autoencoder.keras` (This is a binary file; it cannot be read as text, but the prediction script loads it.)
* **Configuration:** You can view the exact features used by clicking on `09_MODELS/selected_features.json`.
* **Tabular Results:** Click on `08_RESULTS/anomaly_predictions.csv` or `08_RESULTS/feature_error_analysis.csv`. VS Code has great built-in CSV viewers, or you can install the "Edit csv" extension for a spreadsheet view.
* **Visualizations:** Expand the `07_VISUALIZATIONS/autoencoder/` folder and click on any `.png` file (e.g., `normal_vs_anomalous.png`) to see the diagnostic charts.
* **Reports:** Read the detailed markdown reports like `11_REPORTS/autoencoder_report.md` by opening them and clicking the "Open Preview to the Side" button in the top right corner of VS Code.

---

## 5. Troubleshooting Common Issues

* **`ModuleNotFoundError: No module named 'tensorflow'`**
  * *Fix:* TensorFlow is missing from your active environment. Run `pip install tensorflow`.
* **`FileNotFoundError: No such file or directory: '06_FEATURES/wrist_features.csv'`**
  * *Fix:* You ran the autoencoder script before running the preprocessing pipeline. Run `python 12_SCRIPTS/preprocessing/run_pipeline.py` first.
* **Script crashes with path errors**
  * *Fix:* You are likely running the script from inside the `12_SCRIPTS` folder. Always run scripts from the project root (`C:\Users\ROG STRIX\Desktop\Wrist_Compenent`).
* **Long-running commands appear frozen**
  * *Fix:* If a pip installation or training step hangs indefinitely, click into the terminal and press `Enter` (sometimes Windows PowerShell pauses output). If it is truly frozen, press `Ctrl+C` to cancel and restart. Avoid using infinite PowerShell `while` loops for dependency checking.

---

## 6. Final Verification Checklist

- [ ] I am running commands from the project root in a PowerShell terminal.
- [ ] TensorFlow imports successfully (`python -c "import tensorflow"`).
- [ ] The existing model (`autoencoder.keras`) and scaler (`feature_scaler.pkl`) exist in `09_MODELS/`.
- [ ] I can run `python 12_SCRIPTS/modeling/predict_existing.py` and see output in `08_RESULTS/predictions_demo.csv`.
- [ ] I can open the visual plots in `07_VISUALIZATIONS/autoencoder/`.
- [ ] I understand that this is only a **SINGLE-IMU Prototype** and the final dual-IMU model is **NOT YET TRAINED**.
