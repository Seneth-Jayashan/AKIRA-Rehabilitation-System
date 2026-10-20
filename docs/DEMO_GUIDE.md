# AI Elbow Rehabilitation Interactive Demo Guide

## Overview

The **AI Elbow Rehabilitation Assistant** is an interactive web demonstration designed to evaluate and compare deep learning sequence models (**TCN**, **1D-CNN**, **LSTM**, **GRU**) on standardized upper-limb inertial measurement unit (IMU) data.

The system evaluates post-olecranon fracture ORIF recovery through four clinical tasks:
1. **Task 1: Continuous Elbow Flexion/Extension Angle Regression & ROM Analysis (deg)**
2. **Task 2: Continuous Forearm Pronation/Supination Angle Regression & ROM Analysis (deg)**
3. **Task 3: Upper-Limb Rehabilitation Exercise Classification (3-Class)**
4. **Task 4: Movement-Quality Classification (Binary: Correct vs. Incorrect/Compensatory)**
5. **Three-Tier Range of Motion (ROM) & Repetition Cycle Detection**

---

## Directory Architecture

```
demo/
├── backend/
│   ├── app.py             # FastAPI backend server
│   └── engine.py          # Multi-task AI inference engine (loads PyTorch models)
└── frontend/
    ├── index.html         # Interactive web interface
    ├── styles.css         # Clinical UI stylesheet
    └── app.js             # Client logic, Chart.js visualizer, comparison table builder
```

---

## Starting the Demo

Run the root entrypoint:

```bash
python run_demo.py
```

Or start via Uvicorn directly:

```bash
uvicorn demo.backend.app:app --host 127.0.0.1 --port 8000
```

Once running, navigate to:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Key Features

1. **Model Selection**: Choose any individual model (`TCN`, `1D-CNN`, `LSTM`, `GRU`) or select **Compare All Four Models** to execute all architectures side by side under identical inputs.
2. **Preset Held-Out Evaluation Samples**: Select from 8 genuine held-out subject recordings (correct exercises, compensatory movements, multi-speed trials, and unlabeled raw sensor signals).
3. **Custom CSV Upload**: Upload any user-recorded dual-IMU trial matching the 12 canonical sensor channels (`RUA_Ax..z`, `RUA_Gx..z`, `RLA_Ax..z`, `RLA_Gx..z`).
4. **Interactive Waveform Visualizations**: Inspect raw 12-channel accelerations and angular velocities alongside reconstructed joint angle trajectories and ground truth.
5. **Side-by-Side Model Comparison Table**: Compare live sample errors, overall test MAE/RMSE/F1, parameter counts, and inference latency across all four architectures.
