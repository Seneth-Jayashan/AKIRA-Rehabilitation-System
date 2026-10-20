# AI-Based Elbow Rehabilitation Assistance System

An end-to-end artificial intelligence research and clinical evaluation system for post-olecranon fracture open reduction and internal fixation (ORIF) rehabilitation.

The system utilizes a 12-channel dual-IMU configuration:
- **Upper Arm IMU (Humerus segment):** 3-axis Accelerometer + 3-axis Gyroscope (`RUA_Ax`, `Ay`, `Az`, `Gx`, `Gy`, `Gz`)
- **Forearm IMU (Radius/Ulna segment):** 3-axis Accelerometer + 3-axis Gyroscope (`RLA_Ax`, `Ay`, `Az`, `Gx`, `Gy`, `Gz`)

---

## Supported Clinical Tasks

The system trains, benchmarks, and serves four neural sequence architectures (**TCN**, **1D-CNN**, **LSTM**, and **GRU**) across four tasks:

1. **Task 1: Continuous Elbow Flexion/Extension Angle Regression & ROM Analysis (deg)**
   - Continuous time-series joint angle estimation and Range of Motion calculation.
2. **Task 2: Continuous Forearm Pronation/Supination Angle Regression & ROM Analysis (deg)**
   - Continuous rotational kinematics and pronation/supination ROM estimation.
3. **Task 3: Upper-Limb Rehabilitation Exercise Classification (3-Class)**
   - Identification of prescribed exercises: Elbow Flexion-Extension (EFE), Extension of Arms Overhead (EAH), and Squeezing (SQZ).
4. **Task 4: Movement-Quality Classification (Binary)**
   - Detection of correct movement execution versus compensatory/pathological movement patterns.
5. **Three-Tier Range of Motion & Repetition Cycle Detection**
   - Window ROM, full-trial ROM, and repetition cycle-segmented peak-to-peak ROM.

---

## Project Structure

```text
d:/me/project/
├── demo/
│   ├── backend/
│   │   ├── app.py                     # FastAPI web backend service
│   │   └── engine.py                  # Multi-task AI inference engine
│   └── frontend/
│       ├── index.html                 # Interactive clinical testing dashboard
│       ├── styles.css                 # Clean CSS styling
│       └── app.js                     # Waveform visualizer & comparison engine
├── models/
│   └── checkpoints/                   # 16 trained PyTorch model checkpoints
├── data/
│   ├── processed/                     # Standardized datasets & split manifest
│   └── samples/                       # 8 genuine held-out evaluation sample CSVs
├── datasets/
│   └── raw/                           # 13 original raw datasets (strictly read-only)
├── experiments/
│   └── benchmark_summary.json         # Complete 4-model evaluation benchmark metrics
├── src/
│   ├── data/                          # Data curation, schemas, and loader utilities
│   ├── models/                        # Parameter-matched TCN, 1D-CNN, LSTM, GRU
│   ├── evaluation/                    # Kinematic error metrics & ROM segmentation
│   └── training/                      # Training and validation runner
├── scripts/
│   └── verify_system.py               # End-to-end system verification script
├── tests/
│   ├── test_dataset.py                # Dataset standardization & split tests
│   ├── test_models.py                 # Model architecture & checkpoint loading tests
│   └── test_demo.py                   # API endpoints & inference engine tests
├── docs/                              # Research reports & demo guides
├── run_demo.py                        # Single-command launcher for interactive demo
└── requirements.txt                   # Environment dependencies
```

---

## Quick Start

### 1. Launch the Interactive Demo

Start the demo server with one command:

```bash
python run_demo.py
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

### 2. Run Automated Tests

Execute the automated test suite:

```bash
python -m pytest tests/
```

### 3. Verify System Health

Run the complete system health and integrity check:

```bash
python scripts/verify_system.py
```

---

## Benchmark Highlights (Held-Out Test Subjects)

| Architecture | Elbow Flexion MAE (deg) | Forearm Pro/Sup MAE (deg) | Exercise Class Macro F1 | Quality Assessment Macro F1 | P50 Latency (ms) | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TCN** | **1.70°** | 2.50° | **97.0%** | **94.7%** | 3.65 ms | 120,449 |
| **1D-CNN** | 1.83° | 2.56° | 92.8% | 85.8% | **0.86 ms** | **87,489** |
| **LSTM** | 1.76° | **2.48°** | 94.6% | 93.6% | 2.21 ms | 105,281 |
| **GRU** | 1.74° | 2.52° | 93.2% | 92.4% | 1.98 ms | 104,449 |
