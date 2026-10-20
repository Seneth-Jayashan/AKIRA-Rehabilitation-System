# Stage 10.2 & Stage 10.3 — Interactive Model Comparison Dashboard & Sample Inference Preview Guide (`docs/STAGE10_SAMPLE_INFERENCE_PREVIEW.md`)

**Project:** AI-Based Elbow Rehabilitation Assistance System After Olecranon Fracture ORIF  
**Workspace Root:** `D:\me\project`  
**Web Demo URL:** `http://127.0.0.1:8000`  
**Date:** 2026-10-19

---

## 1. Stage 10.3 Simple Interactive Model Comparison Dashboard

The web demo at `http://127.0.0.1:8000` ([`src/inference/static/index.html`](file:///d:/me/project/src/inference/static/index.html), [`src/inference/static/styles.css`](file:///d:/me/project/src/inference/static/styles.css), [`src/inference/static/app.js`](file:///d:/me/project/src/inference/static/app.js), [`src/inference/engine.py`](file:///d:/me/project/src/inference/engine.py), [`src/inference/app.py`](file:///d:/me/project/src/inference/app.py)) now features a clean, beginner-friendly **Simple Task, Sample & Four-Model Comparison Dashboard** as the primary top-level view, while preserving all multi-sample batch evaluation and 14-track benchmark explorer features inside expandable research accordions below.

### 1.1 Streamlined 6-Part User Workflow

1. **1. Select Task (`#stage10-task-select`)** — Choose one of the five core rehabilitation tasks:
   - `elbow_flexion_angle`: **Elbow Flexion/Extension Angle (`deg`)**
   - `forearm_prosup_angle`: **Forearm Pronation/Supination Angle (`deg`)**
   - `rom_estimation`: **Range of Motion (ROM) Estimation (`deg`)**
   - `exercise_classification`: **Rehabilitation Exercise Classification**
   - `execution_quality`: **Exercise Execution Quality (Correct vs. Incorrect)**
2. **2. Select Dataset (`#stage10-dataset-select`)** & **3. Select Recorded Sample (`#stage10-sample-select`)**:
   - Automatically filters to datasets and held-out test recordings compatible with the selected task (with an explicit optgroup/option to test incompatible sample-task pairs against the Task Compatibility Guard).
3. **4. Select Model (`#stage10-model-select`)**:
   - Choose **`Compare All Four Models (TCN, 1D-CNN, LSTM, GRU)`** or focus on an individual architecture (**`TCN`**, **`1D-CNN`**, **`LSTM`**, or **`GRU`**).
4. **Process Data (`#btn-run-stage10-preview`)**:
   - One prominent **"Process Data"** button (plus an optional **"Simulate Missing Ground Truth (Unlabeled Recording Mode)"** checkbox `#stage10-unlabeled-toggle`).
5. **Display Results and Graphs (`#s103-input-signals-panel` & `#s103-task-graphs-panel`)**:
   - **Input Recording & Sensor Signals (`#s103-input-signals-panel`)**:
     - Six beginner-friendly summary cards (Selected Task, Dataset & Subject Split, Sample ID & Exercise, Sampling Rate & Duration, Input Channels & Units, Ground-Truth Reference) + technical preprocessing contract bar (`configs/stage10/`).
     - Two side-by-side sensor signal graphs (`#s103-canvas-acc` and `#s103-canvas-gyr`) showing Upper Arm & Forearm 3-axis Accelerometer (`m/s²`) and Gyroscope (`rad/s`) signals (or 8-channel unit quaternions for `VIDIMU`).
   - **Sample-Level Prediction vs. Overall Held-Out Test-Set Performance (`#s103-sample-vs-testset-box`)**:
     - Side-by-side comparison cards explicitly distinguishing **(A) Sample-Level Prediction on This Single Recording** from **(B) Overall Held-Out Test-Set Benchmark Performance** across all unseen test subjects.
   - **Task-Appropriate Prediction & Error Graphs (`#s103-canvas-task-primary` & `#s103-canvas-task-secondary`)**:
     - **Angle Regression (`elbow_flexion_angle`, `forearm_prosup_angle`)**: Continuous Predicted vs. Ground-Truth Joint Angle Trajectory (`deg`) over time + Instantaneous Prediction Error ($\hat{\theta}(t) - \theta_{\text{gt}}(t)$ in `deg`).
     - **ROM Estimation (`rom_estimation`)**: Continuous Elbow Angle Trajectory (`deg`) + Three-Tier ROM Comparison Bar Chart (`Tier 1: 2.0s Window ROM`, `Tier 2: Full-Trial ROM`, `Tier 3: Segmented Repetition ROM` in `deg`) + Segmented Repetition Cycle cards (`#stage10-rom-cycles-box`).
     - **Classification (`exercise_classification`, `execution_quality`)**: Softmax Class Probability Distribution (`%`) + 2.0s Sliding-Window Confidence & Accuracy Comparison (`%`).
6. **Four-Model Comparison at the Bottom (`#s103-four-model-bottom-section`)**:
   - **Comparison Table (`#stage10-preview-table`)**: Compares `TCN`, `1D-CNN`, `LSTM`, and `GRU` side by side on the exact same preprocessed input, showing:
     - Model Name (with `Stage 9 Pick` and `Selected` badges; clicking any row focuses that model)
     - Sample Prediction
     - Ground-Truth Value or Label (or `"Unlabeled Mode (Hidden)"` when missing)
     - Sample Error / Evaluation Metric (`MAE / RMSE / Pearson r`, `Matched Cycle ROM MAE`, or `Correct/Incorrect + Window Acc %`)
     - Overall Test-Set Benchmark (`Test MAE / R²`, `Test Window ROM MAE`, or `Test Macro-F1 / Acc`)
     - Live CPU Inference Time (`ms` total and `ms/window`)
   - **Two Four-Model Comparison Graphs (`#s103-canvas-four-model-compare` & `#s103-canvas-four-model-latency`)**:
     - Left chart compares all four models' predictions/errors against the overall test-set benchmark.
     - Right chart compares live CPU inference latency (`ms`) across `TCN`, `1D-CNN`, `LSTM`, and `GRU`.

---

## 2. REST API Endpoints (`src/inference/app.py`)

All pre-existing Stage 9.5, Stage 9.6, and Stage 10.2 endpoints remain 100% preserved.

### 2.1 `GET /api/stage10/configs`
Returns the versioned JSON configurations from [`configs/stage10/preprocessing_configs.json`](file:///d:/me/project/configs/stage10/preprocessing_configs.json) and [`configs/stage10/task_candidates.json`](file:///d:/me/project/configs/stage10/task_candidates.json).

### 2.2 `GET /api/stage10/tasks`
Returns the 5 core rehabilitation tasks along with their primary/auxiliary tracks, reusable checkpoints for `TCN`, `1D_CNN`, `LSTM`, and `GRU`, and compatible held-out samples (`compatible_samples`).

### 2.3 `POST /api/stage10/preview`
Request body:
```json
{
  "sample_id": "optotrak_s15_t11_elbow_flex_ext",
  "task_id": "elbow_flexion_angle",
  "track_id": null,
  "selected_model": "COMPARE_ALL",
  "unlabeled_override": false
}
```
Response payload (when `compatible == true`):
- `status`: `"ok"`, `compatible`: `true`
- `prediction_origin`: `"actual_saved_checkpoint"`, `is_placeholder`: `false`
- `selected_model`, `focused_model`, `compare_all_models`, `unlabeled_override`, `has_ground_truth`
- `sample_vs_testset_note` & `test_set_benchmark_context`: Explicit separation of single-sample results from full held-out test-set generalization.
- `recording_metadata`: Full sample, channel, unit, gravity-convention, and versioned preprocessing contract.
- `signal_preview`: Downsampled dual-IMU sensor signals for browser plotting.
- `side_by_side_summary`: 4-element array comparing `TCN`, `1D_CNN`, `LSTM`, and `GRU` with both sample-level and overall test-set benchmark metrics.
- `models`: Full per-model predictions, `three_tier_rom`, and `trajectory_plot` / `class_probabilities`.

---

## 3. Remaining Scientific & Dataset Limitations

1. **Task-Dataset Compatibility Boundaries:** Not all public datasets contain all target annotations (e.g., `PHYTMO` and `JU-IMU` provide exercise/quality/cohort labels without continuous optical joint angles, whereas `Optotrak vs DOT` and `ULTra-MoCap` provide optical joint angles without `Correct vs. Incorrect` execution quality labels). The Task Compatibility Guard blocks incompatible combinations rather than fabricating outputs.
2. **Gravity & Sensor Coordinate Conventions:** `Optotrak vs DOT` (`120 Hz`) records free acceleration (gravity subtracted onboard), whereas `PHYTMO` (`100 Hz`), `ULTra-MoCap` (`100 Hz`), and `JU-IMU` (`80 Hz`) record proper acceleration (gravity included). Versioned dataset-specific preprocessing pipelines (`configs/stage10/preprocessing_configs.json`) preserve exact checkpoint validity.
3. **Forearm Pronation/Supination Observability:** Accurate (`8.48°–10.19° MAE`) during structured exercises with the elbow flexed near `90°`, but subject to unobservable heading drift (`18°–30° MAE`) when the forearm aligns with gravity during complex 3D reaching.
4. **Prospective Clinical Validation Pending:** All checkpoints are trained and evaluated on public datasets of healthy volunteers or neurological cohorts; prospective clinical validation on post-olecranon ORIF surgical patients requires future clinical data collection.
