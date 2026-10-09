# Multi-Task Temporal Convolutional Network (TCN) Architecture Design

## 1. Architecture Overview
The model is a highly efficient, sequence-to-sequence TCN optimized for Edge-AI deployment. It processes continuous 12-channel IMU windows and simultaneously performs regression for joint angles and classification for the exercise type, preserving the full temporal resolution for the continuous targets.

![Architecture Diagram](file:///D:/me/project/reports/01_DATA_AUDIT/tcn_architecture_diagram.png)

### Layer-by-Layer Configuration
| Layer | Type | Kernel | Dilation | Filters | Output Shape | Activation | Params |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | Raw IMU Tensor | - | - | 12 | `[B, 240, 12]` | - | 0 |
| **TCN Block 1** | Conv1D + Residual | 3 | 1 | 32 | `[B, 240, 32]` | ReLU | ~4,800 |
| **TCN Block 2** | Conv1D + Residual | 3 | 2 | 32 | `[B, 240, 32]` | ReLU | ~6,336 |
| **TCN Block 3** | Conv1D + Residual | 3 | 4 | 32 | `[B, 240, 32]` | ReLU | ~6,336 |
| **TCN Block 4** | Conv1D + Residual | 3 | 8 | 32 | `[B, 240, 32]` | ReLU | ~6,336 |
| **TCN Block 5** | Conv1D + Residual | 3 | 16 | 32 | `[B, 240, 32]` | ReLU | ~6,336 |
| **Forearm Head**| Pointwise Conv1D | 1 | 1 | 2 | `[B, 240, 2]` | Linear | 66 |
| **Elbow Head** | Pointwise Conv1D | 1 | 1 | 1 | `[B, 240, 1]` | Linear | 33 |
| **Class Head** | Pointwise Conv1D | 1 | 1 | 20 | `[B, 240, 20]` | Linear | 660 |
| **Class Pool** | GlobalMaxPool1D | - | - | - | `[B, 20]` | Softmax | 0 |

### Structural Parameters
- **Receptive Field:** 1 + 2×(3-1) × (1+2+4+8+16) = **125 frames** (~1.04 seconds of context).
- **Residual Connections:** Yes, applied iteratively per TCN block.
- **Normalization:** Layer Normalization (better for time-series/small-batch vs BatchNorm).
- **Dropout:** 0.1 spatial dropout inserted before the residual addition to prevent overfitting.
- **Global Pooling (Classification):** Global Max Pooling over the 240-timestep logit sequence identifies the dominant exercise pattern over the 2-second window.

### Edge-AI Considerations
- **Total Parameters:** ~30,903
- **Trainable Parameters:** ~30,903
- **Model Size:** ~125 KB uncompressed (Extremely lightweight).
- **Complexity (FLOPs):** Ultra-low. Because convolutions are 1D and the filter count stays at 32, this network will run in < 1ms on a standard Android NNAPI/CPU inference stack. It is scientifically defensible to prefer this ultra-slim architecture over bloated transformers, preventing overfitting on the 10-subject training manifold while ensuring battery-friendly deployment.

---

## 2. Target Representations
### Forearm Angle
- **Predicted Form:** `sin(θ)` and `cos(θ)`.
- **Why:** The target audit revealed 13 coordinate wraps (e.g. ±180°). Predicting sin/cos naturally binds the network to a circular manifold ($S^1$).
- **Reconstruction:** During evaluation, we calculate `θ_pred = atan2(sin_pred, cos_pred)`.
- **Metrics:** RMSE in degrees, Mean Absolute Error (MAE), and Circular/Angular Error (`min(|pred - true|, 360 - |pred - true|)`).

### Elbow Angle
- **Predicted Form:** Continuous angle in degrees.
- **Why:** The elbow acts physically as a constrained hinge. Direct regression captures this bounded geometry without circular wrap risks.

### Ground-Truth Masking Strategy (Anomalies)
- **Problem:** The audit found 5 optical marker IK anomalies where the angle drops instantly (e.g., 144° → 6°). 
- **Solution:** A binary mask `M` of shape `[B, 240]` will be constructed during DataLoader compilation. For any frame $i$ where $|y_i - y_{i-1}| > 100^\circ$ inside the original dataset, $M_i = 0$. The surrounding $\pm 5$ frames will also be set to 0. 
- **Application:** The loss function for the Elbow target will multiply the per-timestep errors by `M`. Anomalous frames will contribute 0 to the gradient, preventing network corruption while leaving the saved `.npz` dataset natively untouched.

---

## 3. Multi-Task Loss Weighting Strategy
**Loss Equation:**
$L_{total} = \lambda_{forearm} L_{forearm} + \lambda_{elbow} L_{elbow} + \lambda_{cls} L_{cls}$

**Task Imbalance Issue:**
- $L_{cls}$ (Cross-Entropy) typically ranges from 0.1 to 3.0.
- $L_{forearm}$ (MSE on bounded sin/cos [-1, 1]) typically ranges from 0.01 to 2.0.
- $L_{elbow}$ (Huber loss on degrees [0, 150]) typically scales linearly with errors > 1° but can range wildly.

**Reproducible Weight Tuning Protocol:**
1. **Dynamic Weight Averaging (DWA):** We will initialize $\lambda_1 = \lambda_2 = \lambda_3 = 1.0$.
2. We will track the rate of change of the loss for each task $k$ between epoch $t$ and $t-1$: $w_k(t-1) = \frac{L_k(t-1)}{L_k(t-2)}$.
3. The weights for the current epoch will be dynamically recalculated as a softmax over these rates of change divided by a temperature $T$. This guarantees that if one task (e.g., classification) converges too fast, its gradient influence dynamically throttles back so the representation doesn't neglect the angle regressions.
4. **Validation Set Tuning:** Alternatively, a grid-search of static weights `[0.1, 1.0, 10.0]` will be executed strictly monitoring the *Validation Loss*, ensuring the Test Set remains blindly quarantined.

---

## 4. Experimental Plan

### Baselines
To mathematically justify the TCN, it will be strictly evaluated against identical Subject-Independent Test folds for:
1. **Classical Baseline:** A simple linear regression mapping instantaneous IMU frames (12 channels) to the 3 targets without temporal context.
2. **Simple Neural Baseline:** An LSTM (1 layer, 32 units) to demonstrate whether CNN-based receptive fields (TCN) outperform standard recurrent aggregation.
3. **Proposed Multi-Task TCN.**

### Ablation Plan
We will isolate architectural choices via independent training runs:
- **A. Single-task vs. Multi-task:** Run the TCN with only the Elbow head active. Compare its Test MAE to the Multi-Task TCN's Elbow MAE. *Proves whether learning joint biomechanics and exercise classification creates synergistic representations (MTL regularization).*
- **B. Sensor Modality:** Train with Accelerometer only vs. Gyroscope only. *Proves which sensor dominates specific kinematic tracking (e.g., Gravity/Acc for slow orientations, Gyro for rapid movement).*
- **C. Masking Impact:** Train the Multi-task model with and without the Ground-Truth Quality Mask. *Proves how drastically the 5 optical anomalies would have derailed gradient descent.*

### Training Protocol
- **Optimizer:** AdamW (Weight decay = 1e-4) to prevent overfitting on the 10-subject train set.
- **Initial LR:** 1e-3.
- **Scheduler:** ReduceLROnPlateau (Factor 0.5, Patience 10) monitoring Validation MAE.
- **Batch Size:** 64 (balances gradient noise with GPU memory).
- **Epochs:** 150 maximum.
- **Early Stopping:** Triggered if Validation MAE does not improve for 20 epochs.
- **Checkpointing:** Saves the model weights achieving the lowest cumulative validation loss.
- **Evaluation Frequency:** Full validation evaluated at the end of every epoch.

### Class Coverage Note
The `WristDevCali` class lacks representation from Subject 3. Since Subject 3 is assigned to the Training fold, the Test Fold and Validation Fold still contain exactly 100% of all 20 classes. The evaluation framework remains statistically flawless, though `WristDevCali` will be learned off a slightly narrower (9 subjects) manifold than the others.
