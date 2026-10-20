# Stage 10.1 — Model Architecture & Checkpoint Inventory (`docs/STAGE10_MODEL_CHECKPOINT_INVENTORY.md`)

**Project:** AI-Based Elbow Rehabilitation Assistance System After Olecranon Fracture ORIF  
**Workspace Root:** `D:\me\project`  
**Checkpoints Directory:** [`D:\me\project\models\checkpoints`](file:///d:/me/project/models/checkpoints) (`64` `.pt` files across `16` benchmark track configurations $\times$ `4` architectures)  
**Architecture Module:** [`src/models/architectures.py`](file:///d:/me/project/src/models/architectures.py)  
**Audit Date:** 2026-10-19

---

## 1. Architectural Specifications of the Four Core Models

All four architectures are defined in [`src/models/architectures.py`](file:///d:/me/project/src/models/architectures.py) via `build_four_models(in_channels, task_mode, num_classes, num_reg_targets)` with matched parameter budgets (`~85k–116k` trainable parameters) and a shared output module (`TaskOutputHead`):

| Model Name | Class in `architectures.py` | Backbone Architecture & Hyperparameters | Temporal Context Mechanism | Output Head (`TaskOutputHead`) | Parameter Count (`12-ch` / `8-ch`) | Checkpoint Size (`KB`) | Single-Core / 16-Thread CPU Latency (`ms/win`) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **`TCN`** | [`TCNModel`](file:///d:/me/project/src/models/architectures.py#L89-L115) | 4 stacked residual `TemporalBlock` layers; `hidden_channels=(56, 56, 56, 56)`, `kernel_size=5`, dilations $d \in \{1, 2, 4, 8\}$, `BatchNorm1d + GELU + Dropout(0.2)`, $1\times 1$ residual skip conv | Centered dilated 1D convolutions (`padding = (k-1)*d // 2`), effective receptive field $= 121$ frames per block stack + global window pooling | **Classification:** Concat(`mean(dim=1)`, `amax(dim=1)`) $\rightarrow$ `Linear(112, C)` (`(B, C)` logits)<br>**Regression:** Frame-wise `Linear(56, 2)` (`(B, W, 2)` angles) | `115,306 – 116,774`<br>(`113,962` on 8-ch) | `472.6 – 483.0 KB` | **`2.22 ms`** (1-core) / `2.22 ms` (16-thread) |
| **`1D_CNN`** | [`CNN1DModel`](file:///d:/me/project/src/models/architectures.py#L118-L152) | 4 sequential non-dilated `Conv1d` blocks; `hidden_channels=(64, 64, 96, 96)`, `kernel_sizes=(7, 5, 5, 3)`, $d=1$, `BatchNorm1d + GELU + Dropout(0.2)` | Local non-dilated receptive field ($17$ frames) + global window pooling | **Classification:** Concat(`mean`, `amax`) $\rightarrow$ `Linear(192, C)`<br>**Regression:** Frame-wise `Linear(96, 2)` | `85,378 – 87,886`<br>(`83,586` on 8-ch) | `341.4 – 357.9 KB` | **`0.78 ms`** (1-core) / `0.78 ms` (16-thread) |
| **`LSTM`** | [`LSTMModel`](file:///d:/me/project/src/models/architectures.py#L155-L181) | `Linear(in_ch, 80) + GELU` $\rightarrow$ 2-layer unidirectional `nn.LSTM(80, 80, num_layers=2, batch_first=True, dropout=0.2)` $\rightarrow$ `LayerNorm(80)` | Recurrent cell state $c_t$ and hidden state $h_t$ ($t = 1 \dots W$) | **Classification:** Concat(`mean`, `amax`) $\rightarrow$ `Linear(160, C)`<br>**Regression:** Frame-wise `Linear(80, 2)` | `105,042 – 107,134`<br>(`104,722` on 8-ch) | `415.7 – 425.1 KB` | **`1.43 ms`** (1-core) / `2.00 ms` (16-thread) |
| **`GRU`** | [`GRUModel`](file:///d:/me/project/src/models/architectures.py#L184-L210) | `Linear(in_ch, 92) + GELU` $\rightarrow$ 2-layer unidirectional `nn.GRU(92, 92, num_layers=2, batch_first=True, dropout=0.2)` $\rightarrow$ `LayerNorm(92)` | Recurrent reset/update gated hidden state $h_t$ ($t = 1 \dots W$) | **Classification:** Concat(`mean`, `amax`) $\rightarrow$ `Linear(184, C)`<br>**Regression:** Frame-wise `Linear(92, 2)` | `104,238 – 106,642`<br>(`103,870` on 8-ch) | `412.4 – 423.1 KB` | **`9.52 ms`** (1-core) / `12.34 ms` (16-thread) |

### Bundled Checkpoint Payload Structure

Every `.pt` file in `D:\me\project\models\checkpoints\` is a self-contained PyTorch dictionary containing:
- **`Classification` Checkpoints (`44` files across `11` tracks):**
  - Keys: `['best_epoch', 'best_val_macro_f1', 'class_names', 'model_name', 'scaler', 'state_dict', 'track_id']`
  - `scaler`: `{'mean': [12 floats], 'std': [12 floats], 'fitted': True}` fitted strictly on `train` windows.
- **`Regression` Checkpoints (`20` files across `5` tracks):**
  - Keys: `['best_epoch', 'best_val_mae_deg', 'model_name', 'state_dict', 'target_names', 'track_id', 'x_scaler', 'y_scaler']`
  - `x_scaler`: `{'mean': [12 or 8 floats], 'std': [12 or 8 floats], 'fitted': True}` fitted strictly on `train` windows.
  - `y_scaler`: `{'mean': [2 floats], 'std': [2 floats], 'fitted': True}` for de-normalizing predicted angles back into physical degrees (`[elbow_flex_ext_deg, forearm_pro_sup_deg]`).

---

## 2. Track-Level Input/Output & Channel Contracts (All 16 Track Prefixes)

| Track Prefix (`models/checkpoints/<prefix>_<MODEL>_best.pt`) | Task Type | Dataset(s) & Split (`Train / Val / Test` Windows) | Expected Input Shape `(B, W, C)` & Rate | Feature / Channel Order | Output Shape & Target Meaning | Direct Sample Inference Status |
| :--- | :---: | :--- | :---: | :--- | :--- | :---: |
| **`track_1a_optotrak_10class_exercise`** | Classification (`10` classes) | `Optotrak vs DOT` (`15` subj: `10/2/3`)<br>`2,668 / 947 / 731` | `(B, 240, 12)` @ `120 Hz` | `CANONICAL_DUAL_IMU_12CH` (`UA` Acc+Gyr, `FA` Acc+Gyr in `m/s²`, `rad/s`; free acc) | `(B, 10)` logits $\rightarrow$ `['elbow_flex_ext', 'forearm_pro_sup', 'shoulder_flex', 'shoulder_abd', 'shoulder_int_ext_rot', 'wrist_flex_ext', 'drinking_cup', 'drawing_circles', 'box_off_shelf', 'reaching']` | **READY (Primary)** |
| **`track_1b_ju_imu_14class_exercise`** | Classification (`14` classes) | `JU-IMU` (`41` subj: `29/6/6`)<br>`4,305 / 1,160 / 979` | `(B, 160, 12)` @ `80 Hz` | `CANONICAL_DUAL_IMU_12CH` (`m/s²`, `rad/s`) | `(B, 14)` logits $\rightarrow$ `['ElbFlex', 'SupPro', 'FrontSupPro', 'WrstFlex', 'WrstExt', 'ShFlex90', 'ShFlex180', 'ShAbd90', 'ShAbd180', 'ShExt', 'ShRot', 'PickCup', 'Write', 'DoorKnock']` | **READY (Primary)** |
| **`track_1c_phytmo_3class_exercise`** | Classification (`3` classes) | `PHYTMO` (`30` subj: `20/5/5`)<br>`8,815 / 3,117 / 2,503` | `(B, 200, 12)` @ `100 Hz` | `CANONICAL_DUAL_IMU_12CH` (`m/s²`, `rad/s`) | `(B, 3)` logits $\rightarrow$ `['EFE (Elbow Flex-Ext)', 'EAH (Elev Arm)', 'SQZ (Squeeze)']` | **READY (Primary)** |
| **`track_2a_phytmo_efe_quality_binary`** | Classification (`2` classes) | `PHYTMO` (`27` subj: `18/4/5`)<br>`3,705 / 1,483 / 1,238` | `(B, 200, 12)` @ `100 Hz` | `CANONICAL_DUAL_IMU_12CH` (`m/s²`, `rad/s`) | `(B, 2)` logits $\rightarrow$ `['EFE0 (Correct)', 'EFE1 (Incorrect)']` | **READY (Primary)** |
| **`track_2b_phytmo_6class_exercise_quality`** | Classification (`6` classes) | `PHYTMO` (`30` subj: `20/5/5`)<br>`8,815 / 3,117 / 2,503` | `(B, 200, 12)` @ `100 Hz` | `CANONICAL_DUAL_IMU_12CH` (`m/s²`, `rad/s`) | `(B, 6)` logits $\rightarrow$ `['EFE0_Correct', 'EFE1_Incorrect', 'EAH0_Correct', 'EAH1_Incorrect', 'SQZ0_Correct', 'SQZ1_Incorrect']` | **READY (Primary)** |
| **`track_3a_optotrak_elbow_prosup_regression`** | Regression (`2` targets) | `Optotrak vs DOT` (`14` subj: `9/2/3`)<br>`2,387 / 947 / 731` | `(B, 240, 12)` @ `120 Hz` | `CANONICAL_DUAL_IMU_12CH` (`m/s²`, `rad/s`; free acc) | `(B, 240, 2)` angles in `deg` $\rightarrow$ `['elbow_flex_ext_deg', 'forearm_pro_sup_deg']` + Window/Trial ROM (`deg`) | **READY (Primary)** |
| **`track_3b_strokerehab_angle_regression`** | Regression (`2` targets) | `StrokeRehab` (`20` Healthy Controls: `14/3/3`)<br>`8,181 / 2,147 / 2,170` | `(B, 200, 12)` @ `100 Hz` | `CANONICAL_DUAL_IMU_12CH` (`m/s²`, differentiated Euler `rad/s`) | `(B, 200, 2)` angles in `deg` $\rightarrow$ `['elbow_flex_ext_deg', 'forearm_pro_sup_deg']` | **READY (Auxiliary)** |
| **`track_3c_ultra_mocap_angle_regression`** | Regression (`2` targets) | `ULTra-MoCap` (`13` subj: `9/2/2`)<br>`5,384 / 1,418 / 1,382` | `(B, 200, 12)` @ `100 Hz` | `CANONICAL_DUAL_IMU_12CH` (`m/s²`, `rad/s`) | `(B, 200, 2)` angles in `deg` $\rightarrow$ `['elbow_flex_ext_deg', 'forearm_pro_sup_deg']` + Window/Trial ROM (`deg`) | **READY (Primary)** |
| **`track_3d_ulf_in_adl_stroke_angle_regression`** | Regression (`2` targets) | `ULF_in_ADL` (`25` subj: `17/4/4`)<br>`582 / 143 / 8,848` | `(B, 120, 12)` @ `60 Hz` | `CANONICAL_DUAL_IMU_12CH` (`m/s²`, `rad/s`) | `(B, 120, 2)` angles in `deg` $\rightarrow$ `['elbow_flex_ext_deg', 'forearm_pro_sup_deg']` | **UNRELIABLE (`P28` skew)** |
| **`track_3e_vidimu_quaternion_angle_regression`** | Regression (`2` targets) | `VIDIMU` (`16` subj: `10/3/3`)<br>`4,054 / 1,187 / 1,110` | `(B, 100, 8)` @ `50 Hz` | `8-ch Quaternions` (`[ua_q0..q3, fa_q0..q3]`) | `(B, 100, 2)` angles in `deg` $\rightarrow$ `['elbow_flex_ext_deg', 'forearm_pro_sup_deg']` | **READY (Aux 8-Ch Only)** |
| **`track_4a_ju_imu_stroke_vs_nd`** | Classification (`2` classes) | `JU-IMU` (`41` subj: `29/6/6`)<br>`4,305 / 1,160 / 979` | `(B, 160, 12)` @ `80 Hz` | `CANONICAL_DUAL_IMU_12CH` (`m/s²`, `rad/s`) | `(B, 2)` logits $\rightarrow$ `['Non-Disabled (ND)', 'Stroke']` | **READY (Aux Stroke)** |
| **`track_4b_ulf_in_adl_3class_limb_impairment`** | Classification (`3` classes) | `ULF_in_ADL` (`25` subj: `17/4/4`)<br>`582 / 143 / 8,848` | `(B, 120, 12)` @ `60 Hz` | `CANONICAL_DUAL_IMU_12CH` (`m/s²`, `rad/s`) | `(B, 3)` logits $\rightarrow$ `['Healthy', 'Non-Paretic', 'Paretic']` | **EXCLUDED (`F1=0.36`)** |
| **`track_4c_rehab_fma_ue_3class`** | Classification (`3` classes) | `REHAB-120` (`222` sessions: `154/34/34`)<br>`2,459 / 543 / 543` | `(B, 150, 12)` @ `50 Hz` | `CANONICAL_DUAL_IMU_12CH` (`m/s²`, `rad/s`) | `(B, 3)` logits $\rightarrow$ `['FMA_0_Cannot', 'FMA_1_Partial', 'FMA_2_Full']` | **READY (Aux Session-Split)** |
| **`track_5a1_cross_dataset_ju_to_optotrak_7class`** | Classification (`7` classes) | Train/Val `JU-IMU` $\rightarrow$ Test `Optotrak`<br>`5,141 / 1,432 / 2,964` | `(B, 200, 12)` @ `100 Hz` | `compute_orientation_invariant_12ch` (`12` invariant features) | `(B, 7)` logits $\rightarrow$ `['elbow_flex_ext', 'forearm_pro_sup', 'wrist_flex_ext', 'shoulder_flex', 'shoulder_abd', 'shoulder_rot', 'drinking_cup']` | **READY (Offline; Non-Causal Trial Mean)** |
| **`track_5a2_cross_dataset_to_phytmo_2class`** | Classification (`2` classes) | Train/Val `Optotrak+JU` $\rightarrow$ Test `PHYTMO`<br>`1,637 / 496 / 5,103` | `(B, 200, 12)` @ `100 Hz` | `compute_orientation_invariant_12ch` (`12` invariant features) | `(B, 2)` logits $\rightarrow$ `['elbow_flex_ext (EFE0)', 'arm_elevation (EAH0)']` | **READY (Offline; Non-Causal Trial Mean)** |
| **`track_5b_ultra_mocap_cross_speed`** | Classification (`3` classes) | `ULTra-MoCap` (`Normal` $\rightarrow$ `Slow/Fast/VeryFast`)<br>`1,427 / 312 / 4,099` | `(B, 200, 12)` @ `100 Hz` | `CANONICAL_DUAL_IMU_12CH` (`m/s²`, `rad/s`) | `(B, 3)` logits $\rightarrow$ `['elbow_flex_ext', 'forearm_pro_sup', 'reaching_adl']` | **READY (Aux Speed-Split)** |

---

## 3. Complete Inventory of All 64 Checkpoint Files (`models/checkpoints/*.pt`)

### 3.1 Classification Checkpoints (`44` Files across `11` Tracks)

| # | Checkpoint Filename (`models/checkpoints/`) | Model | Input `(W, C)` | Output Classes ($C$) | Params | Size (`KB`) | Best Ep | Val Macro-F1 | **Test Macro-F1** | **Test Bal. Acc** | Test Acc | Sample Inference Status |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :-: | :---: | :---: | :---: | :---: | :---: |
| 1 | `track_1a_optotrak_10class_exercise_TCN_best.pt` | `TCN` | `(240, 12)` | `10` | `116,322` | `481.4` | `14` | `0.9364` | `0.9682` | `0.9685` | `0.9699` | **Reusable (Primary)** |
| 2 | `track_1a_optotrak_10class_exercise_1D_CNN_best.pt` | `1D_CNN` | `(240, 12)` | `10` | `87,114` | `354.9` | `5` | `0.9222` | **`0.9808`** | **`0.9825`** | **`0.9767`** | **Reusable (Primary Winner)** |
| 3 | `track_1a_optotrak_10class_exercise_LSTM_best.pt` | `LSTM` | `(240, 12)` | `10` | `106,490` | `422.6` | `5` | `0.9235` | `0.9764` | `0.9760` | `0.9754` | **Reusable (Primary)** |
| 4 | `track_1a_optotrak_10class_exercise_GRU_best.pt` | `GRU` | `(240, 12)` | `10` | `105,902` | `420.3` | `10` | `0.9439` | `0.9753` | `0.9733` | `0.9672` | **Reusable (Primary)** |
| 5 | `track_1b_ju_imu_14class_exercise_TCN_best.pt` | `TCN` | `(160, 12)` | `14` | `116,774` | `483.0` | `8` | `0.7635` | **`0.8030`** | **`0.7699`** | **`0.7692`** | **Reusable (Primary Winner)** |
| 6 | `track_1b_ju_imu_14class_exercise_1D_CNN_best.pt` | `1D_CNN` | `(160, 12)` | `14` | `87,886` | `357.9` | `8` | `0.7460` | `0.7820` | `0.7619` | `0.7589` | **Reusable (Primary)** |
| 7 | `track_1b_ju_imu_14class_exercise_LSTM_best.pt` | `LSTM` | `(160, 12)` | `14` | `107,134` | `425.1` | `2` | `0.7411` | `0.7935` | `0.7655` | `0.7681` | **Reusable (Primary)** |
| 8 | `track_1b_ju_imu_14class_exercise_GRU_best.pt` | `GRU` | `(160, 12)` | `14` | `106,642` | `423.1` | `4` | `0.7522` | `0.7973` | `0.7614` | `0.7640` | **Reusable (Primary)** |
| 9 | `track_1c_phytmo_3class_exercise_TCN_best.pt` | `TCN` | `(200, 12)` | `3` | `115,531` | `477.9` | `2` | `0.8883` | **`0.9400`** | **`0.9429`** | **`0.9393`** | **Reusable (Primary Winner)** |
| 10 | `track_1c_phytmo_3class_exercise_1D_CNN_best.pt` | `1D_CNN` | `(200, 12)` | `3` | `85,763` | `349.3` | `6` | `0.8766` | `0.9333` | `0.9358` | `0.9309` | **Reusable (Primary)** |
| 11 | `track_1c_phytmo_3class_exercise_LSTM_best.pt` | `LSTM` | `(200, 12)` | `3` | `105,363` | `418.0` | `2` | `0.8974` | `0.9289` | `0.9339` | `0.9289` | **Reusable (Primary)** |
| 12 | `track_1c_phytmo_3class_exercise_GRU_best.pt` | `GRU` | `(200, 12)` | `3` | `104,607` | `415.0` | `3` | `0.8869` | `0.9258` | `0.9270` | `0.9241` | **Reusable (Primary)** |
| 13 | `track_2a_phytmo_efe_quality_binary_TCN_best.pt` | `TCN` | `(200, 12)` | `2` | `115,418` | `477.6` | `7` | `0.8113` | **`0.8925`** | **`0.8939`** | **`0.8926`** | **Reusable (Primary Winner)** |
| 14 | `track_2a_phytmo_efe_quality_binary_1D_CNN_best.pt` | `1D_CNN` | `(200, 12)` | `2` | `85,570` | `348.7` | `6` | `0.8075` | `0.8135` | `0.8189` | `0.8142` | **Reusable (Primary)** |
| 15 | `track_2a_phytmo_efe_quality_binary_LSTM_best.pt` | `LSTM` | `(200, 12)` | `2` | `105,202` | `417.3` | `3` | `0.8101` | `0.8819` | `0.8855` | `0.8821` | **Reusable (Primary)** |
| 16 | `track_2a_phytmo_efe_quality_binary_GRU_best.pt` | `GRU` | `(200, 12)` | `2` | `104,422` | `414.3` | `11` | `0.8077` | `0.8829` | `0.8847` | `0.8829` | **Reusable (Primary)** |
| 17 | `track_2b_phytmo_6class_exercise_quality_TCN_best.pt` | `TCN` | `(200, 12)` | `6` | `115,870` | `479.8` | `16` | `0.7094` | `0.7655` | `0.7686` | `0.7623` | **Reusable (Primary)** |
| 18 | `track_2b_phytmo_6class_exercise_quality_1D_CNN_best.pt` | `1D_CNN` | `(200, 12)` | `6` | `86,342` | `352.0` | `10` | `0.6880` | `0.8253` | `0.8278` | `0.8210` | **Reusable (Primary)** |
| 19 | `track_2b_phytmo_6class_exercise_quality_LSTM_best.pt` | `LSTM` | `(200, 12)` | `6` | `105,846` | `420.1` | `2` | `0.6768` | **`0.8356`** | **`0.8385`** | **`0.8358`** | **Reusable (Primary Winner)** |
| 20 | `track_2b_phytmo_6class_exercise_quality_GRU_best.pt` | `GRU` | `(200, 12)` | `6` | `105,162` | `417.4` | `1` | `0.6809` | `0.8081` | `0.8155` | `0.8114` | **Reusable (Primary)** |
| 21 | `track_4a_ju_imu_stroke_vs_nd_TCN_best.pt` | `TCN` | `(160, 12)` | `2` | `115,418` | `474.7` | `1` | `0.6604` | `0.7683` | `0.7650` | `0.7978` | **Reusable (Aux Stroke)** |
| 22 | `track_4a_ju_imu_stroke_vs_nd_1D_CNN_best.pt` | `1D_CNN` | `(160, 12)` | `2` | `85,570` | `348.5` | `3` | `0.6629` | `0.8653` | `0.8732` | `0.8784` | **Reusable (Aux Stroke)** |
| 23 | `track_4a_ju_imu_stroke_vs_nd_LSTM_best.pt` | `LSTM` | `(160, 12)` | `2` | `105,202` | `417.2` | `1` | `0.6459` | `0.7278` | `0.7282` | `0.7589` | **Reusable (Aux Stroke)** |
| 24 | `track_4a_ju_imu_stroke_vs_nd_GRU_best.pt` | `GRU` | `(160, 12)` | `2` | `104,422` | `414.1` | `6` | `0.6537` | **`0.8733`** | **`0.8990`** | **`0.8815`** | **Reusable (Aux Winner)** |
| 25 | `track_4b_ulf_in_adl_3class_limb_impairment_TCN_best.pt` | `TCN` | `(120, 12)` | `3` | `115,531` | `478.6` | `19` | `0.5222` | `0.3435` | `0.4253` | `0.5026` | **Excluded (`P28` + low F1)** |
| 26 | `track_4b_ulf_in_adl_3class_limb_impairment_1D_CNN_best.pt` | `1D_CNN` | `(120, 12)` | `3` | `85,763` | `349.8` | `5` | `0.4130` | `0.3091` | `0.4781` | `0.4977` | **Excluded (`P28` + low F1)** |
| 27 | `track_4b_ulf_in_adl_3class_limb_impairment_LSTM_best.pt` | `LSTM` | `(120, 12)` | `3` | `105,363` | `418.2` | `3` | `0.4529` | `0.3240` | `0.3529` | `0.5075` | **Excluded (`P28` + low F1)** |
| 28 | `track_4b_ulf_in_adl_3class_limb_impairment_GRU_best.pt` | `GRU` | `(120, 12)` | `3` | `104,607` | `415.2` | `6` | `0.4020` | `0.3620` | `0.4367` | `0.5186` | **Excluded (`P28` + low F1)** |
| 29 | `track_4c_rehab_fma_ue_3class_TCN_best.pt` | `TCN` | `(150, 12)` | `3` | `115,531` | `475.1` | `13` | `0.7875` | `0.7720` | `0.7715` | `0.7974` | **Aux Only (Session Split)** |
| 30 | `track_4c_rehab_fma_ue_3class_1D_CNN_best.pt` | `1D_CNN` | `(150, 12)` | `3` | `85,763` | `349.2` | `7` | `0.7328` | `0.6956` | `0.6930` | `0.7293` | **Aux Only (Session Split)** |
| 31 | `track_4c_rehab_fma_ue_3class_LSTM_best.pt` | `LSTM` | `(150, 12)` | `3` | `105,363` | `417.9` | `11` | `0.8180` | `0.7819` | `0.7805` | `0.8011` | **Aux Only (Session Split)** |
| 32 | `track_4c_rehab_fma_ue_3class_GRU_best.pt` | `GRU` | `(150, 12)` | `3` | `104,607` | `414.9` | `23` | `0.8115` | **`0.8051`** | **`0.8036`** | **`0.8250`** | **Aux Only (Session Split)** |
| 33 | `track_5a1_cross_dataset_ju_to_optotrak_7class_TCN_best.pt` | `TCN` | `(200, 12)` | `7` | `115,983` | `480.8` | `6` | `0.7655` | `0.7056` | `0.7576` | `0.6980` | **Reusable (Offline Inv-12ch)** |
| 34 | `track_5a1_cross_dataset_ju_to_optotrak_7class_1D_CNN_best.pt` | `1D_CNN` | `(200, 12)` | `7` | `86,535` | `353.1` | `16` | `0.7810` | `0.7073` | `0.7533` | `0.7055` | **Reusable (Offline Inv-12ch)** |
| 35 | `track_5a1_cross_dataset_ju_to_optotrak_7class_LSTM_best.pt` | `LSTM` | `(200, 12)` | `7` | `106,007` | `420.9` | `3` | `0.7514` | **`0.7392`** | **`0.7781`** | **`0.7281`** | **Reusable (Offline Winner)** |
| 36 | `track_5a1_cross_dataset_ju_to_optotrak_7class_GRU_best.pt` | `GRU` | `(200, 12)` | `7` | `105,347` | `418.3` | `5` | `0.7722` | `0.7374` | `0.7609` | `0.7271` | **Reusable (Offline Inv-12ch)** |
| 37 | `track_5a2_cross_dataset_to_phytmo_2class_TCN_best.pt` | `TCN` | `(200, 12)` | `2` | `115,418` | `478.0` | `5` | `0.9252` | `0.9854` | `0.9866` | `0.9857` | **Reusable (Offline Inv-12ch)** |
| 38 | `track_5a2_cross_dataset_to_phytmo_2class_1D_CNN_best.pt` | `1D_CNN` | `(200, 12)` | `2` | `85,570` | `348.9` | `10` | `0.9252` | **`0.9892`** | **`0.9904`** | **`0.9894`** | **Reusable (Offline Winner)** |
| 39 | `track_5a2_cross_dataset_to_phytmo_2class_LSTM_best.pt` | `LSTM` | `(200, 12)` | `2` | `105,202` | `417.5` | `3` | `0.9252` | `0.9824` | `0.9846` | `0.9828` | **Reusable (Offline Inv-12ch)** |
| 40 | `track_5a2_cross_dataset_to_phytmo_2class_GRU_best.pt` | `GRU` | `(200, 12)` | `2` | `104,422` | `414.4` | `1` | `0.9252` | `0.9856` | `0.9871` | `0.9859` | **Reusable (Offline Inv-12ch)** |
| 41 | `track_5b_ultra_mocap_cross_speed_TCN_best.pt` | `TCN` | `(200, 12)` | `3` | `115,531` | `477.9` | `8` | `0.8263` | `0.8989` | `0.9016` | `0.9005` | **Aux Only (Speed Split)** |
| 42 | `track_5b_ultra_mocap_cross_speed_1D_CNN_best.pt` | `1D_CNN` | `(200, 12)` | `3` | `85,763` | `349.4` | `2` | `0.8345` | `0.9019` | `0.9048` | `0.9029` | **Aux Only (Speed Split)** |
| 43 | `track_5b_ultra_mocap_cross_speed_LSTM_best.pt` | `LSTM` | `(200, 12)` | `3` | `105,363` | `417.9` | `14` | `0.8748` | **`0.9113`** | **`0.9114`** | **`0.9136`** | **Aux Only (Speed Split)** |
| 44 | `track_5b_ultra_mocap_cross_speed_GRU_best.pt` | `GRU` | `(200, 12)` | `3` | `104,607` | `415.0` | `4` | `0.8548` | `0.9076` | `0.9087` | `0.9095` | **Aux Only (Speed Split)** |

---

### 3.2 Continuous Angle & ROM Regression Checkpoints (`20` Files across `5` Tracks)

| # | Checkpoint Filename (`models/checkpoints/`) | Model | Input `(W, C)` | Output `(W, D)` | Params | Size (`KB`) | Best Ep | Overall MAE (`deg`) | **Elbow Flex MAE (`deg`)** ($R^2$) | **Forearm ProSup MAE (`deg`)** ($R^2$) | **Window ROM MAE (`deg`)** | Sample Inference Status |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :-: | :---: | :---: | :---: | :---: | :---: |
| 45 | `track_3a_optotrak_elbow_prosup_regression_TCN_best.pt` | `TCN` | `(240, 12)` | `(240, 2)` | `115,306` | `477.8` | `1` | `16.99°` | `13.28°` (`0.8014`) | `20.70°` (`0.0627`) | `18.45°` | **Reusable (Primary)** |
| 46 | `track_3a_optotrak_elbow_prosup_regression_1D_CNN_best.pt` | `1D_CNN` | `(240, 12)` | `(240, 2)` | `85,378` | `348.3` | `3` | `17.68°` | `12.91°` (`0.8140`) | `22.45°` (`-0.0405`) | `19.92°` | **Reusable (Primary)** |
| 47 | `track_3a_optotrak_elbow_prosup_regression_LSTM_best.pt` | `LSTM` | `(240, 12)` | `(240, 2)` | `105,042` | `417.0` | `2` | `16.77°` | `12.58°` (`0.8087`) | `20.96°` (`0.0743`) | `18.95°` | **Reusable (Primary)** |
| 48 | `track_3a_optotrak_elbow_prosup_regression_GRU_best.pt` | `GRU` | `(240, 12)` | `(240, 2)` | `104,238` | `413.8` | `6` | **`16.19°`** | **`12.24°`** (`0.8103`) | **`20.14°`** (`0.0848`) | **`14.30°`** | **Reusable (Primary Winner)** |
| 49 | `track_3b_strokerehab_angle_regression_TCN_best.pt` | `TCN` | `(200, 12)` | `(200, 2)` | `115,306` | `477.5` | `15` | **`21.71°`** | **`13.52°`** (`0.4634`) | **`29.91°`** (`0.4381`) | **`17.89°`** | **Reusable (Aux Healthy ADL)** |
| 50 | `track_3b_strokerehab_angle_regression_1D_CNN_best.pt` | `1D_CNN` | `(200, 12)` | `(200, 2)` | `85,378` | `348.2` | `22` | `23.63°` | `15.77°` (`0.2898`) | `31.50°` (`0.3497`) | `24.05°` | **Reusable (Aux Healthy ADL)** |
| 51 | `track_3b_strokerehab_angle_regression_LSTM_best.pt` | `LSTM` | `(200, 12)` | `(200, 2)` | `105,042` | `416.9` | `18` | `22.48°` | `14.60°` (`0.3694`) | `30.35°` (`0.4123`) | `21.18°` | **Reusable (Aux Healthy ADL)** |
| 52 | `track_3b_strokerehab_angle_regression_GRU_best.pt` | `GRU` | `(200, 12)` | `(200, 2)` | `104,238` | `413.8` | `9` | `23.66°` | `16.05°` (`0.2815`) | `31.27°` (`0.3753`) | `19.91°` | **Reusable (Aux Healthy ADL)** |
| 53 | `track_3c_ultra_mocap_angle_regression_TCN_best.pt` | `TCN` | `(200, 12)` | `(200, 2)` | `115,306` | `477.5` | `11` | **`10.07°`** | **`9.96°`** (`0.8876`) | **`10.19°`** (`0.8869`) | **`10.95°`** | **Reusable (Primary Winner)** |
| 54 | `track_3c_ultra_mocap_angle_regression_1D_CNN_best.pt` | `1D_CNN` | `(200, 12)` | `(200, 2)` | `85,378` | `348.2` | `11` | `11.72°` | `10.88°` (`0.8663`) | `12.55°` (`0.8278`) | `13.72°` | **Reusable (Primary)** |
| 55 | `track_3c_ultra_mocap_angle_regression_LSTM_best.pt` | `LSTM` | `(200, 12)` | `(200, 2)` | `105,042` | `416.9` | `9` | `11.77°` | `11.96°` (`0.8288`) | `11.58°` (`0.8457`) | `14.16°` | **Reusable (Primary)** |
| 56 | `track_3c_ultra_mocap_angle_regression_GRU_best.pt` | `GRU` | `(200, 12)` | `(200, 2)` | `104,238` | `413.8` | `2` | `10.65°` | `10.06°` (`0.8840`) | `11.23°` (`0.8593`) | `13.60°` | **Reusable (Primary)** |
| 57 | `track_3d_ulf_in_adl_stroke_angle_regression_TCN_best.pt` | `TCN` | `(120, 12)` | `(120, 2)` | `115,306` | `477.9` | `16` | `17.54°` (SubjEq: **`17.59°`**) | `15.70°` (`0.4173`) | `19.38°` (`-0.2306`) | `13.68°` | **Unreliable (`P28` skew)** |
| 58 | `track_3d_ulf_in_adl_stroke_angle_regression_1D_CNN_best.pt` | `1D_CNN` | `(120, 12)` | `(120, 2)` | `85,378` | `348.4` | `20` | `19.15°` (SubjEq: `20.89°`) | `19.32°` (`0.1093`) | `18.99°` (`-0.1796`) | `33.12°` | **Unreliable (`P28` skew)** |
| 59 | `track_3d_ulf_in_adl_stroke_angle_regression_LSTM_best.pt` | `LSTM` | `(120, 12)` | `(120, 2)` | `105,042` | `417.0` | `16` | `16.83°` (SubjEq: `19.33°`) | `15.40°` (`0.2706`) | `18.26°` (`-0.0981`) | `12.98°` | **Unreliable (`P28` skew)** |
| 60 | `track_3d_ulf_in_adl_stroke_angle_regression_GRU_best.pt` | `GRU` | `(120, 12)` | `(120, 2)` | `104,238` | `413.9` | `22` | `17.04°` (SubjEq: `19.04°`) | `14.93°` (`0.3202`) | `19.16°` (`-0.1923`) | `12.63°` | **Unreliable (`P28` skew)** |
| 61 | `track_3e_vidimu_quaternion_angle_regression_TCN_best.pt` | `TCN` | `(100, 8)` | `(100, 2)` | `113,962` | `472.6` | `21` | **`13.98°`** | `13.68°` (`0.7344`) | **`14.29°`** (`0.8106`) | **`8.73°`** | **Reusable (Aux 8-Ch Quat)** |
| 62 | `track_3e_vidimu_quaternion_angle_regression_1D_CNN_best.pt` | `1D_CNN` | `(100, 8)` | `(100, 2)` | `83,586` | `341.4` | `19` | `15.18°` | **`13.33°`** (`0.7028`) | `17.03°` (`0.7188`) | `10.09°` | **Reusable (Aux 8-Ch Quat)** |
| 63 | `track_3e_vidimu_quaternion_angle_regression_LSTM_best.pt` | `LSTM` | `(100, 8)` | `(100, 2)` | `104,722` | `415.7` | `16` | `15.93°` | `14.75°` (`0.6039`) | `17.11°` (`0.6783`) | `9.89°` | **Reusable (Aux 8-Ch Quat)** |
| 64 | `track_3e_vidimu_quaternion_angle_regression_GRU_best.pt` | `GRU` | `(100, 8)` | `(100, 2)` | `103,870` | `412.4` | `4` | `17.03°` | `16.73°` (`0.4894`) | `17.34°` (`0.7264`) | `10.42°` | **Reusable (Aux 8-Ch Quat)** |

---

## 4. Missing or Incompatible Model-Task Combinations Identified for Stage 10.2

1. **No `TaskOutputHead(task_mode="multitask")` Checkpoints Yet:**
   - Although `TaskOutputHead(task_mode="multitask")` and `compute_masked_multitask_loss()` are implemented in [`src/models/architectures.py`](file:///d:/me/project/src/models/architectures.py#L40-L66), all 64 existing checkpoints use either `task_mode="classification"` or `task_mode="regression"`.
2. **Sampling Rate Mismatch Between `Optotrak vs DOT` (`120 Hz`, `W=240`) and `PHYTMO` / `ULTra-MoCap` (`100 Hz`, `W=200`):**
   - `Track 1A` and `Track 3A` checkpoints were trained on `120 Hz` windows (`W=240`), whereas `Tracks 1C, 2A, 2B, 3C` were trained on `100 Hz` windows (`W=200`), and `Tracks 1B, 4A` on `80 Hz` windows (`W=160`).
   - While convolutional/recurrent weights in PyTorch do not hardcode $W$, temporal receptive fields (`kernel_size`, `dilation`) are calibrated to the sampling rate (`Hz`), and the bundled `ChannelStandardScaler` statistics reflect each dataset's gravity convention (`free acceleration` in `Optotrak vs DOT` vs. `proper acceleration` in `ULTra-MoCap` and `PHYTMO`).
3. **Orientation-Invariant Cross-Dataset Checkpoints (`Tracks 5A-1 & 5A-2`) Use Full-Trial Mean:**
   - As documented in `STAGE10_READINESS_AUDIT.md`, `Tracks 5A-1` and `5A-2` expect the 12 orientation-invariant features from `compute_orientation_invariant_12ch()`, which currently subtracts the full-trial mean acceleration before windowing. For causal streaming in Stage 10, a window-local version of `compute_orientation_invariant_12ch()` is needed.
4. **Repetition-Cycle Peak-to-Trough ROM Segmentation:**
   - Existing regression inference in [`src/inference/engine.py`](file:///d:/me/project/src/inference/engine.py#L1325-L1339) outputs both 2.0-second sliding-window ROM (`rom_mae_deg`) and full-trial peak-to-trough ROM (`pred_flexion_rom_deg`, `pred_prosup_rom_deg`). Adding per-repetition cycle detection (identifying individual flexion-extension cycles within a multi-rep recording) in Stage 10 will complete the clinical ROM analytics suite.
