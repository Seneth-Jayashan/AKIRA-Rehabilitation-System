# Android Deployment Specification

- **Input Tensor Specification**: `[1, 240, 12]` Float32.
- **Channel Ordering**: RUA_Ax, RUA_Ay, RUA_Az, RUA_Gx, RUA_Gy, RUA_Gz, RLA_Ax, RLA_Ay, RLA_Az, RLA_Gx, RLA_Gy, RLA_Gz.
- **Sampling Requirement**: 120Hz hardware rate -> filtered via 10Hz Butterworth -> windowed into 240 samples (2 seconds).
- **Normalization Parameters**: Requires Z-score parameters extracted from the training set (`mean`, `std` vectors length 12).
- **Output Tensors**: Forearm `[1, 240, 2]`, Elbow `[1, 240, 1]`, Class `[1, 20]`.
- **Atan2 Post-processing**: `atan2(forearm[...,0], forearm[...,1])` -> convert to degrees.
- **Elbow Output Processing**: Native degrees (clipped to 0-180 if needed).
- **Classification Decoding**: Softmax -> Argmax -> `class_mapping.json` string mapping.
- **FP32 Model Size**: ~125 KB.
- **FP16 Model Size**: ~65 KB.
