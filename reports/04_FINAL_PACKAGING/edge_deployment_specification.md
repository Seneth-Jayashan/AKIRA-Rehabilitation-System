# Edge Deployment Specification

- **Model Size (FP32):** ~125 KB
- **Model Size (FP16/Quantized):** ~65 KB
- **Format:** PyTorch `state_dict` -> Export to ONNX -> Convert to TFLite.
- **Input Tensor:** `[1, 240, 12]` (Float32). Sensor fusion of 6-axis RUA and RLA IMUs.
- **Output Tensors:** 
  - Forearm: `[1, 240, 2]` (sin, cos)
  - Elbow: `[1, 240, 1]` (Degrees)
  - Class: `[1, 20]` (Logits)
- **Postprocessing:** Must apply `atan2(sin, cos)` on Forearm output to reconstruct physical angle.
- **Android Latency:** NOT YET MEASURED.
