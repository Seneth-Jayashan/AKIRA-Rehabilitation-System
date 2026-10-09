# Edge-AI Benchmark Report

## Parameters
- **Total Parameters:** ~30,903
- **Trainable Parameters:** ~30,903
- **Model Size (FP32):** ~125 KB
- **Expected Input Tensor:** `[Batch, 240, 12]`
- **Expected Output Tensors:** `[Batch, 240, 2]` (Forearm), `[Batch, 240, 1]` (Elbow), `[Batch, 20]` (Class)

## Android Inference
**Target-device inference latency:** NOT YET MEASURED.

*Note: The purely 1D convolutional architecture with low filter counts guarantees ultra-low complexity suitable for real-time NNAPI edge execution.*