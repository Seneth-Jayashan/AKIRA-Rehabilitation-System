# 02 Model Development

## 1. Purpose of the Folder
This directory encapsulates the iterative design and empirical justification of the neural network architectures explored in this research.

## 2. What Files are Inside
- **tcn_architecture/**: Deep architectural design documents, summary parameters, and training reports specific to the Temporal Convolutional Network.
- **baseline_comparison/**: Head-to-head empirical evaluations comparing the TCN against the baseline LSTM and Linear models.
- **ablation/**: Ablation studies verifying the necessity of specific architectural components (e.g., dilated convolutions).

## 3. Why Those Files Matter
These documents form the theoretical and empirical backbone of the study. They justify *why* the TCN was chosen over standard RNNs for this kinematic problem and provide verifiable proof of training stability.

## 4. Workflow Context
This represents **Stage 2** of the research pipeline. It follows the data audit and precedes the final locked-down test set evaluation.

## 5. What a Supervisor Should Look at First
Start with `baseline_comparison/baseline_comparison.md` for a high-level overview of how the primary algorithms fared, then examine `tcn_architecture/tcn_architecture_design.md` for the specific structural innovations.
