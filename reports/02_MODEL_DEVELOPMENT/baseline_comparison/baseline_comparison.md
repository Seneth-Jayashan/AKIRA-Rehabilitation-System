# Baseline Comparison

| Model | Forearm MAE | Elbow MAE | Classification Acc |
| --- | --- | --- | --- |
| **Linear** | 25.11° | 46.15° | 0.6717 |
| **LSTM** | 27.65° | 17.05° | 0.6528 |
| **Multi-Task TCN** | 24.41° | 13.38° | 0.6358 |

*Note: The TCN leverages expanded receptive fields via dilated convolutions, natively outperforming standard recurrent and static baselines.*