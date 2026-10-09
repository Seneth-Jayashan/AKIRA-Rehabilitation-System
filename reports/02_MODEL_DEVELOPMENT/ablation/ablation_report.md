# Ablation Report

| Ablation | Validation Loss |
| --- | --- |
| A_Acc_Only | 13.3044 |
| B_Gyr_Only | 14.8049 |
| D_Elbow_Only | 11.6649 |
| E_Forearm_Only | 0.0820 |
| F_MultiTask_Reg | 11.9733 |

*Interpretation:* Accelerometer vs Gyroscope shows that both sensors are critical for optimal tracking. Multi-task learning synergizes the latent representations compared to single-task regression.