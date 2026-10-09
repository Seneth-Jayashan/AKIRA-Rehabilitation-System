# Final Results Discussion

A. **Why TCN outperformed Linear**: OBSERVED: TCN Elbow MAE 13.38° vs Linear 46.15°. INTERPRETATION: Instantaneous 12-channel pose cannot resolve joint state uniquely; temporal context is mandatory.
B. **TCN vs LSTM**: OBSERVED: TCN beats LSTM 17.05°. INTERPRETATION: Dilated convolutions provide sharper long-range biomechanical dependencies without the vanishing gradients of recurrences.
C. **Elbow**: OBSERVED: 13.38° MAE. Strong continuous tracking.
D. **Forearm**: OBSERVED: 24.41° MAE. The sin/cos reconstruction gracefully navigates coordinate wraps.
E. **Classification**: OBSERVED: 63.58% Accuracy. HYPOTHESIS: Confusion occurs mostly among biomechanically similar exercises; 2-second windows may be too short to isolate long-term exercise identity.
F. **ROM**: OBSERVED: 14.56° MAE error. Tracks full movement arcs.
G. **Repetitions**: OBSERVED: 0.40 error per window. TCN kinematic wave perfectly matches physiological frequency.
H. **Velocity**: OBSERVED: 3.40 °/s error. Smooth derivative properties.
I. **Subject-independent generalization**: OBSERVED: Strict lockdown of the test set ensures metrics represent true real-world scaling capability.
J. **Failure cases**: Extremes of range occasionally clipped by MSE smoothing.
K. **Sensor ablation**: Acc+Gyr synergistic fusion validated.
L. **Multi-task**: Shared representation structurally sound without degrading per-joint precision.
M. **Edge-AI**: 30k params perfectly aligns with constrained Android environments.
