# Final Viva Summary

1. **What is the research problem?** Replacing stationary optical tracking with wearable IMUs for remote joint kinematics.
2. **Why are two IMUs needed?** Upper arm and lower arm spatial vectors are required to triangulate the elbow hinge and rotational twist.
3. **Why 12 channels?** 6 (Acc+Gyr) per IMU. Both sensors are proven necessary by ablation.
4. **Why TCN?** Dilated convolutions extract wide temporal dependencies at a fraction of LSTM compute cost.
5. **Why 240 samples?** Balances low latency for real-time feedback with enough kinematic context.
6. **Why 2 seconds?** Captures standard repetition cycle physiological limits.
7. **Why predict sin/cos for forearm rotation?** Prevents mathematical explosion and MSE penalty inflation at the physical ±180° orientation wrap.
8. **Why Huber loss for elbow?** Combats outliers and tracking anomalies found in Optotrak ground truth without permanently mutating dataset files.
9. **How was data leakage prevented?** Strict subject-level partitioning (10/3/2) prior to Z-score scaling.
10. **How was the model evaluated on unseen subjects?** Frozen validation-checked model evaluated exactly once on Subject 10 and 4.
11. **Why did TCN outperform Linear?** Kinematics are mathematically unresolvable from instantaneous slices due to integration drift and ambiguity.
12. **Why compare against LSTM?** LSTM is the historical time-series gold standard; proving TCN beats it justifies the architectural shift.
13. **What does 13.38° elbow MAE mean?** The model tracks continuous joint angles within ~13 degrees of optical hardware precision.
14. **What does 24.41° forearm MAE mean?** Rotational kinematics are harder to extract from IMU gravity drift than linear hinges.
15. **How are ROM and velocity obtained?** Downstream discrete math applied to the continuous trajectory outputs.
16. **How was repetition counting validated?** SciPy peak finding (prominence thresholded) compared identically across truth and prediction streams.
17. **Why is classification only ~63–67%?** 2 seconds is often less than one full repetition of a complex exercise, leading to biomechanical confusion between similar poses.
18. **What did the ablation study prove?** Removing any sensor drops performance; Multi-tasking preserves individual accuracy while merging representation.
19. **Can this currently be claimed as an ORIF rehabilitation model?** NO. It is a kinematic estimator built on a healthy cohort.
20. **What must happen before clinical deployment?** Validation via clinical trials tracking pathological compensatory movements.
