# Phase 5: Feature Importance Analysis 🧠

To ensure our XGBoost model isn't just making arbitrary guesses, we have cracked open the "black box" using two different analytical methods: **XGBoost Native Gain** and **SHAP (SHapley Additive exPlanations)** values. 

This helps us answer a critical research question: *Which sensors (Accelerometer vs. Gyroscope) and which domains (Time vs. Frequency) actually matter the most for detecting lower-limb rehabilitation exercises?*

## 1. SHAP Global Feature Importance
SHAP values provide a robust, game-theoretic approach to explaining the model. This chart shows the top features globally across all classes. It stacks the impact for each specific exercise so we can see if a feature is important for "Walking" vs "Squats".

![SHAP Importance Plot](/home/sjay/.gemini/antigravity-ide/brain/63de1dce-e090-4db0-ba4d-c69296155605/feature_importance_shap.png)

> [!NOTE]
> **What this means:** The longer the bar for a specific feature, the higher its average impact on the model's final prediction! Notice how both Accelerometer (`a`) and Gyroscope (`g`) features appear, and how specific axes (like `az` and `gy`) often dominate.

## 2. XGBoost Native Information Gain
This chart plots the features based on **Gain** — essentially, the average improvement in accuracy brought by a feature to the branches it is on.

![XGBoost Gain Plot](/home/sjay/.gemini/antigravity-ide/brain/63de1dce-e090-4db0-ba4d-c69296155605/feature_importance_xgb.png)

> [!TIP]
> **Takeaway for Future Hardware:** If we notice that certain axes (e.g., `gx`) or metrics (e.g., `_zcr`) are consistently at the bottom of the list with almost zero importance, we can physically drop them or ignore them in our future embedded C++ code for the Android app. This reduces memory footprint and speeds up real-time inference!
