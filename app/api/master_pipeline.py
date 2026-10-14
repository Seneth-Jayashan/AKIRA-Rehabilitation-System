"""
========================================================================================
AKIRA KNEE REHABILITATION: MASTER END-TO-END PIPELINE
========================================================================================
Workflow:
  1. Load Preprocessed, Labeled Raw Sensor Dataset (Dataset 1: 709 trials @ 100 Hz, 12 channels)
  2. Run Math Physics Model on raw signals to dynamically compute the 16 Clinical Features
  3. Form Dual-Stream Inputs: [Input A: Raw 200x12] + [Input B: 16 Math Features]
  4. Feed into Hybrid Model (Any of the 6 architectures) for Training and Evaluation
  5. Demonstrate Single-Trial Live Patient Inference (Raw Data -> Math -> AI -> Prediction)
========================================================================================
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")

from math_feature_extractor import Clinical16FeatureExtractor
from hybrid_models import (
    build_hybrid_1_cnn, build_hybrid_2_bilstm, build_hybrid_3_cnn_bilstm,
    build_hybrid_4_resnet, build_hybrid_5_self_attention, build_hybrid_6_gated_attention
)

EXERCISE_NAMES = ["Walking", "Seated Knee Extension", "Mini Squat", "Step-Up"]
QUALITY_NAMES  = ["Good (Correct Form)", "Needs Improvement (Lag/Deficit)", "Bad / Compensatory Fault"]

def run_master_pipeline(selected_model="1", num_epochs=20):
    print("=" * 80)
    print("AKIRA KNEE REHABILITATION: MASTER END-TO-END PIPELINE EXECUTION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: LOAD PREPROCESSED LABELED RAW SENSOR DATASET (DATASET 1)
    # -------------------------------------------------------------------------
    ds1_path = os.path.join(DATASETS_DIR, "dataset_1_input_A_12channels.npz")
    print(f"\n[STEP 1] Loading preprocessed labeled raw sensor data from:\n        {ds1_path}")
    data = np.load(ds1_path, allow_pickle=True)
    X_raw_seq = data["X_raw_12ch"]          # (709, 200, 12)
    y_class   = data["y_exercise_class"]    # (709,)
    y_qual    = data["y_quality_grade"]      # (709,)
    N_trials  = len(y_class)

    print(f"  [OK] Loaded {N_trials} total trials.")
    print(f"  [OK] Raw Sequence Shape: {X_raw_seq.shape} (2.0s window @ 100 Hz, 12 synchronized IMU channels)")
    print(f"  [OK] Ground Truth Labels: 4 Exercise Classes & 3 Clinical Quality Tiers")

    # -------------------------------------------------------------------------
    # STEP 2: APPLY MATH MODEL TO DERIVE THE 16 CLINICAL FEATURES (INPUT B)
    # -------------------------------------------------------------------------
    print(f"\n[STEP 2] Running Math Physics Model on raw sensor signals to compute 16 inputs...")
    t0_math = time.time()
    extractor = Clinical16FeatureExtractor(fs=100.0)

    computed_features_list = []
    for i in range(N_trials):
        raw_trial = X_raw_seq[i]
        thigh_gyro_y = raw_trial[:, 4]
        shin_gyro_y  = raw_trial[:, 10]

        # Calculate continuous knee angle via relative gyro integration + drift filter
        knee_angle = extractor.compute_knee_angle(thigh_gyro_y, shin_gyro_y)

        # Calculate the 16 clinical biomechanical features
        _, feat_vec = extractor.extract_16_features(knee_angle)
        computed_features_list.append(feat_vec)

    X_computed_feat = np.array(computed_features_list, dtype=np.float32) # (709, 16)
    t_math_total = time.time() - t0_math

    print(f"  [OK] Math computation complete in {t_math_total:.2f} seconds ({t_math_total*1000/N_trials:.2f} ms per trial)!")
    print(f"  [OK] Generated 16 Clinical Features Matrix: {X_computed_feat.shape}")
    print(f"    - ROM range: [{np.min(X_computed_feat[:,0]):.1f} deg, {np.max(X_computed_feat[:,0]):.1f} deg]")
    print(f"    - SPARC smoothness range: [{np.min(X_computed_feat[:,5]):.2f}, {np.max(X_computed_feat[:,5]):.2f}]")

    # -------------------------------------------------------------------------
    # STEP 3: NORMALIZE & PAIR DUAL-STREAM INPUTS FOR HYBRID MODELS
    # -------------------------------------------------------------------------
    print(f"\n[STEP 3] Normalizing and pairing Dual-Stream inputs for Hybrid Model...")

    scaler = StandardScaler()
    X_feat_norm = scaler.fit_transform(X_computed_feat)

    X_seq_norm = np.zeros_like(X_raw_seq)
    for c in range(12):
        c_mean = np.mean(X_raw_seq[:, :, c])
        c_std  = np.std(X_raw_seq[:, :, c]) + 1e-6
        X_seq_norm[:, :, c] = (X_raw_seq[:, :, c] - c_mean) / c_std

    Y_class_cat = keras.utils.to_categorical(y_class, num_classes=4)
    Y_qual_cat  = keras.utils.to_categorical(y_qual, num_classes=3)

    print("  [OK] Input A: Normalized 12-channel sequences ready (shape: 709, 200, 12)")
    print("  [OK] Input B: Normalized 16 clinical features ready (shape: 709, 16)")

    # -------------------------------------------------------------------------
    # STEP 4: TRAIN & EVALUATE HYBRID MODEL ON PAIRED INPUTS
    # -------------------------------------------------------------------------
    models_map = {
        "1": ("Hybrid 1: 1D-CNN + Kinematics (Edge Model)", build_hybrid_1_cnn),
        "2": ("Hybrid 2: BiLSTM + Kinematics (Benchmark Winner)", build_hybrid_2_bilstm),
        "3": ("Hybrid 3: CNN-BiLSTM + Kinematics (AKIRA Core)", build_hybrid_3_cnn_bilstm),
        "4": ("Hybrid 4: 1D-ResNet + Kinematics", build_hybrid_4_resnet),
        "5": ("Hybrid 5: Self-Attention + Kinematics", build_hybrid_5_self_attention),
        "6": ("Hybrid 6: Gated Attention + Kinematics", build_hybrid_6_gated_attention)
    }

    model_title, builder_fn = models_map.get(str(selected_model), models_map["1"])
    print(f"\n[STEP 4] Training {model_title} on Dual-Stream Inputs...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    strata = y_class * 10 + y_qual
    train_idx, test_idx = next(skf.split(X_seq_norm, strata))

    model = builder_fn()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={"exercise_class": "categorical_crossentropy", "quality_grade": "categorical_crossentropy"},
        loss_weights={"exercise_class": 0.5, "quality_grade": 0.5},
        metrics={"exercise_class": "accuracy", "quality_grade": "accuracy"}
    )

    history = model.fit(
        {"input_a_raw_seq": X_seq_norm[train_idx], "input_b_clinical_feat": X_feat_norm[train_idx]},
        {"exercise_class": Y_class_cat[train_idx], "quality_grade": Y_qual_cat[train_idx]},
        validation_data=(
            {"input_a_raw_seq": X_seq_norm[test_idx], "input_b_clinical_feat": X_feat_norm[test_idx]},
            {"exercise_class": Y_class_cat[test_idx], "quality_grade": Y_qual_cat[test_idx]}
        ),
        epochs=num_epochs,
        batch_size=32,
        verbose=1
    )

    preds = model.predict(
        {"input_a_raw_seq": X_seq_norm[test_idx], "input_b_clinical_feat": X_feat_norm[test_idx]},
        verbose=0
    )
    pred_c = np.argmax(preds[0], axis=1)
    pred_q = np.argmax(preds[1], axis=1)

    acc_c = accuracy_score(y_class[test_idx], pred_c) * 100.0
    f1_c  = f1_score(y_class[test_idx], pred_c, average="macro", zero_division=0) * 100.0
    acc_q = accuracy_score(y_qual[test_idx], pred_q) * 100.0
    f1_q  = f1_score(y_qual[test_idx], pred_q, average="macro", zero_division=0) * 100.0

    print("\n" + "=" * 60)
    print("TEST EVALUATION RESULTS ON HELD-OUT DATA (20%):")
    print("=" * 60)
    print(f"  * Task 1 (Exercise Recognition): Accuracy = {acc_c:.2f}%, Macro F1 = {f1_c:.2f}%")
    print(f"  * Task 2 (Movement Quality Grade): Accuracy = {acc_q:.2f}%, Macro F1 = {f1_q:.2f}%")
    print(f"  * Combined Performance Score:    {(0.5*acc_c + 0.5*acc_q):.2f}%")

    # -------------------------------------------------------------------------
    # STEP 5: SIMULATE 1 LIVE PATIENT TRIAL INFERENCE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[STEP 5] LIVE INFERENCE DEMO: SIMULATING 1 NEW PATIENT WEARING SENSORS")
    print("=" * 80)

    demo_idx = test_idx[0]
    single_raw_seq = X_raw_seq[demo_idx]

    print(f"Patient performs movement: Streaming 200 raw samples (2.0s @ 100 Hz)...")

    t0_single = time.time()
    demo_angle = extractor.compute_knee_angle(single_raw_seq[:, 4], single_raw_seq[:, 10])
    demo_dict, demo_feat = extractor.extract_16_features(demo_angle)
    t_math_single = (time.time() - t0_single) * 1000.0

    demo_seq_norm = (single_raw_seq - np.mean(X_raw_seq, axis=(0,1))) / (np.std(X_raw_seq, axis=(0,1)) + 1e-6)
    demo_feat_norm = scaler.transform(demo_feat.reshape(1, -1))

    t0_infer = time.time()
    pred_single = model.predict(
        {"input_a_raw_seq": demo_seq_norm.reshape(1, 200, 12), "input_b_clinical_feat": demo_feat_norm},
        verbose=0
    )
    t_infer_single = (time.time() - t0_infer) * 1000.0

    out_class_idx = int(np.argmax(pred_single[0][0]))
    out_qual_idx  = int(np.argmax(pred_single[1][0]))
    conf_class    = pred_single[0][0][out_class_idx] * 100.0
    conf_qual     = pred_single[1][0][out_qual_idx] * 100.0

    print(f"\nExecution Latencies:")
    print(f"  * Math Model (16 features): {t_math_single:.2f} ms")
    print(f"  * AI Model Inference:       {t_infer_single:.2f} ms")
    print(f"  * Total Pipeline Latency:   {t_math_single + t_infer_single:.2f} ms (Real-Time Edge Capable)")

    print(f"\nFinal Clinical Assessment:")
    print(f"  * Predicted Exercise:  {EXERCISE_NAMES[out_class_idx]} ({conf_class:.1f}% confidence)")
    print(f"  * Predicted Quality:   {QUALITY_NAMES[out_qual_idx]} ({conf_qual:.1f}% confidence)")
    print(f"  * Measured ROM:        {demo_dict['rom_deg']:.1f} deg")
    print(f"  * SPARC Smoothness:    {demo_dict['sparc_smoothness']:.2f}")
    print(f"  * Ground Truth:        {EXERCISE_NAMES[y_class[demo_idx]]} | {QUALITY_NAMES[y_qual[demo_idx]]}")

    if out_class_idx == y_class[demo_idx] and out_qual_idx == y_qual[demo_idx]:
        print("  [OK] PREDICTION PERFECTLY MATCHES GROUND TRUTH!")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AKIRA Master End-to-End Pipeline")
    parser.add_argument("--model", type=str, default="1", help="Model: 1 (1D-CNN), 2 (BiLSTM), 3 (CNN-BiLSTM), 4 (ResNet), 5 (Attention), 6 (Gated)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    args = parser.parse_args()

    run_master_pipeline(selected_model=args.model, num_epochs=args.epochs)
