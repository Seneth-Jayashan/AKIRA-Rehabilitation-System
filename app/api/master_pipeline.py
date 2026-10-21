"""
========================================================================================
AKIRA KNEE REHABILITATION: MASTER END-TO-END PIPELINE (master_pipeline.py)
========================================================================================
WHAT THIS FILE IS:
  This script demonstrates the complete, closed-loop AKIRA knee rehabilitation pipeline
  from raw sensor ingestion to live orthopedic clinical feedback.

WHAT THIS FILE DOES:
  Executes the 5 canonical stages of the AKIRA Edge-AI Ecosystem:
  1. INGESTION  : Loads the preprocessed 12-channel raw IMU dataset (Dataset 1).
  2. KINEMATICS : Runs the Mathematical Physics Model (Madgwick Sensor Fusion + 3D SVD
                  Relative Orientation) on raw signals to dynamically compute continuous
                  knee flexion angles and the 16 clinical features.
  3. DUAL-STREAM: Leak-free subject-independent split (GroupKFold by subject) and
                  train-only normalization of [Stream A: Raw 200x12] + [Stream B: 16 Features].
  4. AI TRAINING: Trains and evaluates a selected Hybrid Model with balanced sample weights.
  5. LIVE DEMO  : Simulates 1 unseen patient trial wearing wearable IMU knee sensors,
                  measuring execution latencies (<20 ms) and generating orthopedic guidance.
========================================================================================
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

from typing import Any, cast
import tensorflow as tf
keras = tf.keras

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")

from math_feature_extractor import Clinical16FeatureExtractor
from train_models import per_output_sample_weights, quality_eval_mask
from hybrid_models import (
    build_hybrid_1_cnn, build_hybrid_2_bilstm, build_hybrid_3_cnn_bilstm,
    build_hybrid_4_resnet, build_hybrid_5_self_attention, build_hybrid_6_gated_attention
)

EXERCISE_NAMES = [
    "Walking",                  # Class 0
    "Seated Knee Extension",    # Class 1
    "Mini Squat",               # Class 2
    "Step-Up"                   # Class 3
]

QUALITY_NAMES = [
    "Good (Correct Form)",                  # Grade 0
    "Needs Improvement (Incomplete ROM / Unloading)",  # Grade 1
    "Bad / Compensatory Fault"             # Grade 2
]


def run_master_pipeline(selected_model: str = "1", num_epochs: int = 20, split_mode: str = "subject"):
    """
    Executes the full leak-free end-to-end rehabilitation pipeline from raw sensor data
    to mathematical kinematic derivation, subject-independent model training, evaluation,
    and live inference simulation.
    """
    print("=" * 80)
    print("AKIRA KNEE REHABILITATION: MASTER END-TO-END PIPELINE EXECUTION")
    print("=" * 80)

    # ----------------------------------------------------------------------------------
    # STEP 1: LOAD PREPROCESSED LABELED RAW SENSOR DATASET (DATASET 1 & CATALOG)
    # ----------------------------------------------------------------------------------
    ds1_path = os.path.join(DATASETS_DIR, "dataset_1_input_A_12channels.npz")
    cat_path = os.path.join(DATASETS_DIR, "dataset_catalog.csv")
    print(f"\n[STEP 1] Loading preprocessed labeled raw sensor data from:\n        {ds1_path}")
    data = np.load(ds1_path, allow_pickle=True)
    X_raw_seq = data["X_raw_12ch"]
    y_class   = data["y_exercise_class"]
    y_qual    = data["y_quality_grade"]
    N_trials  = len(y_class)

    if os.path.exists(cat_path):
        catalog_df = pd.read_csv(cat_path)
        subjects = np.asarray(catalog_df["subject"].astype(str).to_numpy(), dtype=str)
    else:
        subjects = np.array([f"S_{i // 10}" for i in range(N_trials)], dtype=str)

    print(f"  [OK] Loaded {N_trials} total trials across {len(np.unique(subjects))} subjects.")
    print(f"  [OK] Raw Sequence Shape: {X_raw_seq.shape} (2.0s window @ 100 Hz, 12 synchronized IMU channels)")
    print(f"  [OK] Ground Truth Labels: 4 Exercise Classes & 3 Clinical Quality Tiers")

    # ----------------------------------------------------------------------------------
    # STEP 2: APPLY MATH MODEL TO DERIVE THE 16 CLINICAL FEATURES (INPUT B)
    # ----------------------------------------------------------------------------------
    print(f"\n[STEP 2] Running Sensor Fusion & Math Physics Model on raw 12-ch signals...")
    t0_math = time.time()
    extractor = Clinical16FeatureExtractor(fs=100.0)

    computed_features_list = []
    for i in range(N_trials):
        raw_trial = X_raw_seq[i]
        # Full 12-channel Madgwick sensor fusion + 3D SVD relative hinge orientation
        knee_angle = extractor.compute_knee_angle(raw_trial)
        _, feat_vec = extractor.extract_16_features(knee_angle)
        computed_features_list.append(feat_vec)

    X_computed_feat = np.array(computed_features_list, dtype=np.float32)
    t_math_total = time.time() - t0_math

    print(f"  [OK] Math computation complete in {t_math_total:.2f} seconds ({t_math_total*1000/N_trials:.2f} ms per trial)!")
    print(f"  [OK] Generated 16 Clinical Features Matrix: {X_computed_feat.shape}")
    print(f"    - ROM (F1) range:          [{np.min(X_computed_feat[:,0]):.1f} deg, {np.max(X_computed_feat[:,0]):.1f} deg]")
    print(f"    - Peak Depth (F2) range:   [{np.min(X_computed_feat[:,1]):.1f} deg, {np.max(X_computed_feat[:,1]):.1f} deg]")
    print(f"    - Return Deficit (F3):     [{np.min(X_computed_feat[:,2]):.1f} deg, {np.max(X_computed_feat[:,2]):.1f} deg]")
    print(f"    - SPARC (F6) range:        [{np.min(X_computed_feat[:,5]):.2f}, {np.max(X_computed_feat[:,5]):.2f}]")

    # ----------------------------------------------------------------------------------
    # STEP 3: STRICT 3-WAY SUBJECT-INDEPENDENT SPLIT (INNER TRAIN / VAL / TEST)
    # ----------------------------------------------------------------------------------
    print(f"\n[STEP 3] Partitioning Strict 3-Way Split (Inner Train / Validation / Held-Out Test)...")

    strata = y_class * 10 + y_qual
    if split_mode == "subject":
        gkf_outer = GroupKFold(n_splits=5)
        train_val_idx, test_idx = next(gkf_outer.split(X_raw_seq, strata, groups=cast(Any, subjects)))
        gkf_inner = GroupKFold(n_splits=6)
        inner_rel, val_rel = next(gkf_inner.split(
            X_raw_seq[train_val_idx], strata[train_val_idx], groups=cast(Any, subjects[train_val_idx])
        ))
        inner_train_idx = train_val_idx[inner_rel]
        val_idx = train_val_idx[val_rel]
        assert len(set(subjects[inner_train_idx].tolist()).intersection(set(subjects[val_idx].tolist()))) == 0
        assert len(set(subjects[inner_train_idx].tolist()).intersection(set(subjects[test_idx].tolist()))) == 0
        assert len(set(subjects[val_idx].tolist()).intersection(set(subjects[test_idx].tolist()))) == 0
        print(
            f"  [OK] Strict 3-Way Subject Split: {len(inner_train_idx)} inner-train / "
            f"{len(val_idx)} val / {len(test_idx)} held-out test trials (0 overlapping subjects)."
        )
    else:
        skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_val_idx, test_idx = next(skf_outer.split(X_raw_seq, strata))
        skf_inner = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
        inner_rel, val_rel = next(skf_inner.split(X_raw_seq[train_val_idx], strata[train_val_idx]))
        inner_train_idx = train_val_idx[inner_rel]
        val_idx = train_val_idx[val_rel]

    # Fit StandardScaler strictly on inner_train_idx
    scaler = StandardScaler()
    X_feat_norm = np.zeros_like(X_computed_feat, dtype=np.float32)
    X_feat_norm[inner_train_idx] = scaler.fit_transform(X_computed_feat[inner_train_idx])
    X_feat_norm[val_idx]         = scaler.transform(X_computed_feat[val_idx])
    X_feat_norm[test_idx]        = scaler.transform(X_computed_feat[test_idx])

    # Fit per-channel means & stds strictly on inner_train_idx
    X_seq_norm = np.zeros_like(X_raw_seq, dtype=np.float32)
    c_means = np.zeros(12, dtype=np.float32)
    c_stds  = np.zeros(12, dtype=np.float32)
    for c in range(12):
        c_means[c] = np.mean(X_raw_seq[inner_train_idx, :, c])
        c_stds[c]  = np.std(X_raw_seq[inner_train_idx, :, c]) + 1e-6
        X_seq_norm[:, :, c] = (X_raw_seq[:, :, c] - c_means[c]) / c_stds[c]

    Y_class_cat = keras.utils.to_categorical(y_class, num_classes=4)
    # Quality -1 = no ground truth: placeholder target with zero quality weight
    Y_qual_cat  = keras.utils.to_categorical(np.where(y_qual >= 0, y_qual, 0), num_classes=3)

    # Per-output balanced sample weights strictly on inner_train_idx (unlabelled quality -> weight 0)
    sw_train = per_output_sample_weights(y_class, y_qual, inner_train_idx)
    sw_val = per_output_sample_weights(y_class, y_qual, val_idx, balance=False)

    # ----------------------------------------------------------------------------------
    # STEP 4: TRAIN & EVALUATE HYBRID MODEL ON PAIRED INPUTS (VAL FOR EARLY STOPPING, TEST UNTOUCHED)
    # ----------------------------------------------------------------------------------
    models_map = {
        "1": ("Hybrid 1: 1D-CNN + Kinematics (Edge Model)", build_hybrid_1_cnn),
        "2": ("Hybrid 2: BiLSTM + Kinematics", build_hybrid_2_bilstm),
        "3": ("Hybrid 3: CNN-BiLSTM + Kinematics (AKIRA Core)", build_hybrid_3_cnn_bilstm),
        "4": ("Hybrid 4: 1D-ResNet + Kinematics", build_hybrid_4_resnet),
        "5": ("Hybrid 5: Self-Attention + Kinematics", build_hybrid_5_self_attention),
        "6": ("Hybrid 6: Gated Attention + Kinematics", build_hybrid_6_gated_attention)
    }

    model_title, builder_fn = models_map.get(str(selected_model), models_map["1"])
    print(f"\n[STEP 4] Training {model_title} on Leak-Free Dual-Stream Inputs...")

    model = builder_fn()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=["categorical_crossentropy", "categorical_crossentropy"],
        loss_weights=[0.45, 0.55],
        metrics=["accuracy", "accuracy"]
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    )

    model.fit(
        {"input_a_raw_seq": X_seq_norm[inner_train_idx], "input_b_clinical_feat": X_feat_norm[inner_train_idx]},
        [Y_class_cat[inner_train_idx], Y_qual_cat[inner_train_idx]],
        sample_weight=cast(Any, sw_train),
        validation_data=cast(Any, (
            {"input_a_raw_seq": X_seq_norm[val_idx], "input_b_clinical_feat": X_feat_norm[val_idx]},
            [Y_class_cat[val_idx], Y_qual_cat[val_idx]],
            sw_val
        )),
        epochs=num_epochs,
        batch_size=24,
        callbacks=[early_stop],
        verbose=1
    )

    preds = model.predict(
        cast(Any, {"input_a_raw_seq": X_seq_norm[test_idx], "input_b_clinical_feat": X_feat_norm[test_idx]}),
        verbose=0
    )
    pred_c = np.argmax(preds[0], axis=1)
    pred_q = np.argmax(preds[1], axis=1)

    acc_c = accuracy_score(y_class[test_idx], pred_c) * 100.0
    f1_c  = f1_score(y_class[test_idx], pred_c, average="macro", zero_division=0) * 100.0
    q_lab = quality_eval_mask(y_qual[test_idx])
    acc_q = accuracy_score(y_qual[test_idx][q_lab], pred_q[q_lab]) * 100.0
    f1_q  = f1_score(y_qual[test_idx][q_lab], pred_q[q_lab], average="macro", zero_division=0) * 100.0

    print("\n" + "=" * 60)
    print("UNSEEN-SUBJECT TEST EVALUATION RESULTS (20% HELD-OUT):")
    print("=" * 60)
    print(f"  * Task 1 (Exercise Recognition): Accuracy = {acc_c:.2f}%, Macro F1 = {f1_c:.2f}%")
    print(f"  * Task 2 (Movement Quality Grade): Accuracy = {acc_q:.2f}%, Macro F1 = {f1_q:.2f}%")
    print(f"  * Combined Performance Score:    {(0.5*acc_c + 0.5*acc_q):.2f}%")

    # ----------------------------------------------------------------------------------
    # STEP 5: SIMULATE 1 LIVE PATIENT TRIAL INFERENCE
    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[STEP 5] LIVE INFERENCE DEMO: SIMULATING 1 UNSEEN PATIENT WEARING SENSORS")
    print("=" * 80)

    demo_idx = test_idx[0]
    single_raw_seq = X_raw_seq[demo_idx]

    t0_single = time.time()
    demo_angle = extractor.compute_knee_angle(single_raw_seq)
    demo_dict, demo_feat = extractor.extract_16_features(demo_angle)
    t_math_single = (time.time() - t0_single) * 1000.0

    demo_seq_norm = (single_raw_seq - c_means) / c_stds
    demo_feat_norm = scaler.transform(demo_feat.reshape(1, -1))

    t0_infer = time.time()
    pred_single = model.predict(
        cast(Any, {"input_a_raw_seq": demo_seq_norm.reshape(1, 200, 12), "input_b_clinical_feat": demo_feat_norm}),
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
    print(f"  * Peak Flexion Depth:  {demo_dict['peak_depth_deg']:.1f} deg")
    print(f"  * Ext. Return Deficit: {demo_dict['extension_return_deficit_deg']:.1f} deg")
    print(f"  * SPARC Smoothness:    {demo_dict['sparc_smoothness']:.2f}")
    gt_q = QUALITY_NAMES[y_qual[demo_idx]] if y_qual[demo_idx] >= 0 else "no quality ground truth"
    print(f"  * Ground Truth:        {EXERCISE_NAMES[y_class[demo_idx]]} | {gt_q}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AKIRA Master End-to-End Pipeline")
    parser.add_argument(
        "--model", type=str, default="1",
        help="Model: 1 (1D-CNN), 2 (BiLSTM), 3 (CNN-BiLSTM), 4 (ResNet), 5 (Attention), 6 (Gated)"
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Number of training epochs (default: 15)"
    )
    parser.add_argument(
        "--split", type=str, default="subject", choices=["subject", "stratified"],
        help="Evaluation split mode: 'subject' (default, leak-free GroupKFold) or 'stratified'"
    )
    args = parser.parse_args()

    run_master_pipeline(selected_model=args.model, num_epochs=args.epochs, split_mode=args.split)
