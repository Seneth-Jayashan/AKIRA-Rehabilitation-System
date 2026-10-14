"""
========================================================================================
AKIRA KNEE REHABILITATION: HYBRID MODELS TRAINING & BENCHMARKING SCRIPT (train_models.py)
========================================================================================
WHAT THIS FILE IS:
  This script trains, validates, and benchmarks the 6 Dual-Stream Hybrid Deep Learning
  models on the combined dataset using 5-Fold Cross-Validation.

WHAT THIS FILE DOES:
  1. Parses command-line hyperparameters (model choice, epochs, batch size, learning rate).
  2. Loads the multi-modal clinical dataset:
     - Input A: Raw 12-channel IMU time series (709 trials, 200 time steps @ 100 Hz).
     - Input B: 16 Clinical Biomechanical Features (ROM, Depth, Lag, SPARC Smoothness, etc.).
     - Target 1: Exercise Class (4 classes).
     - Target 2: Movement Quality Grade (3 tiers).
  3. Prepares dual-stream normalization:
     - Z-score per-channel normalization across temporal steps for raw signals.
     - StandardScaler normalization for the 16 clinical kinematics.
  4. Implements rigorous 5-Fold Cross-Validation:
     - Option A: Stratified 5-Fold (Trial-level stratification across class + quality).
     - Option B: Subject-Independent GroupKFold (Train on 48 subjects, test on 12 unseen subjects).
  5. Multi-task loss compilation (50% Exercise Classification + 50% Quality Assessment).
  6. Evaluates and reports Accuracy and Macro F1-scores across all folds.

CLI ARGUMENTS:
  --model      : 'all' or '1'..'6' (choose specific model architecture)
  --epochs     : Maximum epochs per fold (default: 35)
  --batch_size : Mini-batch size (default: 32)
  --patience   : Early stopping patience epochs on val_loss (default: 6)
  --lr         : Adam optimizer learning rate (default: 0.001)
  --split      : 'stratified' (trial-level) or 'subject' (GroupKFold by patient)
  --verbose    : Keras output verbosity: 0 (silent), 1 (progress bar), 2 (one line/epoch)

TABLE OF CONTENTS:
  SECTION 1: Imports & Dependencies
  SECTION 2: Directory Configuration
  SECTION 3: CLI Argument Parsing
  SECTION 4: Dataset Loading & Dual-Stream Normalization
  SECTION 5: Cross-Validation Splitting Setup
  SECTION 6: Training & Evaluation Loop Across Folds
  SECTION 7: Benchmark Summary & Reporting
========================================================================================
"""

# ======================================================================================
# SECTION 1: IMPORTS & DEPENDENCIES
# ======================================================================================
import os
import sys
import time
import argparse

# Data manipulation and numerical operations
import numpy as np
import pandas as pd

# Deep Learning framework
import tensorflow as tf
from tensorflow import keras

# Scikit-learn for cross-validation splits, feature scaling, and evaluation metrics
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Import the 6 dual-stream hybrid model architecture builders
from hybrid_models import (
    build_hybrid_1_cnn, build_hybrid_2_bilstm, build_hybrid_3_cnn_bilstm,
    build_hybrid_4_resnet, build_hybrid_5_self_attention, build_hybrid_6_gated_attention
)


# ======================================================================================
# SECTION 2: DIRECTORY CONFIGURATION
# ======================================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ======================================================================================
# SECTION 3: CLI ARGUMENT PARSING
# ======================================================================================
def parse_arguments():
    """Configures command-line interface arguments for customized training runs."""
    parser = argparse.ArgumentParser(
        description="AKIRA Knee Rehabilitation: Train Dual-Stream Hybrid Models"
    )
    parser.add_argument(
        "--model", type=str, default="all",
        help="Model to train: 'all', '1' (1D-CNN), '2' (BiLSTM), '3' (CNN-BiLSTM), '4' (1D-ResNet), '5' (Self-Attention), '6' (Gated Attention)"
    )
    parser.add_argument(
        "--epochs", type=int, default=35,
        help="Maximum epochs per fold (default: 35)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Mini-batch size (default: 32)"
    )
    parser.add_argument(
        "--patience", type=int, default=6,
        help="Early stopping patience epochs on validation loss (default: 6)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Adam optimizer learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--split", type=str, default="stratified", choices=["stratified", "subject"],
        help="CV split strategy: 'stratified' (trial-level) or 'subject' (GroupKFold across 60 subjects)"
    )
    parser.add_argument(
        "--verbose", type=int, default=1, choices=[0, 1, 2],
        help="Keras training verbosity: 0 (silent), 1 (progress bar), 2 (one line per epoch)"
    )
    return parser.parse_args()


# ======================================================================================
# MAIN TRAINING PIPELINE
# ======================================================================================
def main():
    args = parse_arguments()

    print("=" * 80)
    print("AKIRA KNEE REHABILITATION: DUAL-INPUT HYBRID MODELS TRAINING")
    print(f"Framework: TensorFlow {tf.__version__} / Keras Functional API")
    print(f"Split Strategy: {args.split.upper()} (5-Fold)")
    print(f"Hyperparameters: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.lr}, Patience={args.patience}")
    print("Inputs: Dual-Stream [Input A: Raw 200x12] + [Input B: 16 Clinical Kinematics]")
    print("Outputs: Dual-Task  [Head 1: 4-Class Exercise] + [Head 2: 3-Tier Quality]")
    print("=" * 80)

    # ----------------------------------------------------------------------------------
    # SECTION 4: DATASET LOADING & DUAL-STREAM NORMALIZATION
    # ----------------------------------------------------------------------------------
    dataset_path = os.path.join(DATASETS_DIR, "dataset_3_final_combined.npz")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset 3 not found at {dataset_path}")
        return

    data = np.load(dataset_path)
    X_seq = data["input_A_raw_sequence"]       # Shape: (709, 200, 12) - Raw sensor stream
    X_feat = data["input_B_clinical_features"] # Shape: (709, 16) - Clinical kinematic features
    y_class = data["y_exercise_class"]         # Shape: (709,) - Exercise IDs 0..3
    y_qual = data["y_quality_grade"]           # Shape: (709,) - Quality Tiers 0..2

    N = len(y_class)
    print(f"\n[DATA] Loaded {N} total trials from {dataset_path}.")

    # Step 4.1: Standardize Input B (16 clinical features) using StandardScaler
    scaler = StandardScaler()
    X_feat_norm = scaler.fit_transform(X_feat)

    # Step 4.2: Standardize Input A (per-channel Z-score standardization across time)
    X_seq_norm = np.zeros_like(X_seq)
    for c in range(12):
        c_mean = np.mean(X_seq[:, :, c])
        c_std = np.std(X_seq[:, :, c]) + 1e-6
        X_seq_norm[:, :, c] = (X_seq[:, :, c] - c_mean) / c_std

    # Step 4.3: One-hot encode targets for multi-task categorical crossentropy
    Y_class_cat = keras.utils.to_categorical(y_class, num_classes=4)
    Y_qual_cat  = keras.utils.to_categorical(y_qual, num_classes=3)

    class_names = ["Walking", "Seated Knee Extension", "Mini Squat", "Step-Up"]
    qual_names  = ["Good (Correct Form)", "Needs Improvement (Lag/Deficit)", "Bad / Compensatory Fault"]

    # Model architecture directory
    all_model_builders = {
        "Hybrid 1: 1D-CNN": build_hybrid_1_cnn,
        "Hybrid 2: BiLSTM": build_hybrid_2_bilstm,
        "Hybrid 3: CNN-BiLSTM (AKIRA)": build_hybrid_3_cnn_bilstm,
        "Hybrid 4: 1D-ResNet": build_hybrid_4_resnet,
        "Hybrid 5: Self-Attention": build_hybrid_5_self_attention,
        "Hybrid 6: Gated Attention": build_hybrid_6_gated_attention
    }

    # Filter to requested model or train all
    if args.model != "all":
        key_map = {
            "1": "Hybrid 1: 1D-CNN", "cnn": "Hybrid 1: 1D-CNN",
            "2": "Hybrid 2: BiLSTM", "bilstm": "Hybrid 2: BiLSTM",
            "3": "Hybrid 3: CNN-BiLSTM (AKIRA)", "cnn-bilstm": "Hybrid 3: CNN-BiLSTM (AKIRA)",
            "4": "Hybrid 4: 1D-ResNet", "resnet": "Hybrid 4: 1D-ResNet",
            "5": "Hybrid 5: Self-Attention", "attention": "Hybrid 5: Self-Attention",
            "6": "Hybrid 6: Gated Attention", "gated": "Hybrid 6: Gated Attention",
        }
        chosen_key = key_map.get(args.model.lower().strip())
        if not chosen_key:
            print(f"Error: Unknown model '{args.model}'. Choose from 1, 2, 3, 4, 5, 6, or 'all'.")
            return
        model_builders = {chosen_key: all_model_builders[chosen_key]}
    else:
        model_builders = all_model_builders

    # ----------------------------------------------------------------------------------
    # SECTION 5: CROSS-VALIDATION SPLITTING SETUP
    # ----------------------------------------------------------------------------------
    # Create combined stratification target to preserve class + quality distributions
    strata = y_class * 10 + y_qual
    if args.split == "subject":
        # GroupKFold ensures no subject appears in both train and test partitions
        catalog_path = os.path.join(DATASETS_DIR, "dataset_catalog.csv")
        catalog_df = pd.read_csv(catalog_path)
        groups = catalog_df["subject"].values
        gkf = GroupKFold(n_splits=5)
        folds = list(gkf.split(X_seq_norm, strata, groups=groups))
        print("[CV] Split Mode: Subject-Independent (GroupKFold across 60 subjects, 12 unseen test subjects per fold).")
    else:
        # Standard Stratified 5-Fold split (80% train / 20% validation)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(skf.split(X_seq_norm, strata))
        print("[CV] Split Mode: Stratified 5-Fold (80% train trials / 20% test trials per fold).")

    # ----------------------------------------------------------------------------------
    # SECTION 6: TRAINING & EVALUATION LOOP ACROSS FOLDS
    # ----------------------------------------------------------------------------------
    for name, builder_fn in model_builders.items():
        print(f"\n" + "-" * 80)
        print(f"BENCHMARKING MODEL: {name}")
        print("-" * 80)

        # Print model parameter count
        sample_model = builder_fn()
        num_params = sample_model.count_params()
        print(f"Total Parameters: {num_params:,}")

        fold_acc_c, fold_f1_c = [], []
        fold_acc_q, fold_f1_q = [], []
        t0 = time.time()

        for fold, (train_idx, val_idx) in enumerate(folds, 1):
            # Split train and validation sets
            X_seq_tr, X_seq_val = X_seq_norm[train_idx], X_seq_norm[val_idx]
            X_feat_tr, X_feat_val = X_feat_norm[train_idx], X_feat_norm[val_idx]

            Y_c_tr, Y_c_val = Y_class_cat[train_idx], Y_class_cat[val_idx]
            Y_q_tr, Y_q_val = Y_qual_cat[train_idx], Y_qual_cat[val_idx]

            # Build a fresh model instance for this fold
            model = builder_fn()

            # Multi-Task Compilation: Equal 50/50 loss weighting between Head 1 and Head 2
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=args.lr),
                loss={
                    "exercise_class": "categorical_crossentropy",
                    "quality_grade": "categorical_crossentropy"
                },
                loss_weights={"exercise_class": 0.5, "quality_grade": 0.5},
                metrics={"exercise_class": "accuracy", "quality_grade": "accuracy"}
            )

            # Early stopping callback restores the best validation weights
            early_stop = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                restore_best_weights=True,
                verbose=0
            )

            print(f"  [Fold {fold}/5] Training on {len(train_idx)} samples, validating on {len(val_idx)} samples...")
            history = model.fit(
                {"input_a_raw_seq": X_seq_tr, "input_b_clinical_feat": X_feat_tr},
                {"exercise_class": Y_c_tr, "quality_grade": Y_q_tr},
                validation_data=(
                    {"input_a_raw_seq": X_seq_val, "input_b_clinical_feat": X_feat_val},
                    {"exercise_class": Y_c_val, "quality_grade": Y_q_val}
                ),
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=[early_stop],
                verbose=args.verbose
            )

            # Evaluate predictions on the validation fold
            preds = model.predict(
                {"input_a_raw_seq": X_seq_val, "input_b_clinical_feat": X_feat_val},
                verbose=0
            )
            pred_class = np.argmax(preds[0], axis=1)
            pred_qual  = np.argmax(preds[1], axis=1)
            true_class = y_class[val_idx]
            true_qual  = y_qual[val_idx]

            # Store metrics for this fold
            fold_acc_c.append(accuracy_score(true_class, pred_class))
            fold_f1_c.append(f1_score(true_class, pred_class, average="macro", zero_division=0))
            fold_acc_q.append(accuracy_score(true_qual, pred_qual))
            fold_f1_q.append(f1_score(true_qual, pred_qual, average="macro", zero_division=0))

        # ------------------------------------------------------------------------------
        # SECTION 7: BENCHMARK SUMMARY & REPORTING
        # ------------------------------------------------------------------------------
        train_time = time.time() - t0
        mean_acc_c = float(np.mean(fold_acc_c))
        mean_f1_c  = float(np.mean(fold_f1_c))
        mean_acc_q = float(np.mean(fold_acc_q))
        mean_f1_q  = float(np.mean(fold_f1_q))
        combined_score = 0.5 * mean_acc_c + 0.5 * mean_acc_q

        print(f"\nFinal 5-Fold Benchmark Results for {name}:")
        print(f"  -> Exercise Recognition: Accuracy = {mean_acc_c*100:.2f}%, Macro F1 = {mean_f1_c*100:.2f}%")
        print(f"  -> Quality Assessment:   Accuracy = {mean_acc_q*100:.2f}%, Macro F1 = {mean_f1_q*100:.2f}%")
        print(f"  -> Combined Score:       {combined_score*100:.2f}% | Total Time: {train_time:.1f}s")


if __name__ == "__main__":
    main()
