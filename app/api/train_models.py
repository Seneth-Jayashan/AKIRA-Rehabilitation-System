"""
========================================================================================
AKIRA KNEE REHABILITATION: HYBRID MODELS TRAINING & BENCHMARKING SCRIPT
========================================================================================
Supports training all 6 models or individual models with custom hyperparameters:
  --model [1-6 or all]
  --epochs [int, default 35]
  --batch_size [int, default 32]
  --patience [int, default 6]
  --split [stratified | subject]
  --verbose [0 | 1 | 2]
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
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from hybrid_models import (
    build_hybrid_1_cnn, build_hybrid_2_bilstm, build_hybrid_3_cnn_bilstm,
    build_hybrid_4_resnet, build_hybrid_5_self_attention, build_hybrid_6_gated_attention
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def main():
    parser = argparse.ArgumentParser(description="AKIRA Knee Rehabilitation: Train Dual-Stream Hybrid Models")
    parser.add_argument("--model", type=str, default="all",
                        help="Model to train: 'all', '1' (1D-CNN), '2' (BiLSTM), '3' (CNN-BiLSTM), '4' (1D-ResNet), '5' (Self-Attention), '6' (Gated Attention)")
    parser.add_argument("--epochs", type=int, default=35, help="Maximum epochs per fold (default: 35)")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (default: 6)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--split", type=str, default="stratified", choices=["stratified", "subject"],
                        help="CV split: 'stratified' (trial-level) or 'subject' (GroupKFold across 60 subjects)")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                        help="Keras training verbosity: 0 (silent), 1 (progress bar), 2 (one line per epoch)")
    args = parser.parse_args()

    print("=" * 75)
    print("AKIRA KNEE REHABILITATION: DUAL-INPUT HYBRID MODELS TRAINING")
    print(f"Framework: TensorFlow {tf.__version__} / Keras Functional API")
    print(f"Split Strategy: {args.split.upper()} (5-Fold)")
    print(f"Hyperparameters: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.lr}, Patience={args.patience}")
    print("Inputs: Dual-Stream [Input A: Raw 200x12] + [Input B: 16 Clinical Kinematics]")
    print("Outputs: Dual-Task  [Head 1: 4-Class Exercise] + [Head 2: 3-Tier Quality]")
    print("=" * 75)

    dataset_path = os.path.join(DATASETS_DIR, "dataset_3_final_combined.npz")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset 3 not found at {dataset_path}")
        return

    data = np.load(dataset_path)
    X_seq = data["input_A_raw_sequence"]       # (709, 200, 12)
    X_feat = data["input_B_clinical_features"] # (709, 16)
    y_class = data["y_exercise_class"]         # (709,)
    y_qual = data["y_quality_grade"]           # (709,)

    N = len(y_class)
    print(f"\nLoaded {N} total trials from {dataset_path}.")

    # Normalize Input B (16 clinical features) using StandardScaler
    scaler = StandardScaler()
    X_feat_norm = scaler.fit_transform(X_feat)

    # Normalize Input A (per-channel standardization across time)
    X_seq_norm = np.zeros_like(X_seq)
    for c in range(12):
        c_mean = np.mean(X_seq[:, :, c])
        c_std = np.std(X_seq[:, :, c]) + 1e-6
        X_seq_norm[:, :, c] = (X_seq[:, :, c] - c_mean) / c_std

    # One-hot encode targets
    Y_class_cat = keras.utils.to_categorical(y_class, num_classes=4)
    Y_qual_cat  = keras.utils.to_categorical(y_qual, num_classes=3)

    class_names = ["Walking", "Seated Knee Extension", "Mini Squat", "Step-Up"]
    qual_names  = ["Good (Correct Form)", "Needs Improvement (Lag/Deficit)", "Bad / Compensatory Fault"]

    all_model_builders = {
        "Hybrid 1: 1D-CNN": build_hybrid_1_cnn,
        "Hybrid 2: BiLSTM": build_hybrid_2_bilstm,
        "Hybrid 3: CNN-BiLSTM (AKIRA)": build_hybrid_3_cnn_bilstm,
        "Hybrid 4: 1D-ResNet": build_hybrid_4_resnet,
        "Hybrid 5: Self-Attention": build_hybrid_5_self_attention,
        "Hybrid 6: Gated Attention": build_hybrid_6_gated_attention
    }

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

    # Cross-Validation setup
    strata = y_class * 10 + y_qual
    if args.split == "subject":
        catalog_path = os.path.join(DATASETS_DIR, "dataset_catalog.csv")
        catalog_df = pd.read_csv(catalog_path)
        groups = catalog_df["subject"].values
        gkf = GroupKFold(n_splits=5)
        folds = list(gkf.split(X_seq_norm, strata, groups=groups))
        print("Split Mode: Subject-Independent (GroupKFold across 60 subjects, 12 unseen test subjects per fold).")
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(skf.split(X_seq_norm, strata))
        print("Split Mode: Stratified 5-Fold (80% train trials / 20% test trials per fold).")

    benchmark_records = []

    for name, builder_fn in model_builders.items():
        print(f"\n" + "-" * 75)
        print(f"BENCHMARKING MODEL: {name}")
        print("-" * 75)

        sample_model = builder_fn()
        num_params = sample_model.count_params()
        print(f"Total Parameters: {num_params:,}")

        fold_acc_c, fold_f1_c = [], []
        fold_acc_q, fold_f1_q = [], []
        t0 = time.time()

        for fold, (train_idx, val_idx) in enumerate(folds, 1):
            X_seq_tr, X_seq_val = X_seq_norm[train_idx], X_seq_norm[val_idx]
            X_feat_tr, X_feat_val = X_feat_norm[train_idx], X_feat_norm[val_idx]

            Y_c_tr, Y_c_val = Y_class_cat[train_idx], Y_class_cat[val_idx]
            Y_q_tr, Y_q_val = Y_qual_cat[train_idx], Y_qual_cat[val_idx]

            model = builder_fn()
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=args.lr),
                loss={"exercise_class": "categorical_crossentropy", "quality_grade": "categorical_crossentropy"},
                loss_weights={"exercise_class": 0.5, "quality_grade": 0.5},
                metrics={"exercise_class": "accuracy", "quality_grade": "accuracy"}
            )

            early_stop = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=0
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

            preds = model.predict(
                {"input_a_raw_seq": X_seq_val, "input_b_clinical_feat": X_feat_val},
                verbose=0
            )
            pred_class = np.argmax(preds[0], axis=1)
            pred_qual  = np.argmax(preds[1], axis=1)
            true_class = y_class[val_idx]
            true_qual  = y_qual[val_idx]

            fold_acc_c.append(accuracy_score(true_class, pred_class))
            fold_f1_c.append(f1_score(true_class, pred_class, average="macro", zero_division=0))
            fold_acc_q.append(accuracy_score(true_qual, pred_qual))
            fold_f1_q.append(f1_score(true_qual, pred_qual, average="macro", zero_division=0))

        train_time = time.time() - t0
        mean_acc_c = float(np.mean(fold_acc_c))
        mean_f1_c  = float(np.mean(fold_f1_c))
        mean_acc_q = float(np.mean(fold_acc_q))
        mean_f1_q  = float(np.mean(fold_f1_q))
        combined_score = 0.5 * mean_acc_c + 0.5 * mean_acc_q

        print(f"\nResults for {name}:")
        print(f"  -> Exercise Recognition: Accuracy = {mean_acc_c*100:.2f}%, F1 = {mean_f1_c*100:.2f}%")
        print(f"  -> Quality Assessment:   Accuracy = {mean_acc_q*100:.2f}%, F1 = {mean_f1_q*100:.2f}%")
        print(f"  -> Combined Score:       {combined_score*100:.2f}% | Total Time: {train_time:.1f}s")

if __name__ == "__main__":
    main()
