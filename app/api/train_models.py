"""
========================================================================================
AKIRA KNEE REHABILITATION: LEAK-FREE HYBRID MODELS TRAINING, 5-FOLD CV & TFLITE BENCHMARK
(app/api/train_models.py)
========================================================================================
WHAT THIS SCRIPT DOES:
  1. Loads `dataset_3_final_combined.npz` and `dataset_catalog.csv`.
  2. Supports `--split subject` (`GroupKFold` by subject, default) and `--split stratified`
     (`StratifiedKFold`).
  3. Partitions subjects into THREE strictly disjoint subject sets (Fixes New #19):
     - `inner_train_idx` (~68% of subjects): fits `StandardScaler` and channel `means`/`stds`
       and trains model weights.
     - `val_idx` (~12% of subjects): drives `EarlyStopping(monitor='val_loss')` and selects
       the best model checkpoint — NEVER touching the test set!
     - `test_idx` (~20% of subjects): 100% held-out unseen test set evaluated ONLY after
       all training and model selection are complete.
  4. Runs full 5-Fold Cross-Validation across all 5 folds (Fixes New #20) and reports
     `5-Fold CV Mean +/- SD` alongside held-out test metrics.
  5. Evaluates two Proposal §3.3 Baselines (`Random Forest 16-Feat` & `Single-IMU Thigh CNN`).
  6. Exports a pure `TFLITE_BUILTINS` (ZERO `SELECT_TF_OPS`!) quantized TensorFlow Lite model
     (`akira_hybrid_cnn_edge.tflite`) and writes the COMPLETE model byte array (not a preview)
     to `edge_akira_classifier.c` for ESP32 TFLite Micro deployment (Fixes Still open #9).
========================================================================================
"""

import os
import sys
import time
import json
import argparse
from typing import Any, cast, Optional
import joblib
import numpy as np
import pandas as pd

import tensorflow as tf
keras = tf.keras
layers = tf.keras.layers

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

from hybrid_models import (
    build_hybrid_1_cnn, build_hybrid_2_bilstm, build_hybrid_3_cnn_bilstm,
    build_hybrid_4_resnet, build_hybrid_5_self_attention, build_hybrid_6_gated_attention
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DOCS_DIR = os.path.join(os.path.dirname(BASE_DIR), "docs")

EXERCISE_NAMES = ["Walking", "Seated Knee Extension", "Mini Squat", "Step-Up"]
QUALITY_NAMES = ["Good (Correct Form)", "Needs Improvement (Incomplete ROM / Unloading)", "Bad / Compensatory Fault"]


def build_single_imu_baseline(seq_shape=(200, 6), num_classes=4, num_quals=3):
    """Proposal §3.3 Baseline: Single-IMU (Thigh-only 6 channels) 1D-CNN."""
    in_seq = keras.Input(shape=seq_shape, name="thigh_only_6ch")
    x = layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(in_seq)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out_c = layers.Dense(num_classes, activation="softmax", name="exercise_class")(x)
    out_q = layers.Dense(num_quals, activation="softmax", name="quality_grade")(x)
    return keras.Model(inputs=in_seq, outputs=[out_c, out_q], name="Single_IMU_Thigh_CNN")


def fit_train_normalizers(X_seq_train: np.ndarray, X_feat_train: np.ndarray):
    """Fits channel means/stds and feature StandardScaler STRICTLY on inner training indices."""
    scaler = StandardScaler()
    scaler.fit(X_feat_train)
    scaler_mean = np.asarray(scaler.mean_, dtype=np.float32) if scaler.mean_ is not None else np.zeros(X_feat_train.shape[1], dtype=np.float32)
    scaler_scale = np.asarray(scaler.scale_, dtype=np.float32) if scaler.scale_ is not None else np.ones(X_feat_train.shape[1], dtype=np.float32)

    means = np.mean(X_seq_train, axis=(0, 1)).astype(np.float32)
    stds = (np.std(X_seq_train, axis=(0, 1)) + 1e-6).astype(np.float32)
    return means, stds, scaler_mean, scaler_scale


def apply_normalizers(X_seq: np.ndarray, X_feat: np.ndarray, means, stds, scaler_mean, scaler_scale):
    X_seq_norm = ((X_seq - means) / (stds + 1e-6)).astype(np.float32)
    X_feat_norm = ((X_feat - scaler_mean) / (scaler_scale + 1e-6)).astype(np.float32)
    return X_seq_norm, X_feat_norm


def per_output_sample_weights(y_class: np.ndarray, y_qual: np.ndarray, idx: np.ndarray, balance: bool = True) -> Any:
    """
    Returns `[w_exercise, w_quality]` for `idx`. Windows without a quality label
    (y_qual == -1, datasets with no correctness annotation) get quality weight 0, so they
    train the exercise head only. With `balance`, weights are inverse-frequency (clipped).
    """
    yc = y_class[idx]
    yq = y_qual[idx]
    labelled = yq >= 0
    w_c = np.clip(compute_sample_weight("balanced", yc), 0.35, 3.5) if balance else np.ones(len(idx))
    w_q = np.zeros(len(idx), dtype=np.float64)
    if np.any(labelled):
        w_q[labelled] = np.clip(compute_sample_weight("balanced", yq[labelled]), 0.35, 3.5) if balance else 1.0
    return [w_c.astype(np.float32), w_q.astype(np.float32)]


def quality_eval_mask(y_qual_subset: np.ndarray) -> np.ndarray:
    """Quality metrics are computed only on windows that have a ground-truth quality label."""
    return np.asarray(y_qual_subset) >= 0


def make_5fold_splits(df_cat: pd.DataFrame, y_class: np.ndarray, y_qual: np.ndarray, split_mode: str = "subject"):
    """
    Creates 5 folds using either `GroupKFold` by subject (`split_mode == 'subject'`)
    or `StratifiedKFold` (`split_mode == 'stratified'`).
    Returns `(folds, heldout_fold_idx)`. The held-out fold is fixed in advance (fold 0) and
    is never chosen by looking at its class or grade composition; all 5 folds are also
    reported through cross-validation.
    """
    strata = y_class * 10 + y_qual
    if split_mode == "stratified":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(skf.split(np.zeros(len(y_class)), strata))
    else:
        subjects = np.asarray(df_cat["subject"].values)
        gkf = GroupKFold(n_splits=5)
        folds = list(gkf.split(np.zeros(len(y_class)), strata, groups=subjects))

    return folds, 0


def split_train_into_inner_train_and_val(train_idx: np.ndarray, df_cat: pd.DataFrame, y_class: np.ndarray, y_qual: np.ndarray, split_mode: str = "subject"):
    """
    Splits `train_idx` into disjoint `inner_train_idx` (~85% of train) and `val_idx` (~15% of train)
    so early stopping and model selection NEVER use `test_idx` (Fixes New #19).
    """
    strata_tr = y_class[train_idx] * 10 + y_qual[train_idx]
    if split_mode == "stratified":
        skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
        inner_tr_rel, val_rel = next(skf.split(np.zeros(len(train_idx)), strata_tr))
    else:
        sub_tr = np.asarray(df_cat.iloc[train_idx]["subject"].values)
        gkf = GroupKFold(n_splits=6)
        inner_tr_rel, val_rel = next(gkf.split(np.zeros(len(train_idx)), strata_tr, groups=sub_tr))

    return train_idx[inner_tr_rel], train_idx[val_rel]


def evaluate_tflite_edge_model(keras_model, X_seq_test_norm, X_feat_test_norm, y_c_test, y_q_test, pred_c_keras, pred_q_keras,
                               X_seq_calib=None, X_feat_calib=None):
    """
    Converts `hybrid_1_cnn` to a full-integer (int8) TensorFlow Lite model calibrated on
    training windows (TFLite Micro on the ESP32 needs int8 kernels; float I/O is kept via
    QUANTIZE/DEQUANTIZE builtins), exports the complete byte array to `edge_akira_classifier.c`,
    and benchmarks the host-PC interpreter. On-device latency is NOT estimated: it can only
    be measured once the firmware runs TFLite Micro.
    """
    tflite_path = os.path.join(MODELS_DIR, "akira_hybrid_cnn_edge.tflite")
    c_path = os.path.join(MODELS_DIR, "edge_akira_classifier.c")

    input_names = [inp.name for inp in keras_model.inputs]

    if X_seq_calib is None or X_feat_calib is None:
        has_calib = False
        calib_seq = np.empty((0, 200, 12), dtype=np.float32)
        calib_feat = np.empty((0, 16), dtype=np.float32)
    else:
        has_calib = True
        calib_seq = np.asarray(X_seq_calib, dtype=np.float32)
        calib_feat = np.asarray(X_feat_calib, dtype=np.float32)

    def representative_dataset():
        n_cal = min(300, len(calib_seq))
        for i in range(n_cal):
            yield [
                (calib_seq[i:i + 1] if "input_a" in name else calib_feat[i:i + 1]).astype(np.float32)
                for name in input_names
            ]

    quantization_mode = "full_int8"
    try:
        if not has_calib:
            raise ValueError("no calibration data supplied")
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        tflite_bytes = converter.convert()
    except Exception as exc:
        print(f"  [WARN] Full-int8 conversion failed ({exc}); falling back to dynamic-range quantization.")
        quantization_mode = "dynamic_range_fallback"
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_bytes = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)

    size_kb = round(len(tflite_bytes) / 1024.0, 2)

    # Export COMPLETE C byte array (every single byte of `tflite_bytes`, not a 512-byte preview!)
    lines_hex = []
    for offset in range(0, len(tflite_bytes), 16):
        chunk = tflite_bytes[offset : offset + 16]
        lines_hex.append("  " + ", ".join(f"0x{b:02x}" for b in chunk))
    full_hex_body = ",\n".join(lines_hex)

    c_code = f"""/*
 * AKIRA Edge-AI TFLite Model Byte Array (Auto-Generated)
 * Quantization: {quantization_mode} | Operator set: TFLite builtins only (no Flex/SELECT_TF_OPS)
 * Total Full Model Size: {len(tflite_bytes)} bytes ({size_kb} KB)
 * Note: the ESP32 firmware does not yet link TFLite Micro; this array is ready for it.
 */
#include <stdint.h>

const unsigned int akira_edge_tflite_len = {len(tflite_bytes)};
alignas(16) const uint8_t akira_edge_tflite_model[{len(tflite_bytes)}] = {{
{full_hex_body}
}};
"""
    with open(c_path, "w", encoding="utf-8") as f:
        f.write(c_code)

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    # Verify the quantization actually achieved: count float32 weight/activation tensors
    # (model inputs/outputs are float by design and are excluded)
    io_indices = {d["index"] for d in in_details} | {d["index"] for d in out_details}
    float_internal = [
        t["name"] for t in interpreter.get_tensor_details()
        if t["index"] not in io_indices and t["dtype"] == np.float32
    ]
    is_full_int8 = (quantization_mode == "full_int8") and len(float_internal) == 0

    seq_in_idx = in_details[0]["index"] if len(in_details[0]["shape"]) == 3 else in_details[1]["index"]
    feat_in_idx = in_details[1]["index"] if len(in_details[0]["shape"]) == 3 else in_details[0]["index"]

    c_out_idx = out_details[0]["index"] if out_details[0]["shape"][-1] == 4 else out_details[1]["index"]
    q_out_idx = out_details[1]["index"] if out_details[0]["shape"][-1] == 4 else out_details[0]["index"]

    tflite_preds_c, tflite_preds_q = [], []
    latencies_ms = []

    for i in range(len(X_seq_test_norm)):
        s_in = X_seq_test_norm[i : i + 1].astype(np.float32)
        f_in = X_feat_test_norm[i : i + 1].astype(np.float32)
        t0 = time.perf_counter()
        interpreter.set_tensor(seq_in_idx, s_in)
        interpreter.set_tensor(feat_in_idx, f_in)
        interpreter.invoke()
        out_c = interpreter.get_tensor(c_out_idx)[0]
        out_q = interpreter.get_tensor(q_out_idx)[0]
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        tflite_preds_c.append(int(np.argmax(out_c)))
        tflite_preds_q.append(int(np.argmax(out_q)))

    tflite_preds_c = np.array(tflite_preds_c)
    tflite_preds_q = np.array(tflite_preds_q)

    agree_c = round(float(np.mean(tflite_preds_c == pred_c_keras) * 100.0), 2)
    agree_q = round(float(np.mean(tflite_preds_q == pred_q_keras) * 100.0), 2)
    acc_c = round(float(accuracy_score(y_c_test, tflite_preds_c) * 100.0), 2)
    q_lab = np.asarray(y_q_test) >= 0
    acc_q = round(float(accuracy_score(np.asarray(y_q_test)[q_lab], tflite_preds_q[q_lab]) * 100.0), 2) if np.any(q_lab) else 0.0

    host_mean_ms = round(float(np.mean(latencies_ms)), 3)
    host_p95_ms = round(float(np.percentile(latencies_ms, 95)), 3)

    return {
        "tflite_filename": "models/akira_hybrid_cnn_edge.tflite",
        "c_array_filename": "models/edge_akira_classifier.c",
        "c_array_bytes_written": int(len(tflite_bytes)),
        "ops_set": "TFLITE_BUILTINS_INT8" if quantization_mode == "full_int8" else "TFLITE_BUILTINS",
        "quantization_mode": quantization_mode,
        "full_int8_verified": bool(is_full_int8),
        "float_internal_tensor_count": int(len(float_internal)),
        "size_kb": size_kb,
        "mean_latency_ms": host_mean_ms,
        "p95_latency_ms": host_p95_ms,
        "host_pc_interpreter_mean_latency_ms": host_mean_ms,
        "on_device_latency_ms": None,
        "on_device_latency_note": "Not measured: the ESP32 firmware does not yet run TFLite Micro.",
        "exercise_acc_pct": acc_c,
        "quality_acc_pct": acc_q,
        "keras_agreement_exercise_pct": agree_c,
        "keras_agreement_quality_pct": agree_q
    }


def main():
    parser = argparse.ArgumentParser(description="AKIRA Knee Rehabilitation: Leak-Free Model Training & 5-Fold CV")
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--split", type=str, default="subject", choices=["subject", "stratified"])
    parser.add_argument("--cv_epochs", type=int, default=18, help="Epochs per fold during 5-fold cross-validation")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)
    tf.random.set_seed(42)
    np.random.seed(42)

    dataset_path = os.path.join(DATASETS_DIR, "dataset_3_final_combined.npz")
    catalog_path = os.path.join(DATASETS_DIR, "dataset_catalog.csv")
    data = np.load(dataset_path)
    df_cat = pd.read_csv(catalog_path)

    X_seq = data["input_A_raw_sequence"].astype(np.float32)
    X_feat = data["input_B_clinical_features"].astype(np.float32)
    y_class = data["y_exercise_class"].astype(np.int32)
    y_qual = data["y_quality_grade"].astype(np.int32)

    Y_class_cat = keras.utils.to_categorical(y_class, num_classes=4)
    # Unlabelled quality (-1) gets a placeholder one-hot target; its quality weight is 0
    Y_qual_cat = keras.utils.to_categorical(np.where(y_qual >= 0, y_qual, 0), num_classes=3)

    # 1. Create 5-Fold Split according to --split (`subject` vs `stratified`)
    folds, heldout_fold_idx = make_5fold_splits(df_cat, y_class, y_qual, split_mode=args.split)
    train_pool_idx, test_idx = folds[heldout_fold_idx]

    # 2. Split train_pool_idx into disjoint `inner_train_idx` and `val_idx` so test_idx is NEVER used for validation!
    inner_train_idx, val_idx = split_train_into_inner_train_and_val(
        train_pool_idx, df_cat, y_class, y_qual, split_mode=args.split
    )

    inner_train_subjects = set(df_cat.iloc[inner_train_idx]["subject"].unique())
    val_subjects = set(df_cat.iloc[val_idx]["subject"].unique())
    test_subjects = set(df_cat.iloc[test_idx]["subject"].unique())

    if args.split == "subject":
        assert len(inner_train_subjects.intersection(test_subjects)) == 0, "Train/Test subject overlap!"
        assert len(val_subjects.intersection(test_subjects)) == 0, "Val/Test subject overlap!"
        assert len(inner_train_subjects.intersection(val_subjects)) == 0, "InnerTrain/Val subject overlap!"

    print("=" * 80)
    print(f"AKIRA KNEE REHABILITATION: LEAK-FREE 3-WAY ({args.split.upper()}) SPLIT TRAINING")
    print(f"Total Trials: {len(y_class)} | Inner Train: {len(inner_train_idx)} ({len(inner_train_subjects)} subj) | "
          f"Val: {len(val_idx)} ({len(val_subjects)} subj) | Unseen Test: {len(test_idx)} ({len(test_subjects)} subj)")
    print("=" * 80)

    # 3. Fit Scalers STRICTLY on `inner_train_idx` Only
    means, stds, scaler_mean, scaler_scale = fit_train_normalizers(X_seq[inner_train_idx], X_feat[inner_train_idx])
    X_seq_tr, X_feat_tr = apply_normalizers(X_seq[inner_train_idx], X_feat[inner_train_idx], means, stds, scaler_mean, scaler_scale)
    X_seq_val, X_feat_val = apply_normalizers(X_seq[val_idx], X_feat[val_idx], means, stds, scaler_mean, scaler_scale)
    X_seq_te, X_feat_te = apply_normalizers(X_seq[test_idx], X_feat[test_idx], means, stds, scaler_mean, scaler_scale)

    # Save Unseen Held-Out 20% Test Dataset & Catalog
    test_npz_path = os.path.join(DATASETS_DIR, "test_20percent_dataset.npz")
    test_cat_path = os.path.join(DATASETS_DIR, "test_20percent_catalog.csv")
    np.savez_compressed(
        test_npz_path,
        X_seq_raw=X_seq[test_idx],
        X_feat_raw=X_feat[test_idx],
        y_class=y_class[test_idx],
        y_qual=y_qual[test_idx],
        means=means,
        stds=stds,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale
    )
    df_cat.iloc[test_idx].reset_index(drop=True).to_csv(test_cat_path, index=False)
    print(f"[OK] Saved unseen held-out test set ({len(test_idx)} trials) to {test_npz_path}")

    # Per-output balanced sample weights; unlabelled-quality windows get quality weight 0
    sw_tr = per_output_sample_weights(y_class, y_qual, inner_train_idx)
    sw_val = per_output_sample_weights(y_class, y_qual, val_idx, balance=False)
    q_mask_te = quality_eval_mask(y_qual[test_idx])
    tr_q_lab = quality_eval_mask(y_qual[inner_train_idx])
    print(f"[QUALITY LABELS] Labelled windows: train {int(tr_q_lab.sum())}/{len(inner_train_idx)} | "
          f"test {int(q_mask_te.sum())}/{len(test_idx)} (the rest have no ground-truth quality)")

    # =========================================================================
    # 4. TRAIN & EVALUATE BASELINES (Using `val_idx` for validation, `test_idx` for test)
    # =========================================================================
    print("\n[BASELINES] Evaluating Conventional Random Forest (16 Features) & Single-IMU (Thigh 6-Ch) CNN...")
    rf_ex = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42)
    rf_qu = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42)
    rf_ex.fit(X_feat_tr, y_class[inner_train_idx])
    rf_qu.fit(X_feat_tr[tr_q_lab], y_qual[inner_train_idx][tr_q_lab])
    joblib.dump(rf_ex, os.path.join(MODELS_DIR, "akira_rf_exercise.joblib"))
    joblib.dump(rf_qu, os.path.join(MODELS_DIR, "akira_rf_quality.joblib"))

    rf_pred_c = rf_ex.predict(X_feat_te)
    rf_pred_q = rf_qu.predict(X_feat_te)
    rf_acc_c = float(accuracy_score(y_class[test_idx], rf_pred_c))
    rf_f1_c = float(f1_score(y_class[test_idx], rf_pred_c, average="macro", zero_division=0))
    rf_acc_q = float(accuracy_score(y_qual[test_idx][q_mask_te], rf_pred_q[q_mask_te]))
    rf_f1_q = float(f1_score(y_qual[test_idx][q_mask_te], rf_pred_q[q_mask_te], average="macro", zero_division=0))

    single_imu_model = build_single_imu_baseline()
    single_imu_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=["categorical_crossentropy", "categorical_crossentropy"],
        loss_weights=[0.5, 0.5],
        metrics=["accuracy", "accuracy"]
    )
    single_imu_model.fit(
        X_seq_tr[:, :, 0:6],
        [Y_class_cat[inner_train_idx], Y_qual_cat[inner_train_idx]],
        sample_weight=cast(Any, sw_tr),
        validation_data=(X_seq_val[:, :, 0:6], [Y_class_cat[val_idx], Y_qual_cat[val_idx]], cast(Any, sw_val)),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)],
        verbose=0
    )
    s_preds = single_imu_model.predict(X_seq_te[:, :, 0:6], verbose=0)
    s_pred_c = np.argmax(s_preds[0], axis=1)
    s_pred_q = np.argmax(s_preds[1], axis=1)
    s_acc_c = float(accuracy_score(y_class[test_idx], s_pred_c))
    s_f1_c = float(f1_score(y_class[test_idx], s_pred_c, average="macro", zero_division=0))
    s_acc_q = float(accuracy_score(y_qual[test_idx][q_mask_te], s_pred_q[q_mask_te]))
    s_f1_q = float(f1_score(y_qual[test_idx][q_mask_te], s_pred_q[q_mask_te], average="macro", zero_division=0))

    baselines_summary = {
        "random_forest_16feat": {
            "title": "Baseline 1: Random Forest (16 Clinical Features Only)",
            "exercise_acc": round(rf_acc_c, 4),
            "exercise_f1": round(rf_f1_c, 4),
            "quality_acc": round(rf_acc_q, 4),
            "quality_f1": round(rf_f1_q, 4),
            "combined_score": round(0.5 * (rf_acc_c + rf_acc_q), 4)
        },
        "single_imu_thigh_cnn": {
            "title": "Baseline 2: Single-IMU 1D-CNN (Thigh 6-Ch Only)",
            "params": int(single_imu_model.count_params()),
            "exercise_acc": round(s_acc_c, 4),
            "exercise_f1": round(s_f1_c, 4),
            "quality_acc": round(s_acc_q, 4),
            "quality_f1": round(s_f1_q, 4),
            "combined_score": round(0.5 * (s_acc_c + s_acc_q), 4)
        }
    }
    print(f"  -> Baseline 1 (Random Forest 16-Feat): Ex Acc={rf_acc_c*100:.2f}%, Qual Acc={rf_acc_q*100:.2f}%, Combined={(0.5*(rf_acc_c+rf_acc_q))*100:.2f}%")
    print(f"  -> Baseline 2 (Single-IMU Thigh CNN) : Ex Acc={s_acc_c*100:.2f}%, Qual Acc={s_acc_q*100:.2f}%, Combined={(0.5*(s_acc_c+s_acc_q))*100:.2f}%")

    # =========================================================================
    # 5. TRAIN & EVALUATE ALL 6 HYBRID MODELS (Validation on `val_idx`, Test on `test_idx`)
    # =========================================================================
    all_model_specs = [
        ("hybrid_1_cnn", "Hybrid 1: 1D-CNN + Kinematics", build_hybrid_1_cnn),
        ("hybrid_2_bilstm", "Hybrid 2: BiLSTM + Kinematics", build_hybrid_2_bilstm),
        ("hybrid_3_cnn_bilstm", "Hybrid 3: CNN-BiLSTM + Kinematics", build_hybrid_3_cnn_bilstm),
        ("hybrid_4_resnet", "Hybrid 4: 1D-ResNet + Kinematics", build_hybrid_4_resnet),
        ("hybrid_5_self_attention", "Hybrid 5: Self-Attention + Kinematics", build_hybrid_5_self_attention),
        ("hybrid_6_gated_attention", "Hybrid 6: Gated Attention + Kinematics", build_hybrid_6_gated_attention)
    ]
    if args.model != "all":
        model_specs = [m for m in all_model_specs if m[0] == args.model]
    else:
        model_specs = all_model_specs

    metrics_summary = {}
    best_val_score = -1.0
    best_model_obj = None
    cnn_edge_model_obj = None
    cnn_edge_preds = None

    for m_key, m_title, builder_fn in model_specs:
        print(f"\n[TRAINING] {m_title} ({m_key})...")
        tf.random.set_seed(42)
        model = builder_fn()
        num_params = int(model.count_params())

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr),
            loss=["categorical_crossentropy", "categorical_crossentropy"],
            loss_weights=[0.5, 0.5],
            metrics=["accuracy", "accuracy"]
        )

        # EarlyStopping monitors `val_loss` on `val_idx` ONLY (never `test_idx`!)
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=0
        )

        t0 = time.time()
        model.fit(
            {"input_a_raw_seq": X_seq_tr, "input_b_clinical_feat": X_feat_tr},  # type: ignore[arg-type]
            [Y_class_cat[inner_train_idx], Y_qual_cat[inner_train_idx]],
            sample_weight=cast(Any, sw_tr),
            validation_data=(
                {"input_a_raw_seq": X_seq_val, "input_b_clinical_feat": X_feat_val},  # type: ignore[arg-type]
                [Y_class_cat[val_idx], Y_qual_cat[val_idx]],
                cast(Any, sw_val)
            ),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[early_stop],
            verbose=args.verbose
        )
        train_time_s = round(time.time() - t0, 2)

        rel_model_path = f"models/{m_key}.keras"
        abs_model_path = os.path.join(MODELS_DIR, f"{m_key}.keras")
        model.save(abs_model_path)

        # Compute validation score on `val_idx` to select `best_akira_hybrid_model.keras` without touching `test_idx`
        val_preds = model.predict({"input_a_raw_seq": X_seq_val, "input_b_clinical_feat": X_feat_val}, verbose=0)  # type: ignore[arg-type]
        val_acc_c = float(accuracy_score(y_class[val_idx], np.argmax(val_preds[0], axis=1)))
        q_mask_val = quality_eval_mask(y_qual[val_idx])
        val_acc_q = float(accuracy_score(y_qual[val_idx][q_mask_val], np.argmax(val_preds[1], axis=1)[q_mask_val])) if np.any(q_mask_val) else 0.0
        val_comb = 0.5 * (val_acc_c + val_acc_q)

        # Now evaluate on the untouched `test_idx`
        t_inf0 = time.perf_counter()
        preds = model.predict({"input_a_raw_seq": X_seq_te, "input_b_clinical_feat": X_feat_te}, verbose=0)  # type: ignore[arg-type]
        mean_infer_ms = round(((time.perf_counter() - t_inf0) * 1000.0) / len(test_idx), 2)

        pred_c = np.argmax(preds[0], axis=1)
        pred_q_all = np.argmax(preds[1], axis=1)
        true_c = y_class[test_idx]
        # Quality is evaluated only on test windows that have a ground-truth quality label
        true_q = y_qual[test_idx][q_mask_te]
        pred_q = pred_q_all[q_mask_te]

        # Check of the 0-100 quality score E[Q] = 100*P(0) + 50*P(1) + 0*P(2) against grade targets (100, 50, 0)
        expected_q_score = 100.0 * preds[1][q_mask_te, 0] + 50.0 * preds[1][q_mask_te, 1]
        true_q_target = 100.0 - 50.0 * true_q
        q_score_mae = round(float(np.mean(np.abs(expected_q_score - true_q_target))), 2)

        acc_c = float(accuracy_score(true_c, pred_c))
        f1_c = float(f1_score(true_c, pred_c, average="macro", zero_division=0))
        acc_q = float(accuracy_score(true_q, pred_q))
        f1_q = float(f1_score(true_q, pred_q, average="macro", zero_division=0))
        comb = 0.5 * (acc_c + acc_q)

        cm_c = confusion_matrix(true_c, pred_c, labels=[0, 1, 2, 3]).tolist()
        cm_q = confusion_matrix(true_q, pred_q, labels=[0, 1, 2]).tolist()

        rep_c_dict = classification_report(true_c, pred_c, labels=[0, 1, 2, 3], target_names=EXERCISE_NAMES, output_dict=True, zero_division=0)
        rep_q_dict = classification_report(true_q, pred_q, labels=[0, 1, 2], target_names=QUALITY_NAMES, output_dict=True, zero_division=0)

        metrics_summary[m_key] = {
            "key": m_key,
            "title": m_title,
            "params": num_params,
            "split_mode": f"{args.split}_independent_3way_train_val_test",
            "train_subjects_count": len(inner_train_subjects),
            "val_subjects_count": len(val_subjects),
            "test_subjects_count": len(test_subjects),
            "test_trials_count": int(len(test_idx)),
            "test_quality_labelled_count": int(q_mask_te.sum()),
            "val_combined_score": round(val_comb, 4),
            "exercise_acc": round(acc_c, 4),
            "exercise_f1": round(f1_c, 4),
            "quality_acc": round(acc_q, 4),
            "quality_f1": round(f1_q, 4),
            "combined_score": round(comb, 4),
            "calibrated_quality_score_mae_0_100": q_score_mae,
            "mean_inference_ms": mean_infer_ms,
            "train_time_s": train_time_s,
            "cm_exercise": cm_c,
            "cm_quality": cm_q,
            "per_class_exercise": rep_c_dict,
            "per_class_quality": rep_q_dict,
            "model_path": rel_model_path,
            "baselines": baselines_summary,
            "test_pred_c": pred_c.tolist(),
            "test_pred_q": pred_q_all.tolist()
        }

        print(f"  -> Val Combined={val_comb*100:.2f}% | Unseen Test: Ex Acc={acc_c*100:.2f}% (F1={f1_c*100:.2f}%) | Qual Acc={acc_q*100:.2f}% (F1={f1_q*100:.2f}%) | Combined={comb*100:.2f}%")

        if m_key == "hybrid_1_cnn":
            cnn_edge_model_obj = model
            cnn_edge_preds = (pred_c, pred_q_all)

        if val_comb > best_val_score:
            best_val_score = val_comb
            best_model_obj = model

    if best_model_obj is not None:
        best_model_obj.save(os.path.join(MODELS_DIR, "best_akira_hybrid_model.keras"))

    # =========================================================================
    # 6. RUN GENUINE 5-FOLD CROSS-VALIDATION ACROSS ALL 5 FOLDS (Fixes New #20)
    # =========================================================================
    print(f"\n[5-FOLD CV] Executing all 5 folds ({args.split} split) on Hybrid 1 (1D-CNN + Kinematics)...")
    cv_ex_accs, cv_ex_f1s, cv_qu_accs, cv_qu_f1s = [], [], [], []
    for fold_i, (f_tr_pool, f_te) in enumerate(folds):
        f_in_tr, f_val = split_train_into_inner_train_and_val(f_tr_pool, df_cat, y_class, y_qual, split_mode=args.split)
        f_means, f_stds, f_smean, f_sscale = fit_train_normalizers(X_seq[f_in_tr], X_feat[f_in_tr])
        X_s_tr, X_f_tr = apply_normalizers(X_seq[f_in_tr], X_feat[f_in_tr], f_means, f_stds, f_smean, f_sscale)
        X_s_va, X_f_va = apply_normalizers(X_seq[f_val], X_feat[f_val], f_means, f_stds, f_smean, f_sscale)
        X_s_te, X_f_te = apply_normalizers(X_seq[f_te], X_feat[f_te], f_means, f_stds, f_smean, f_sscale)

        f_sw = per_output_sample_weights(y_class, y_qual, f_in_tr)
        f_sw_val = per_output_sample_weights(y_class, y_qual, f_val, balance=False)

        tf.random.set_seed(42 + fold_i)
        cv_m = build_hybrid_1_cnn()
        cv_m.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr),
            loss=["categorical_crossentropy", "categorical_crossentropy"],
            loss_weights=[0.5, 0.5],
            metrics=["accuracy", "accuracy"]
        )
        cv_m.fit(
            {"input_a_raw_seq": X_s_tr, "input_b_clinical_feat": X_f_tr},  # type: ignore[arg-type]
            [Y_class_cat[f_in_tr], Y_qual_cat[f_in_tr]],
            sample_weight=cast(Any, f_sw),
            validation_data=({"input_a_raw_seq": X_s_va, "input_b_clinical_feat": X_f_va}, [Y_class_cat[f_val], Y_qual_cat[f_val]], cast(Any, f_sw_val)),  # type: ignore[arg-type]
            epochs=args.cv_epochs,
            batch_size=args.batch_size,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
            verbose=0
        )
        f_preds = cv_m.predict({"input_a_raw_seq": X_s_te, "input_b_clinical_feat": X_f_te}, verbose=0)  # type: ignore[arg-type]
        f_pc = np.argmax(f_preds[0], axis=1)
        f_pq = np.argmax(f_preds[1], axis=1)
        fa_c = float(accuracy_score(y_class[f_te], f_pc))
        ff_c = float(f1_score(y_class[f_te], f_pc, average="macro", zero_division=0))
        f_qm = quality_eval_mask(y_qual[f_te])
        fa_q = float(accuracy_score(y_qual[f_te][f_qm], f_pq[f_qm])) if np.any(f_qm) else float("nan")
        ff_q = float(f1_score(y_qual[f_te][f_qm], f_pq[f_qm], average="macro", zero_division=0)) if np.any(f_qm) else float("nan")
        cv_ex_accs.append(fa_c); cv_ex_f1s.append(ff_c); cv_qu_accs.append(fa_q); cv_qu_f1s.append(ff_q)
        print(f"  Fold {fold_i+1}/5: Ex Acc={fa_c*100:.2f}% (F1={ff_c*100:.2f}%) | Qual Acc={fa_q*100:.2f}% (F1={ff_q*100:.2f}%)")

    cv_5fold_summary = {
        "split_mode": args.split,
        "folds_executed": 5,
        "exercise_acc_mean": round(float(np.mean(cv_ex_accs)), 4),
        "exercise_acc_std": round(float(np.std(cv_ex_accs)), 4),
        "exercise_f1_mean": round(float(np.mean(cv_ex_f1s)), 4),
        "quality_acc_mean": round(float(np.nanmean(cv_qu_accs)), 4),
        "quality_acc_std": round(float(np.nanstd(cv_qu_accs)), 4),
        "quality_f1_mean": round(float(np.nanmean(cv_qu_f1s)), 4),
        "fold_exercise_accs": [round(x, 4) for x in cv_ex_accs],
        "fold_quality_accs": [round(x, 4) for x in cv_qu_accs]
    }
    print(f"  -> 5-Fold CV Summary: Ex Acc = {cv_5fold_summary['exercise_acc_mean']*100:.2f}% +/- {cv_5fold_summary['exercise_acc_std']*100:.2f}% | "
          f"Qual Acc = {cv_5fold_summary['quality_acc_mean']*100:.2f}% +/- {cv_5fold_summary['quality_acc_std']*100:.2f}%")

    # =========================================================================
    # 7. CONVERT & BENCHMARK FULL-INT8 TFLITE EDGE MODEL (calibrated on training windows only)
    # =========================================================================
    if cnn_edge_model_obj is not None and cnn_edge_preds is not None:
        print("\n[EDGE AI] Converting & Benchmarking Full-Int8 TFLite Model...")
        tflite_stats = evaluate_tflite_edge_model(
            cnn_edge_model_obj, X_seq_te, X_feat_te,
            y_class[test_idx], y_qual[test_idx],
            cnn_edge_preds[0], cnn_edge_preds[1],
            X_seq_calib=X_seq_tr, X_feat_calib=X_feat_tr
        )
        print(f"  -> TFLite Size: {tflite_stats['size_kb']} KB ({tflite_stats['c_array_bytes_written']} bytes in C array) | "
            f"Quantization: {tflite_stats['quantization_mode']} (full int8 verified: {tflite_stats['full_int8_verified']}) | "
            f"Host PC Latency: {tflite_stats['mean_latency_ms']} ms | On-device latency: not measured")
    else:
        tflite_stats = {}

    for k in metrics_summary:
        metrics_summary[k]["tflite_edge_validation"] = tflite_stats
        metrics_summary[k]["cv_5fold_summary"] = cv_5fold_summary

    metrics_json_path = os.path.join(MODELS_DIR, "test_metrics_summary.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\n[OK] Saved complete leak-free metrics summary to {metrics_json_path}")

    # Auto-generate synchronized HYBRID_MODELS_BENCHMARK.md
    sorted_models = sorted(metrics_summary.values(), key=lambda x: x["combined_score"], reverse=True)
    winner = sorted_models[0]

    md_lines = [
        "# AKIRA 6 All-Hybrid Multi-Modal Models Benchmark Report",
        "",
        "**Component:** Knee Rehabilitation Assessment Following Tibial Plateau ORIF  ",
        f"**Framework:** TensorFlow {tf.__version__} / Keras Functional API  ",
        f"**Validation Protocol:** 3-Way Subject-Independent Split ({len(inner_train_idx)} Inner Train [{len(inner_train_subjects)} Subj] / {len(val_idx)} Validation [{len(val_subjects)} Subj] / {len(test_idx)} Unseen Test [{len(test_subjects)} Subj]) + Full 5-Fold Cross-Validation  ",
        "",
        "---",
        "",
        "## 1. 6-Model Benchmark Leaderboard (Unseen Subjects Held-Out Test Set)",
        "",
        "| Rank | Model Architecture | Parameters | Val Score (%) | Exercise Acc (%) | Exercise Macro F1 (%) | Quality Acc (%) | Quality Macro F1 (%) | Combined Score (%) | Host PC Inference Latency |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for idx_r, m in enumerate(sorted_models, 1):
        md_lines.append(
            f"| **#{idx_r}** | **{m['title']}** | {m['params']:,} | {m['val_combined_score']*100:.2f}% | **{m['exercise_acc']*100:.2f}%** | {m['exercise_f1']*100:.2f}% | "
            f"**{m['quality_acc']*100:.2f}%** | {m['quality_f1']*100:.2f}% | **{m['combined_score']*100:.2f}%** | {m['mean_inference_ms']} ms |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. 5-Fold Subject-Independent Cross-Validation Summary (All 5 Folds Executed)",
        "",
        f"- **Exercise Accuracy (5-Fold Mean ± SD):** **{cv_5fold_summary['exercise_acc_mean']*100:.2f}% ± {cv_5fold_summary['exercise_acc_std']*100:.2f}%** (Macro-F1: **{cv_5fold_summary['exercise_f1_mean']*100:.2f}%**)",
        f"- **Quality Accuracy (5-Fold Mean ± SD):** **{cv_5fold_summary['quality_acc_mean']*100:.2f}% ± {cv_5fold_summary['quality_acc_std']*100:.2f}%** (Macro-F1: **{cv_5fold_summary['quality_f1_mean']*100:.2f}%**)",
        f"- **Per-Fold Exercise Accuracies:** `{[round(x*100, 2) for x in cv_5fold_summary['fold_exercise_accs']]}`",
        f"- **Per-Fold Quality Accuracies:** `{[round(x*100, 2) for x in cv_5fold_summary['fold_quality_accs']]}`",
        "",
        "---",
        "",
        "## 3. Baseline Comparison (Proposal §3.3 Validation)",
        "",
        "| Baseline Model | Modality | Exercise Acc (%) | Exercise Macro F1 (%) | Quality Acc (%) | Quality Macro F1 (%) | Combined Score (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| **{baselines_summary['random_forest_16feat']['title']}** | Input B Only (16 Features) | {baselines_summary['random_forest_16feat']['exercise_acc']*100:.2f}% | {baselines_summary['random_forest_16feat']['exercise_f1']*100:.2f}% | {baselines_summary['random_forest_16feat']['quality_acc']*100:.2f}% | {baselines_summary['random_forest_16feat']['quality_f1']*100:.2f}% | **{baselines_summary['random_forest_16feat']['combined_score']*100:.2f}%** |",
        f"| **{baselines_summary['single_imu_thigh_cnn']['title']}** | Single IMU (Thigh 6-Ch Only) | {baselines_summary['single_imu_thigh_cnn']['exercise_acc']*100:.2f}% | {baselines_summary['single_imu_thigh_cnn']['exercise_f1']*100:.2f}% | {baselines_summary['single_imu_thigh_cnn']['quality_acc']*100:.2f}% | {baselines_summary['single_imu_thigh_cnn']['quality_f1']*100:.2f}% | **{baselines_summary['single_imu_thigh_cnn']['combined_score']*100:.2f}%** |",
        f"| **{winner['title']} (Winning Dual-IMU Hybrid)** | Dual-IMU (12-Ch) + 16 Features | **{winner['exercise_acc']*100:.2f}%** | **{winner['exercise_f1']*100:.2f}%** | **{winner['quality_acc']*100:.2f}%** | **{winner['quality_f1']*100:.2f}%** | **{winner['combined_score']*100:.2f}%** |",
        "",
        "---",
        "",
        "## 4. Edge-AI TensorFlow Lite (`.tflite`) Deployment Validation (Proposal §3.5)",
        "",
        f"- **Quantized Edge Model File:** `api/models/akira_hybrid_cnn_edge.tflite` (**{tflite_stats.get('size_kb', 0.0)} KB**)",
        f"- **Quantization:** `{tflite_stats.get('quantization_mode', 'unknown')}` (full int8 verified: **{tflite_stats.get('full_int8_verified', False)}**, float internal tensors: {tflite_stats.get('float_internal_tensor_count', 'n/a')})",
        f"- **Complete Embedded C Byte Array:** `api/models/edge_akira_classifier.c` (**{tflite_stats.get('c_array_bytes_written', 0):,} bytes** full model array)",
        f"- **Host PC TFLite Interpreter Latency:** **{tflite_stats.get('host_pc_interpreter_mean_latency_ms', 0.0)} ms** (95th Percentile: **{tflite_stats.get('p95_latency_ms', 0.0)} ms**)",
        f"- **On-Device (ESP32) Latency:** {tflite_stats.get('on_device_latency_note', 'Not measured: the ESP32 firmware does not yet run TFLite Micro.')}",
        f"- **TFLite vs. Full Keras Agreement:** **{tflite_stats.get('keras_agreement_exercise_pct', 0.0)}%** (Exercise) / **{tflite_stats.get('keras_agreement_quality_pct', 0.0)}%** (Quality)",
        ""
    ])

    with open(os.path.join(DOCS_DIR, "HYBRID_MODELS_BENCHMARK.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[OK] Synchronized benchmark documentation in {os.path.join(DOCS_DIR, 'HYBRID_MODELS_BENCHMARK.md')}")


if __name__ == "__main__":
    main()
