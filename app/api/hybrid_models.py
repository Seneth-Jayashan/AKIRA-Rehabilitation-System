"""
========================================================================================
AKIRA KNEE REHABILITATION: 6 DUAL-STREAM HYBRID MODEL ARCHITECTURES
========================================================================================
All 6 models are true multi-modal dual-stream networks fusing:
  - Stream A: Raw 12-channel IMU sequences (200 time steps @ 100 Hz, Thigh + Shin)
  - Stream B: 16 Clinical Kinematic features (ROM, Depth, Lag, SPARC Smoothness, etc.)
Outputs:
  - Head 1: Exercise Recognition (4 classes: Walk, Extension, Squat, Step-Up)
  - Head 2: Movement Quality Assessment (3 grades: Good, Needs Improvement, Bad)
========================================================================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_hybrid_1_cnn(seq_shape=(200, 12), feat_dim=16, num_classes=4, num_quals=3):
    """Model 1: Hybrid 1D-CNN + Kinematics (Convolutional Feature Fusion)"""
    in_seq = keras.Input(shape=seq_shape, name="input_a_raw_seq")
    in_feat = keras.Input(shape=(feat_dim,), name="input_b_clinical_feat")

    # Stream A: 1D-CNN temporal feature extractor
    x_a = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(in_seq)
    x_a = layers.BatchNormalization()(x_a)
    x_a = layers.MaxPooling1D(2)(x_a)
    x_a = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x_a)
    x_a = layers.GlobalAveragePooling1D()(x_a)
    x_a = layers.Dense(64, activation="relu")(x_a)

    # Stream B: Clinical kinematics dense embedding
    x_b = layers.Dense(64, activation="relu")(in_feat)
    x_b = layers.BatchNormalization()(x_b)

    # Fusion
    fused = layers.concatenate([x_a, x_b])
    fused = layers.Dense(128, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)

    out_class = layers.Dense(num_classes, activation="softmax", name="exercise_class")(fused)
    out_qual  = layers.Dense(num_quals, activation="softmax", name="quality_grade")(fused)

    return keras.Model(inputs=[in_seq, in_feat], outputs=[out_class, out_qual], name="Hybrid_1D_CNN")

def build_hybrid_2_bilstm(seq_shape=(200, 12), feat_dim=16, num_classes=4, num_quals=3):
    """Model 2: Hybrid BiLSTM + Kinematics (Recurrent Sequence Fusion - Benchmark Winner)"""
    in_seq = keras.Input(shape=seq_shape, name="input_a_raw_seq")
    in_feat = keras.Input(shape=(feat_dim,), name="input_b_clinical_feat")

    # Stream A: Bi-directional recurrent temporal rhythm
    x_a = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(in_seq)
    x_a = layers.Bidirectional(layers.LSTM(32))(x_a)
    x_a = layers.Dense(64, activation="relu")(x_a)

    # Stream B: Clinical kinematics dense embedding
    x_b = layers.Dense(64, activation="relu")(in_feat)
    x_b = layers.BatchNormalization()(x_b)

    # Fusion
    fused = layers.concatenate([x_a, x_b])
    fused = layers.Dense(128, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)

    out_class = layers.Dense(num_classes, activation="softmax", name="exercise_class")(fused)
    out_qual  = layers.Dense(num_quals, activation="softmax", name="quality_grade")(fused)

    return keras.Model(inputs=[in_seq, in_feat], outputs=[out_class, out_qual], name="Hybrid_BiLSTM")

def build_hybrid_3_cnn_bilstm(seq_shape=(200, 12), feat_dim=16, num_classes=4, num_quals=3):
    """Model 3: Hybrid CNN-BiLSTM + Kinematics (Spatiotemporal Deep Fusion - AKIRA Spec)"""
    in_seq = keras.Input(shape=seq_shape, name="input_a_raw_seq")
    in_feat = keras.Input(shape=(feat_dim,), name="input_b_clinical_feat")

    # Stream A: Conv1D for local micro-patterns -> BiLSTM for cycle rhythm
    x_a = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(in_seq)
    x_a = layers.BatchNormalization()(x_a)
    x_a = layers.MaxPooling1D(2)(x_a)
    x_a = layers.Bidirectional(layers.LSTM(64))(x_a)
    x_a = layers.Dense(64, activation="relu")(x_a)

    # Stream B: Clinical kinematics embedding
    x_b = layers.Dense(64, activation="relu")(in_feat)
    x_b = layers.BatchNormalization()(x_b)

    # Fusion
    fused = layers.concatenate([x_a, x_b])
    fused = layers.Dense(128, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)

    out_class = layers.Dense(num_classes, activation="softmax", name="exercise_class")(fused)
    out_qual  = layers.Dense(num_quals, activation="softmax", name="quality_grade")(fused)

    return keras.Model(inputs=[in_seq, in_feat], outputs=[out_class, out_qual], name="Hybrid_CNN_BiLSTM")

def build_hybrid_4_resnet(seq_shape=(200, 12), feat_dim=16, num_classes=4, num_quals=3):
    """Model 4: Hybrid 1D-ResNet + Kinematics (Residual Skip-Connection Fusion)"""
    in_seq = keras.Input(shape=seq_shape, name="input_a_raw_seq")
    in_feat = keras.Input(shape=(feat_dim,), name="input_b_clinical_feat")

    # Stream A: 1D Residual Blocks with identity skip-connections
    x = layers.Conv1D(64, kernel_size=1, padding="same")(in_seq)
    
    # Residual Block 1
    res1 = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    res1 = layers.Conv1D(64, kernel_size=3, padding="same")(res1)
    x = layers.add([x, res1])
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)

    # Residual Block 2
    proj = layers.Conv1D(128, kernel_size=1, padding="same")(x)
    res2 = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(proj)
    res2 = layers.Conv1D(128, kernel_size=3, padding="same")(res2)
    x = layers.add([proj, res2])
    x = layers.Activation("relu")(x)
    x_a = layers.GlobalAveragePooling1D()(x)
    x_a = layers.Dense(64, activation="relu")(x_a)

    # Stream B: Clinical kinematics embedding
    x_b = layers.Dense(64, activation="relu")(in_feat)
    x_b = layers.BatchNormalization()(x_b)

    # Fusion
    fused = layers.concatenate([x_a, x_b])
    fused = layers.Dense(128, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)

    out_class = layers.Dense(num_classes, activation="softmax", name="exercise_class")(fused)
    out_qual  = layers.Dense(num_quals, activation="softmax", name="quality_grade")(fused)

    return keras.Model(inputs=[in_seq, in_feat], outputs=[out_class, out_qual], name="Hybrid_ResNet1D")

def build_hybrid_5_self_attention(seq_shape=(200, 12), feat_dim=16, num_classes=4, num_quals=3):
    """Model 5: Hybrid Multi-Head Self-Attention + Kinematics (Transformer-Style Fusion)"""
    in_seq = keras.Input(shape=seq_shape, name="input_a_raw_seq")
    in_feat = keras.Input(shape=(feat_dim,), name="input_b_clinical_feat")

    # Stream A: Multi-Head Self-Attention to focus on key movement points
    x_proj = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(in_seq)
    attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x_proj, x_proj)
    x_a = layers.add([x_proj, attn_out])
    x_a = layers.LayerNormalization()(x_a)
    x_a = layers.GlobalAveragePooling1D()(x_a)
    x_a = layers.Dense(64, activation="relu")(x_a)

    # Stream B: Clinical kinematics embedding
    x_b = layers.Dense(64, activation="relu")(in_feat)
    x_b = layers.BatchNormalization()(x_b)

    # Fusion
    fused = layers.concatenate([x_a, x_b])
    fused = layers.Dense(128, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)

    out_class = layers.Dense(num_classes, activation="softmax", name="exercise_class")(fused)
    out_qual  = layers.Dense(num_quals, activation="softmax", name="quality_grade")(fused)

    return keras.Model(inputs=[in_seq, in_feat], outputs=[out_class, out_qual], name="Hybrid_SelfAttention")

def build_hybrid_6_gated_attention(seq_shape=(200, 12), feat_dim=16, num_classes=4, num_quals=3):
    """Model 6: Hybrid Cross-Modal Gated Attention (Co-Attention Dynamic Fusion)"""
    in_seq = keras.Input(shape=seq_shape, name="input_a_raw_seq")
    in_feat = keras.Input(shape=(feat_dim,), name="input_b_clinical_feat")

    # Stream A: Temporal feature representation
    x_a = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(in_seq)
    x_a = layers.MaxPooling1D(2)(x_a)
    x_a = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x_a)
    x_a = layers.GlobalAveragePooling1D()(x_a)
    x_a = layers.Dense(64, activation="relu")(x_a)

    # Stream B: Clinical feature representation
    x_b = layers.Dense(64, activation="relu")(in_feat)
    x_b = layers.BatchNormalization()(x_b)

    # Cross-Modal Attention Gate: Clinical features dynamically weigh temporal signal channels
    gate = layers.Dense(64, activation="sigmoid", name="cross_modal_gate")(x_b)
    gated_a = layers.multiply([x_a, gate])

    # Fusion
    fused = layers.concatenate([gated_a, x_b])
    fused = layers.Dense(128, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)

    out_class = layers.Dense(num_classes, activation="softmax", name="exercise_class")(fused)
    out_qual  = layers.Dense(num_quals, activation="softmax", name="quality_grade")(fused)

    return keras.Model(inputs=[in_seq, in_feat], outputs=[out_class, out_qual], name="Hybrid_GatedAttention")
