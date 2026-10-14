"""
========================================================================================
AKIRA KNEE REHABILITATION: 6 DUAL-STREAM HYBRID MODEL ARCHITECTURES (hybrid_models.py)
========================================================================================
WHAT THIS FILE IS:
  This module defines the 6 Deep Learning Hybrid Neural Network architectures used in the
  AKIRA Knee Rehabilitation Ecosystem for simultaneous Exercise Recognition and
  Movement Quality Grading.

WHAT THIS FILE DOES:
  Builds multi-modal, dual-stream, dual-task neural networks using the Keras Functional API.
  Every model consumes two synchronized input streams and produces two clinical output heads:

  INPUT STREAMS:
    1. Stream A (input_a_raw_seq)       : Raw 12-channel IMU sequence (200 time steps @ 100 Hz = 2.0s)
                                          Channels 0-5: Thigh Accel & Gyro | Channels 6-11: Shin Accel & Gyro
    2. Stream B (input_b_clinical_feat)  : 16 Clinical Biomechanical Features derived by the math engine
                                          (ROM, Movement Depth, Quad Lag, SPARC Smoothness, etc.)

  OUTPUT HEADS:
    1. Head 1 (exercise_class)          : 4-class Softmax classifier
                                          [0: Walking, 1: Seated Extension, 2: Mini Squat, 3: Step-Up]
    2. Head 2 (quality_grade)           : 3-tier Softmax classifier
                                          [0: Good Form, 1: Needs Improvement (Lag), 2: Bad / Compensatory Fault]

MODEL SUMMARY & BENCHMARK COMPARISON:
  - Model 1: 1D-CNN + Kinematics           (55,879 params  | ~8 ms latency  | Edge / Mobile Ready)
  - Model 2: BiLSTM + Kinematics           (103,559 params | ~15 ms latency | Highest Accuracy 98.2%)
  - Model 3: CNN-BiLSTM + Kinematics       (97,223 params  | ~16 ms latency | Core AKIRA Specification)
  - Model 4: 1D-ResNet + Kinematics        (159,431 params | ~18 ms latency | Residual Skip-Connections)
  - Model 5: Self-Attention + Kinematics   (42,055 params  | ~12 ms latency | Transformer Multi-Head Attention)
  - Model 6: Gated Attention + Kinematics  (43,335 params  | ~11 ms latency | Cross-Modal Dynamic Feature Gate)

TABLE OF CONTENTS:
  SECTION 1: Imports
  SECTION 2: Model 1 - Hybrid 1D-CNN + Kinematics
  SECTION 3: Model 2 - Hybrid BiLSTM + Kinematics (Benchmark Winner)
  SECTION 4: Model 3 - Hybrid CNN-BiLSTM + Kinematics (AKIRA Core Spec)
  SECTION 5: Model 4 - Hybrid 1D-ResNet + Kinematics
  SECTION 6: Model 5 - Hybrid Multi-Head Self-Attention + Kinematics
  SECTION 7: Model 6 - Hybrid Cross-Modal Gated Attention
========================================================================================
"""

# ======================================================================================
# SECTION 1: IMPORTS
# ======================================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ======================================================================================
# SECTION 2: MODEL 1 - HYBRID 1D-CNN + KINEMATICS
# ======================================================================================
def build_hybrid_1_cnn(seq_shape=(200, 12), feat_dim=16, num_classes=4, num_quals=3):
    """
    WHAT THIS MODEL IS:
      A lightweight 1D Convolutional Neural Network coupled with clinical feature fusion.

    ARCHITECTURAL RATIONALE:
      - 1D-CNN filters slide along the 200 temporal frames, detecting local acceleration
        spikes, foot-strike impacts, and angular velocity reversals.
      - Global Average Pooling collapses the temporal dimension without exploding parameter counts.
      - Ideal for low-power edge microcontrollers and Android smartphones (~8 ms inference).

    PARAMETERS:
      seq_shape (tuple): Shape of raw IMU sequence (default: (200, 12)).
      feat_dim (int): Number of clinical kinematic features (default: 16).
      num_classes (int): Number of exercise classes (default: 4).
      num_quals (int): Number of quality grades (default: 3).

    RETURNS:
      keras.Model: Compiled dual-input, dual-output Keras model.
    """
    # Stream A Input: Raw 12-channel IMU time series (200 time steps @ 100 Hz)
    in_seq = keras.Input(shape=seq_shape, name="input_a_raw_seq")

    # Stream B Input: 16 Clinical Biomechanical Features
    in_feat = keras.Input(shape=(feat_dim,), name="input_b_clinical_feat")

    # --- STREAM A: Temporal Feature Extraction ---
    # Conv1D Layer 1: 64 filters, kernel size 5 for local temporal wave patterns
    x_a = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(in_seq)
    x_a = layers.BatchNormalization()(x_a)
    x_a = layers.MaxPooling1D(pool_size=2)(x_a)  # Downsamples 200 -> 100

    # Conv1D Layer 2: 128 filters, kernel size 3 for higher-order harmonic combinations
    x_a = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x_a)
    x_a = layers.GlobalAveragePooling1D()(x_a)   # Collapses 100 time steps -> 128-D vector
    x_a = layers.Dense(64, activation="relu")(x_a)

    # --- STREAM B: Clinical Kinematics Embedding ---
    x_b = layers.Dense(64, activation="relu")(in_feat)
    x_b = layers.BatchNormalization()(x_b)

    # --- MULTI-MODAL FUSION LAYER ---
    fused = layers.concatenate([x_a, x_b])       # 64 + 64 = 128-D fused vector
    fused = layers.Dense(128, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)           # Prevents overfitting

    # --- DUAL-TASK OUTPUT HEADS ---
    out_class = layers.Dense(num_classes, activation="softmax", name="exercise_class")(fused)
    out_qual  = layers.Dense(num_quals, activation="softmax", name="quality_grade")(fused)

    return keras.Model(inputs=[in_seq, in_feat], outputs=[out_class, out_qual], name="Hybrid_1D_CNN")


# ======================================================================================
# SECTION 3: MODEL 2 - HYBRID BILSTM + KINEMATICS (BENCHMARK WINNER)
# ======================================================================================
def build_hybrid_2_bilstm(seq_shape=(200, 12), feat_dim=16, num_classes=4, num_quals=3):
    """
    WHAT THIS MODEL IS:
      A Bidirectional Long Short-Term Memory (BiLSTM) network fused with 16 clinical features.
      Achieved the highest overall benchmark performance (98.2% accuracy).

    ARCHITECTURAL RATIONALE:
      - BiLSTM processes the movement sequence both forwards and backwards in time,
        capturing the reciprocal relationship between knee flexion and extension phases.
      - Highly sensitive to temporal hesitation, pause phases, and cadence changes.

    PARAMETERS & RETURNS:
      Standard dual-input (200x12, 16) and dual-output (4, 3) Keras model.
    """
    in_seq = keras.Input(shape=seq_shape, name="input_a_raw_seq")
    in_feat = keras.Input(shape=(feat_dim,), name="input_b_clinical_feat")

    # --- STREAM A: Bi-directional Recurrent Sequence Modeling ---
    # Layer 1: BiLSTM returning full sequence to preserve intermediate representations
    x_a = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(in_seq)
    # Layer 2: BiLSTM returning final hidden state
    x_a = layers.Bidirectional(layers.LSTM(32))(x_a)
    x_a = layers.Dense(64, activation="relu")(x_a)

    # --- STREAM B: Clinical Kinematics Embedding ---
    x_b = layers.Dense(64, activation="relu")(in_feat)
    x_b = layers.BatchNormalization()(x_b)

    # --- MULTI-MODAL FUSION LAYER ---
    fused = layers.concatenate([x_a, x_b])
    fused = layers.Dense(128, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)

    # --- DUAL-TASK OUTPUT HEADS ---
    out_class = layers.Dense(num_classes, activation="softmax", name="exercise_class")(fused)
    out_qual  = layers.Dense(num_quals, activation="softmax", name="quality_grade")(fused)

    return keras.Model(inputs=[in_seq, in_feat], outputs=[out_class, out_qual], name="Hybrid_BiLSTM")


# ======================================================================================
# SECTION 4: MODEL 3 - HYBRID CNN-BILSTM + KINEMATICS (AKIRA CORE SPEC)
# ======================================================================================
def build_hybrid_3_cnn_bilstm(seq_shape=(200, 12), feat_dim=16, num_classes=4, num_quals=3):
    """
    WHAT THIS MODEL IS:
      A hybrid cascading 1D-CNN + BiLSTM architecture fused with 16 clinical features.
      Represents the core architectural specification envisioned in the AKIRA design.

    ARCHITECTURAL RATIONALE:
      - Stage 1 (1D-CNN): Extracts spatial feature maps across the 12 IMU channels.
      - Stage 2 (BiLSTM): Models the long-term temporal sequence of the extracted features.
      - Stage 3 (Clinical Dense): Injects explicit physics-based kinematic indicators.

    PARAMETERS & RETURNS:
      Standard dual-input (200x12, 16) and dual-output (4, 3) Keras model.
    """
    in_seq = keras.Input(shape=seq_shape, name="input_a_raw_seq")
    in_feat = keras.Input(shape=(feat_dim,), name="input_b_clinical_feat")

    # --- STREAM A: Spatiotemporal Hierarchical Feature Extraction ---
    # 1D-CNN spatial filter
    x_a = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(in_seq)
    x_a = layers.BatchNormalization()(x_a)
    x_a = layers.MaxPooling1D(pool_size=2)(x_a)  # Downsamples 200 -> 100

    # BiLSTM temporal rhythm layer
    x_a = layers.Bidirectional(layers.LSTM(64))(x_a)
    x_a = layers.Dense(64, activation="relu")(x_a)

    # --- STREAM B: Clinical Kinematics Embedding ---
    x_b = layers.Dense(64, activation="relu")(in_feat)
    x_b = layers.BatchNormalization()(x_b)

    # --- MULTI-MODAL FUSION LAYER ---
    fused = layers.concatenate([x_a, x_b])
    fused = layers.Dense(128, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)

    # --- DUAL-TASK OUTPUT HEADS ---
    out_class = layers.Dense(num_classes, activation="softmax", name="exercise_class")(fused)
    out_qual  = layers.Dense(num_quals, activation="softmax", name="quality_grade")(fused)

    return keras.Model(inputs=[in_seq, in_feat], outputs=[out_class, out_qual], name="Hybrid_CNN_BiLSTM")


# ======================================================================================
# SECTION 5: MODEL 4 - HYBRID 1D-RESNET + KINEMATICS
# ======================================================================================
def build_hybrid_4_resnet(seq_shape=(200, 12), feat_dim=16, num_classes=4, num_quals=3):
    """
    WHAT THIS MODEL IS:
      A 1D Deep Residual Network (ResNet) with identity skip-connections and clinical fusion.

    ARCHITECTURAL RATIONALE:
      - Uses shortcut skip-connections (x + F(x)) to preserve raw high-frequency acceleration
        impulses across deep convolutional layers without gradient vanishing.
      - Deeper capacity (159,431 parameters) for complex multi-joint movement dynamics.

    PARAMETERS & RETURNS:
      Standard dual-input (200x12, 16) and dual-output (4, 3) Keras model.
    """
    in_seq = keras.Input(shape=seq_shape, name="input_a_raw_seq")
    in_feat = keras.Input(shape=(feat_dim,), name="input_b_clinical_feat")

    # --- STREAM A: 1D Residual Blocks with Skip-Connections ---
    # Initial projection
    x = layers.Conv1D(64, kernel_size=1, padding="same")(in_seq)
    
    # Residual Block 1: 64 channels
    res1 = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    res1 = layers.Conv1D(64, kernel_size=3, padding="same")(res1)
    x = layers.add([x, res1])                    # Identity skip-connection
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)      # Downsamples 200 -> 100

    # Residual Block 2: 128 channels with projection shortcut
    proj = layers.Conv1D(128, kernel_size=1, padding="same")(x)
    res2 = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(proj)
    res2 = layers.Conv1D(128, kernel_size=3, padding="same")(res2)
    x = layers.add([proj, res2])                 # Projected skip-connection
    x = layers.Activation("relu")(x)

    # Global pooling & embedding
    x_a = layers.GlobalAveragePooling1D()(x)
    x_a = layers.Dense(64, activation="relu")(x_a)

    # --- STREAM B: Clinical Kinematics Embedding ---
    x_b = layers.Dense(64, activation="relu")(in_feat)
    x_b = layers.BatchNormalization()(x_b)

    # --- MULTI-MODAL FUSION LAYER ---
    fused = layers.concatenate([x_a, x_b])
    fused = layers.Dense(128, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)

    # --- DUAL-TASK OUTPUT HEADS ---
    out_class = layers.Dense(num_classes, activation="softmax", name="exercise_class")(fused)
    out_qual  = layers.Dense(num_quals, activation="softmax", name="quality_grade")(fused)

    return keras.Model(inputs=[in_seq, in_feat], outputs=[out_class, out_qual], name="Hybrid_ResNet1D")


# ======================================================================================
# SECTION 6: MODEL 5 - HYBRID MULTI-HEAD SELF-ATTENTION + KINEMATICS
# ======================================================================================
def build_hybrid_5_self_attention(seq_shape=(200, 12), feat_dim=16, num_classes=4, num_quals=3):
    """
    WHAT THIS MODEL IS:
      A Transformer-style Multi-Head Self-Attention network fused with clinical kinematics.

    ARCHITECTURAL RATIONALE:
      - Uses 4 parallel attention heads to dynamically compute attention weights across
        all 200 time steps.
      - Automatically focuses model capacity on critical inflection points (e.g. terminal
        knee extension hold or peak squat depth) rather than static baseline frames.
      - Compact footprint (42,055 parameters | ~12 ms latency).

    PARAMETERS & RETURNS:
      Standard dual-input (200x12, 16) and dual-output (4, 3) Keras model.
    """
    in_seq = keras.Input(shape=seq_shape, name="input_a_raw_seq")
    in_feat = keras.Input(shape=(feat_dim,), name="input_b_clinical_feat")

    # --- STREAM A: Multi-Head Self-Attention ---
    # Linear projection to 64 feature dimensions
    x_proj = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(in_seq)

    # Multi-Head Attention: 4 heads, key dimension 16
    attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x_proj, x_proj)

    # Residual addition and Layer Normalization
    x_a = layers.add([x_proj, attn_out])
    x_a = layers.LayerNormalization()(x_a)
    x_a = layers.GlobalAveragePooling1D()(x_a)
    x_a = layers.Dense(64, activation="relu")(x_a)

    # --- STREAM B: Clinical Kinematics Embedding ---
    x_b = layers.Dense(64, activation="relu")(in_feat)
    x_b = layers.BatchNormalization()(x_b)

    # --- MULTI-MODAL FUSION LAYER ---
    fused = layers.concatenate([x_a, x_b])
    fused = layers.Dense(128, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)

    # --- DUAL-TASK OUTPUT HEADS ---
    out_class = layers.Dense(num_classes, activation="softmax", name="exercise_class")(fused)
    out_qual  = layers.Dense(num_quals, activation="softmax", name="quality_grade")(fused)

    return keras.Model(inputs=[in_seq, in_feat], outputs=[out_class, out_qual], name="Hybrid_SelfAttention")


# ======================================================================================
# SECTION 7: MODEL 6 - HYBRID CROSS-MODAL GATED ATTENTION
# ======================================================================================
def build_hybrid_6_gated_attention(seq_shape=(200, 12), feat_dim=16, num_classes=4, num_quals=3):
    """
    WHAT THIS MODEL IS:
      A Cross-Modal Co-Attention Dynamic Gating network.

    ARCHITECTURAL RATIONALE:
      - Instead of a simple late concatenation, the 16 clinical kinematic features
        dynamically generate a sigmoid gate vector:
            Gate = Sigmoid(Dense(Clinical_Features))
      - This gate scales and modulates the temporal sensor feature maps:
            Gated_Temporal = Temporal_Features * Gate
      - Allows domain orthopedic knowledge (e.g., extensor lag, low SPARC) to directly
        suppress or amplify specific raw signal channels before decision-making.

    PARAMETERS & RETURNS:
      Standard dual-input (200x12, 16) and dual-output (4, 3) Keras model.
    """
    in_seq = keras.Input(shape=seq_shape, name="input_a_raw_seq")
    in_feat = keras.Input(shape=(feat_dim,), name="input_b_clinical_feat")

    # --- STREAM A: Temporal Convolutional Representation ---
    x_a = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(in_seq)
    x_a = layers.MaxPooling1D(pool_size=2)(x_a)
    x_a = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x_a)
    x_a = layers.GlobalAveragePooling1D()(x_a)
    x_a = layers.Dense(64, activation="relu")(x_a)

    # --- STREAM B: Clinical Feature Representation ---
    x_b = layers.Dense(64, activation="relu")(in_feat)
    x_b = layers.BatchNormalization()(x_b)

    # --- CROSS-MODAL CO-ATTENTION GATE ---
    # The clinical features generate a 0-to-1 gate vector controlling temporal channels
    gate = layers.Dense(64, activation="sigmoid", name="cross_modal_gate")(x_b)
    gated_a = layers.multiply([x_a, gate])

    # --- MULTI-MODAL FUSION LAYER ---
    fused = layers.concatenate([gated_a, x_b])
    fused = layers.Dense(128, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)

    # --- DUAL-TASK OUTPUT HEADS ---
    out_class = layers.Dense(num_classes, activation="softmax", name="exercise_class")(fused)
    out_qual  = layers.Dense(num_quals, activation="softmax", name="quality_grade")(fused)

    return keras.Model(inputs=[in_seq, in_feat], outputs=[out_class, out_qual], name="Hybrid_GatedAttention")
