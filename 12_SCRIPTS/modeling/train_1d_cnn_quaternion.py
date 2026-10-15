import os
import zipfile
import ast
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Config
os.makedirs('09_MODELS', exist_ok=True)
os.makedirs('08_RESULTS', exist_ok=True)
os.makedirs('07_VISUALIZATIONS/1d_cnn_quaternion', exist_ok=True)
os.makedirs('11_REPORTS', exist_ok=True)

MAX_TIME_STEPS = 250  # Fixed sequence length
NUM_FEATURES = 4      # W, X, Y, Z

gesture_map = {
    0: 'NEUTRAL',
    1: 'RADIAL_DEVIATION',
    2: 'WRIST_FLEXION',
    3: 'ULNAR_DEVIATION'
}

label_to_id = {v: k for k, v in gesture_map.items()}

# 1. Parse Data
print("Loading temporal sequences from 3DCDataset zip...")
zip_path = '01_DATASETS/RAW/3DCDataset-main.zip'

sequences = []
labels = []
groups = []
recordings = []

with zipfile.ZipFile(zip_path, 'r') as z:
    imu_files = [f for f in z.namelist() if 'IMU' in f and f.endswith('.txt')]
    for f in imu_files:
        basename = f.split('/')[-1]
        part = f.split('/')[1]
        g_id = int(basename.split('gesture_')[1].split('_')[0])
        t_id = int(basename.split('_')[-1].split('.txt')[0])
        
        with z.open(f) as file:
            content = file.read().decode('utf-8').strip().split('\n')
            
        parsed_quats = []
        for line in content:
            if not line: continue
            try:
                q = ast.literal_eval(line)
                parsed_quats.append([q[0], q[1], q[2], q[3]])
            except:
                pass
                
        if len(parsed_quats) == 0:
            continue
            
        # Pad or Truncate to MAX_TIME_STEPS
        q_arr = np.array(parsed_quats)
        seq_len = len(q_arr)
        
        if seq_len >= MAX_TIME_STEPS:
            q_arr = q_arr[:MAX_TIME_STEPS, :]
        else:
            pad = np.zeros((MAX_TIME_STEPS - seq_len, 4))
            q_arr = np.vstack([q_arr, pad])
            
        sequences.append(q_arr)
        labels.append(g_id)
        groups.append(part)
        recordings.append(f"{part}_g{g_id}_t{t_id}")

X = np.array(sequences)
y = np.array(labels)
groups = np.array(groups)
recordings = np.array(recordings)

print(f"Loaded {len(X)} sequences. Shape: {X.shape}")

# 2. Train/Test Split (Same as Random Forest)
print("Splitting dataset (Participant-Independent)...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
X_test, y_test, groups_test = X[test_idx], y[test_idx], groups[test_idx]

train_participants = np.unique(groups_train)
test_participants = np.unique(groups_test)
print(f"Train participants: {train_participants}")
print(f"Test participants: {test_participants}")

# 3. Model Definition
print("Building 1D CNN model...")
model = Sequential([
    Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(MAX_TIME_STEPS, NUM_FEATURES)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 4. Training
print("Training 1D CNN...")
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[es],
    verbose=1
)

model.save('09_MODELS/1d_cnn_quaternion_classifier.keras')

config = {
    'max_time_steps': MAX_TIME_STEPS,
    'num_features': NUM_FEATURES,
    'feature_names': ['W', 'X', 'Y', 'Z'],
    'gesture_map': gesture_map,
    'padding_strategy': 'post_zero_padding',
    'truncation_strategy': 'post_truncation'
}
with open('09_MODELS/1d_cnn_quaternion_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# 5. Evaluation
print("Evaluating on Test Set...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

metrics = {
    'accuracy': float(acc),
    'macro_precision': float(prec),
    'macro_recall': float(rec),
    'macro_f1': float(f1)
}

with open('08_RESULTS/1d_cnn_quaternion_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
    
labels_mapped = [gesture_map[i] for i in range(4)]
y_test_mapped = [gesture_map[i] for i in y_test]
y_pred_mapped = [gesture_map[i] for i in y_pred]

class_report = classification_report(y_test_mapped, y_pred_mapped, output_dict=True)
pd.DataFrame(class_report).transpose().to_csv('08_RESULTS/1d_cnn_quaternion_classification_report.csv')

# Visualizations
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test_mapped, y_pred_mapped, labels=labels_mapped)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=labels_mapped, yticklabels=labels_mapped)
plt.title('1D CNN Quaternion Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('07_VISUALIZATIONS/1d_cnn_quaternion/confusion_matrix.png')
plt.close()

plt.figure(figsize=(10,6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('1D CNN Training History')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('07_VISUALIZATIONS/1d_cnn_quaternion/training_history.png')
plt.close()

print(f"CNN Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%")
print("1D CNN pipeline completed successfully.")
