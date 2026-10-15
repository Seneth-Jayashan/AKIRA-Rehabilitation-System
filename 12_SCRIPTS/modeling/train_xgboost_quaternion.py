import os
import zipfile
import ast
import json
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('06_FEATURES', exist_ok=True)
os.makedirs('08_RESULTS', exist_ok=True)
os.makedirs('09_MODELS', exist_ok=True)
os.makedirs('07_VISUALIZATIONS/xgboost_quaternion', exist_ok=True)
os.makedirs('11_REPORTS', exist_ok=True)

print("Extracting features from 3DCDataset zip...")
zip_path = '01_DATASETS/RAW/3DCDataset-main.zip'
features_list = []

gesture_map = {
    0: 'NEUTRAL',
    1: 'RADIAL_DEVIATION',
    2: 'WRIST_FLEXION',
    3: 'ULNAR_DEVIATION'
}

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
                parsed_quats.append(q)
            except:
                pass
                
        if len(parsed_quats) == 0:
            continue
            
        q_arr = np.array(parsed_quats)
        
        means = np.mean(q_arr, axis=0)
        stds = np.std(q_arr, axis=0)
        mins = np.min(q_arr, axis=0)
        maxs = np.max(q_arr, axis=0)
        ranges = maxs - mins
        
        feature_dict = {
            'participant_id': part,
            'exercise_label': gesture_map[g_id],
            'recording_id': f"{part}_g{g_id}_t{t_id}"
        }
        
        for i, comp in enumerate(['W', 'X', 'Y', 'Z']):
            feature_dict[f'q_{comp}_mean'] = means[i]
            feature_dict[f'q_{comp}_std'] = stds[i]
            feature_dict[f'q_{comp}_min'] = mins[i]
            feature_dict[f'q_{comp}_max'] = maxs[i]
            feature_dict[f'q_{comp}_range'] = ranges[i]
            
        features_list.append(feature_dict)

df_features = pd.DataFrame(features_list)
df_features.to_csv('06_FEATURES/xgboost_quaternion_features.csv', index=False)
print(f"Extracted features for {len(df_features)} recordings.")

print("Splitting dataset (Participant-Independent)...")
feature_cols = [c for c in df_features.columns if c.startswith('q_')]

X = df_features[feature_cols].values
y = df_features['exercise_label'].values
groups = df_features['participant_id'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y_encoded, groups))

X_train, y_train, groups_train = X[train_idx], y_encoded[train_idx], groups[train_idx]
X_test, y_test, groups_test = X[test_idx], y_encoded[test_idx], groups[test_idx]

train_participants = np.unique(groups_train)
test_participants = np.unique(groups_test)

from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

print("Training XGBoost Classifier...")
model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    random_state=42,
    use_label_encoder=False,
    learning_rate=0.1,
    max_depth=6,
    n_estimators=100
)

model.fit(X_train, y_train, sample_weight=sample_weights)

print("Evaluating...")
y_pred_encoded = model.predict(X_test)
y_pred = le.inverse_transform(y_pred_encoded)
y_test_labels = le.inverse_transform(y_test)

acc = accuracy_score(y_test_labels, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test_labels, y_pred, average='macro')

metrics = {
    'accuracy': float(acc),
    'macro_precision': float(precision),
    'macro_recall': float(recall),
    'macro_f1': float(f1)
}

with open('08_RESULTS/xgboost_quaternion_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
    
df_report = pd.DataFrame(classification_report(y_test_labels, y_pred, output_dict=True)).T
df_report.to_csv('08_RESULTS/xgboost_quaternion_classification_report.csv')

print(f"Accuracy: {acc*100:.2f}% | Macro F1: {f1*100:.2f}%")

cm = confusion_matrix(y_test_labels, y_pred, labels=le.classes_)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('XGBoost Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('07_VISUALIZATIONS/xgboost_quaternion/confusion_matrix.png')
plt.close()

# Save Model Pipeline
# Since XGBoost expects integer labels internally, we can save both the model and the LabelEncoder
model_data = {
    'model': model,
    'label_encoder': le,
    'features': feature_cols
}
joblib.dump(model_data, '09_MODELS/xgboost_quaternion_classifier.joblib')

with open('09_MODELS/xgboost_quaternion_features.json', 'w') as f:
    json.dump(feature_cols, f, indent=4)

# Save split details
split_info = f"""# XGBoost Dataset Split
- Total Recordings: {len(X)}
- Train Recordings: {len(X_train)}
- Test Recordings: {len(X_test)}
- Train Participants: {list(train_participants)}
- Test Participants: {list(test_participants)}
"""
with open('08_RESULTS/xgboost_quaternion_split_report.md', 'w') as f:
    f.write(split_info)
    
print("Saved all XGBoost artifacts.")
