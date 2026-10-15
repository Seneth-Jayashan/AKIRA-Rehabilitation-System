import os
import zipfile
import ast
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure directories exist
os.makedirs('06_FEATURES', exist_ok=True)
os.makedirs('08_RESULTS', exist_ok=True)
os.makedirs('09_MODELS', exist_ok=True)
os.makedirs('07_VISUALIZATIONS/random_forest_quaternion', exist_ok=True)
os.makedirs('11_REPORTS', exist_ok=True)

# 1. Feature Extraction
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
            
        # Parse (W, X, Y, Z)
        parsed_quats = []
        for line in content:
            if not line: continue
            try:
                # ast.literal_eval safely parses the string tuple "(W, X, Y, Z)"
                q = ast.literal_eval(line)
                parsed_quats.append(q)
            except:
                pass
                
        if len(parsed_quats) == 0:
            continue
            
        q_arr = np.array(parsed_quats) # shape (N, 4)
        
        # Calculate features (Mean, Std, Min, Max, Range for W, X, Y, Z)
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
df_features.to_csv('06_FEATURES/random_forest_quaternion_features.csv', index=False)
print(f"Extracted features for {len(df_features)} recordings.")

# 2. Participant-Independent Split
print("Splitting dataset (Participant-Independent)...")
feature_cols = [c for c in df_features.columns if c.startswith('q_')]

X = df_features[feature_cols].values
y = df_features['exercise_label'].values
groups = df_features['participant_id'].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
X_test, y_test, groups_test = X[test_idx], y[test_idx], groups[test_idx]

train_participants = np.unique(groups_train)
test_participants = np.unique(groups_test)

with open('08_RESULTS/random_forest_quaternion_split_report.md', 'w') as f:
    f.write("# Participant-Independent Split Report\\n\\n")
    f.write("## Strategy\\n")
    f.write("GroupShuffleSplit ensuring no overlapping participants between Train and Test sets.\\n\\n")
    f.write("## Train Set\\n")
    f.write(f"- Participants: {len(train_participants)} ({', '.join(train_participants)})\\n")
    f.write(f"- Samples: {len(X_train)}\\n\\n")
    f.write("## Test Set\\n")
    f.write(f"- Participants: {len(test_participants)} ({', '.join(test_participants)})\\n")
    f.write(f"- Samples: {len(X_test)}\\n")

# 3. Model Training
print("Training Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

joblib.dump(rf, '09_MODELS/random_forest_quaternion_classifier.joblib')
with open('09_MODELS/random_forest_quaternion_features.json', 'w') as f:
    json.dump(feature_cols, f)
    
# 4. Evaluation
print("Evaluating on Test Set...")
y_pred = rf.predict(X_test)
labels = np.unique(y)

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

metrics = {
    'accuracy': float(acc),
    'macro_precision': float(prec),
    'macro_recall': float(rec),
    'macro_f1': float(f1)
}
with open('08_RESULTS/random_forest_quaternion_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
    
class_report = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(class_report).transpose().to_csv('08_RESULTS/random_forest_quaternion_classification_report.csv')

# Save predictions
df_preds = pd.DataFrame({
    'recording_id': df_features.iloc[test_idx]['recording_id'],
    'participant_id': groups_test,
    'true_label': y_test,
    'predicted_label': y_pred
})
df_preds.to_csv('08_RESULTS/random_forest_quaternion_predictions.csv', index=False)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Quaternion Random Forest Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('07_VISUALIZATIONS/random_forest_quaternion/confusion_matrix.png')
plt.close()

# Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,8))
plt.title('Feature Importances (Quaternion Temporal Stats)')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [feature_cols[i] for i in indices], rotation=90)
plt.xlim([-1, len(importances)])
plt.tight_layout()
plt.savefig('07_VISUALIZATIONS/random_forest_quaternion/feature_importance.png')
plt.close()

print("Training, evaluation, and visualizations completed successfully!")
