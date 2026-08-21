import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, concatenate
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
WINDOW_SIZE = 100 # 2 seconds at 50 Hz
STEP_SIZE = 50    # 50% overlap

def create_windows(df):
    """
    Takes a dataframe (representing one continuous trial) and chops it into overlapping windows.
    Returns: X_imu (windows, 100, 6), X_age (windows, 1), y_label (windows,)
    """
    imu_data = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
    age = df['age'].values[0]
    correct = df['correct_label'].values[0]
    
    X_imu, X_age, y = [], [], []
    
    for i in range(0, len(imu_data) - WINDOW_SIZE + 1, STEP_SIZE):
        window = imu_data[i : i + WINDOW_SIZE]
        X_imu.append(window)
        X_age.append(age)
        y.append(correct)
        
    return np.array(X_imu), np.array(X_age), np.array(y)

def load_data():
    print("Loading and windowing data for Compensation Detection Model...")
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    X_imu_list, X_age_list, y_list, subject_list = [], [], [], []
    
    valid_activities = ["Squat", "Leg Extension"]
    
    for file in all_files:
        try:
            df = pd.read_csv(file)
            if len(df) < WINDOW_SIZE:
                continue
                
            activity = df['activity_label'].values[0]
            correct_label = df['correct_label'].values[0]
            
            # Filter for specific exercises and known compensation labels
            if activity not in valid_activities:
                continue
            if correct_label not in ["Correct", "Incorrect"]:
                continue
                
            subject_id = df['subject_id'].values[0]
            
            x_i, x_a, y_l = create_windows(df)
            
            if len(x_i) > 0:
                X_imu_list.append(x_i)
                X_age_list.append(x_a)
                y_list.append(y_l)
                
                # Track subjects for leakage-free splitting
                subject_list.extend([subject_id] * len(y_l))
        except Exception as e:
            continue
            
    if not X_imu_list:
        raise ValueError("No valid data found! Check that Standardized_Dataset has KneE-PAD files with Correct/Incorrect labels.")

    X_imu = np.vstack(X_imu_list)
    X_age = np.concatenate(X_age_list)
    y = np.concatenate(y_list)
    subjects = np.array(subject_list)
    
    return X_imu, X_age, y, subjects

def build_binary_model():
    # Input 1: IMU Data
    imu_input = Input(shape=(WINDOW_SIZE, 6), name='imu_input')
    x = Conv1D(32, 3, activation='relu')(imu_input)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    
    # Input 2: Age
    age_input = Input(shape=(1,), name='age_input')
    
    # Concatenate
    concat = concatenate([x, age_input])
    
    # Dense Layers
    d = Dense(64, activation='relu')(concat)
    d = Dropout(0.5)(d)
    # Binary Output
    output = Dense(1, activation='sigmoid', name='output')(d)
    
    model = Model(inputs=[imu_input, age_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    # 1. Load Data
    try:
        X_imu, X_age, y, subjects = load_data()
    except ValueError as e:
        print(e)
        return

    print(f"Total windows generated: {len(X_imu)}")
    
    # 2. Encode Labels (Correct = 1, Incorrect = 0)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes mapped to 0/1: {le.classes_}") 
    # Usually: ['Correct' 'Incorrect'] -> 0: Correct, 1: Incorrect. 
    # Let's print exactly which is which to be sure.
    for i, cls in enumerate(le.classes_):
        print(f"Label {cls} -> {i}")
    
    # 3. Subject-Aware Train/Test Split
    unique_subjects = np.unique(subjects)
    train_subs, test_subs = train_test_split(unique_subjects, test_size=0.2, random_state=42)
    train_subs, val_subs = train_test_split(train_subs, test_size=0.2, random_state=42)
    
    train_idx = np.isin(subjects, train_subs)
    val_idx = np.isin(subjects, val_subs)
    test_idx = np.isin(subjects, test_subs)
    
    X_imu_train, X_age_train, y_train = X_imu[train_idx], X_age[train_idx], y_encoded[train_idx]
    X_imu_val, X_age_val, y_val = X_imu[val_idx], X_age[val_idx], y_encoded[val_idx]
    X_imu_test, X_age_test, y_test = X_imu[test_idx], X_age[test_idx], y_encoded[test_idx]
    
    print(f"Training windows: {len(X_imu_train)}")
    print(f"Validation windows: {len(X_imu_val)}")
    print(f"Testing windows: {len(X_imu_test)}")

    # Standardize Age
    valid_ages = X_age_train[X_age_train > 0]
    mean_age = np.mean(valid_ages) if len(valid_ages) > 0 else 40
    
    X_age_train = np.where(X_age_train == -1, mean_age, X_age_train)
    X_age_val = np.where(X_age_val == -1, mean_age, X_age_val)
    X_age_test = np.where(X_age_test == -1, mean_age, X_age_test)
    
    age_scaler = StandardScaler()
    X_age_train = age_scaler.fit_transform(X_age_train.reshape(-1, 1))
    X_age_val = age_scaler.transform(X_age_val.reshape(-1, 1))
    X_age_test = age_scaler.transform(X_age_test.reshape(-1, 1))
    
    # 4. Build and Train Model
    model = build_binary_model()
    model.summary()
    
    print("\nStarting Training...")
    history = model.fit(
        {'imu_input': X_imu_train, 'age_input': X_age_train},
        y_train,
        validation_data=({'imu_input': X_imu_val, 'age_input': X_age_val}, y_val),
        epochs=15, 
        batch_size=32
    )
    
    # 5. Evaluate
    print("\nEvaluating on Test Set (Unseen Patients)...")
    loss, accuracy = model.evaluate({'imu_input': X_imu_test, 'age_input': X_age_test}, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    y_pred_prob = model.predict({'imu_input': X_imu_test, 'age_input': X_age_test})
    y_pred_classes = (y_pred_prob > 0.5).astype(int).flatten()
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=le.classes_))
    
    # Save Model
    model.save("../models/compensation_detection_model.keras")
    print("\nModel saved as ../models/compensation_detection_model.keras")

if __name__ == "__main__":
    main()
