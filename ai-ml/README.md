# AKIRA: Hip and Knee Rehabilitation AI

This repository contains the AI/ML pipeline for the AKIRA rehabilitation project. It processes raw IMU data from public datasets and trains 1D Convolutional Neural Networks to recognize exercises, detect incorrect form, and diagnose specific postural compensations.

## 1. Environment Setup


```bash
# Create a new virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install all required libraries
pip install -r requirements.txt
```

## 2. Generating Processed Data


```bash
# Make sure your virtual environment is active first
source venv/bin/activate

# Run the preprocessing script
# This reads from data/raw/ and writes to data/processed/
cd src
python preprocess.py
```

## 3. Training the Models


```bash
# Train Model 1: Exercise Recognition (Squat vs Leg Extension vs Walk)
cd src
python train_model.py

# Train Model 2: Binary Compensation Detection (Correct vs Incorrect)
python train_compensation_model.py

# Train Model 3: Multi-Class Compensation Diagnosis (e.g. Weight Transfer)
python train_multiclass_compensation_model.py
```

*Note: The models will automatically save themselves into the `models/` directory upon completion.*

## 4. Model Details & Training Procedures

All models utilize 1D Convolutional Neural Networks (CNN) to process the overlapping windowed time-series IMU data (accelerometer and gyroscope). The patient's `age` is concatenated with the flattened CNN features before the final dense layers to provide demographic context.

### Model 1: Exercise Recognition
- **Goal:** Classifies the exercise being performed (Squat, Leg Extension, Walking, Other Rehab).
- **Architecture:** 2x Conv1D layers (32 and 64 filters) with MaxPooling1D -> Flatten -> Concatenate(Age) -> Dense(64) -> Dropout(0.5) -> Dense(Softmax).
- **Compilation:** Adam optimizer, Sparse Categorical Crossentropy loss.
- **Training Setup:** 10 Epochs, Batch Size of 32.

### Model 2: Binary Compensation Detection
- **Goal:** Detects if an exercise form is "Correct" or "Incorrect".
- **Architecture:** 2x Conv1D layers (32 and 64 filters) with MaxPooling1D -> Flatten -> Concatenate(Age) -> Dense(64) -> Dropout(0.5) -> Dense(1, Sigmoid).
- **Compilation:** Adam optimizer, Binary Crossentropy loss.
- **Training Setup:** 15 Epochs, Batch Size of 32.

### Model 3: Multi-Class Compensation Diagnosis
- **Goal:** Diagnoses specific postural compensations (e.g., Squat Weight Transfer, Walk Hip Abduction).
- **Architecture:** Deeper CNN with Batch Normalization. 2x Conv1D layers (32 and 64 filters) -> Flatten -> Concatenate(Age) -> Dense(128) with L2 regularization -> Dense(64) with L2 regularization -> Dense(Softmax). Includes Dropout(0.5) and Batch Normalization between dense layers.
- **Compilation:** Adam optimizer (learning rate = 0.001), Sparse Categorical Crossentropy loss.
- **Training Setup:** 100 Epochs (max), Batch Size of 32.
- **Advanced Techniques:** Uses class weighting to handle data imbalance. Incorporates `EarlyStopping` (patience of 15) to prevent overfitting, and `ReduceLROnPlateau` to decrease the learning rate when validation loss plateaus.
