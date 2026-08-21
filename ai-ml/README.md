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

## 2. Data Pipelines & Datasets

### Raw Datasets
The preprocessing script ingests four public datasets:
1. **KneE-PAD Dataset:** Contains labeled IMU data for Squats, Leg Extensions, and Walking. Crucial for this project as it includes specific labels for postural compensations (e.g., Squat Weight Transfer, Leg Extension Lift, Walk Hip Abduction) and the subject's age.
2. **PHYTMO Dataset:** Contains general lower-limb movement data during gait and physical therapy.
3. **Physical Therapy Exercises Dataset:** Contains generic IMU templates and test data for various exercises.
4. **10m Walk Test Dataset:** Contains accelerometer and gyroscope data from patients performing a 10-meter walk test.

### Preprocessing Pipeline
To create a unified, robust training set, `preprocess.py` performs the following steps:
1. **Data Standardization:** Reads diverse file formats (Numpy, CSV, TXT), handles different IMU orientations, and converts units (e.g., accelerations to `g`, gyroscope to `deg/s`).
2. **Resampling:** Since the source datasets were recorded at different frequencies (e.g., 25Hz, 148Hz), all data is interpolated and resampled to a uniform **50 Hz** (target `dt` of 0.02s).
3. **Metadata Alignment:** Extracts subject ID, age, and normalizes activity labels into consistent categories (Squat, Leg Extension, Walking). Maps specific compensation classes into standard formats.
4. **CSV Export:** Saves the standardized data into `data/processed/` with uniform columns: `timestamp, dataset_name, subject_id, age, activity_label, correct_label, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`.

To run the data ingestion pipeline:

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

### Machine Learning Pipeline
Before training, the time-series data undergoes a final transformation pipeline:
1. **Windowing:** The continuous IMU streams are sliced into fixed-size windows of **100 samples** (representing 2 seconds at 50 Hz). A step size of 50 samples is used to create a **50% overlap** between consecutive windows, which helps augment the data and capture movements across boundaries.
2. **Subject-Aware Splitting:** To prevent data leakage (where the model memorizes a specific person's movement rather than the general exercise form), the data is split into Train/Validation/Test sets based on `subject_id` rather than random window shuffling. This ensures the Test Set accuracy represents the model's performance on *completely unseen* patients.
3. **Feature Scaling:** Age is imputed with the mean for missing values and standardized using `StandardScaler` to improve gradient flow.

### Architectures
All models utilize 1D Convolutional Neural Networks (CNN) to process the windowed IMU data (6 channels: accelerometer and gyroscope). The patient's `age` is concatenated with the flattened CNN features before the final dense layers to provide demographic context.

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

## 5. Visualizing the Models

You can generate PNG images of the 1D CNN architectures using the visualization script provided. This requires `graphviz` to be installed on your system.

```bash
# First, ensure you have the system dependencies (Linux example)
sudo apt-get install graphviz

# Ensure you have the python libraries (pydot, graphviz)
pip install -r requirements.txt

# Run the visualization script
cd visualization
python visualize_models.py
```

The model architecture diagrams will be saved in the `visualization/output/` directory.

## 6. Model Evaluation Dashboard (Streamlit)

AKIRA AI features a highly advanced, premium interactive dashboard to evaluate the trained neural networks in a modern, cyber-clinical telemetry console style.

### Running the Dashboard
To launch the evaluation interface, ensure you are in the virtual environment and run:
```bash
streamlit run app.py
```

### Dashboard Features & Architecture

- **Premium AI Interface:** Built with a glassmorphism aesthetic, dynamic glowing background gradients, cyber-neon colors (cyan and purple), and an animated IMU telemetry waveform. 
- **Strict Data Isolation (Patient-Held-Out Test Set):** The true metric of a rehabilitation model is how it performs on *unseen patients*. The dashboard dynamically reconstructs the exact test set by utilizing the identical `Subject-Aware Split` logic and random seeds (`random_state=42`) used during model training. This guarantees that **no patient in the test set was ever seen by the model during training**, providing a mathematically honest evaluation.
- **Dynamic Neural Inference:** The app loads the trained `.keras` weights and directly applies standard scaling and inference over the unseen patient windows.
- **Visual Diagnostics:** 
  - Generates a **Neon Confusion Matrix** allowing developers to see exact false positives and misclassifications (e.g., misdiagnosing a Walk as a Squat).
  - Generates a highly detailed **Classification Report** (Precision, Recall, F1-Score) formatted to match the glowing cyberpunk UI.
  - Highlights top-line metrics: **Test Accuracy**, **Unseen Patients Analyzed**, and **Total Windows Analyzed**.
