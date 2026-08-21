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
python src/preprocess.py
```

## 3. Training the Models


```bash
# Train Model 1: Exercise Recognition (Squat vs Leg Extension vs Walk)
python src/train_model.py

# Train Model 2: Binary Compensation Detection (Correct vs Incorrect)
python src/train_compensation_model.py

# Train Model 3: Multi-Class Compensation Diagnosis (e.g. Weight Transfer)
python src/train_multiclass_compensation_model.py
```

*Note: The models will automatically save themselves into the `models/` directory upon completion.*
