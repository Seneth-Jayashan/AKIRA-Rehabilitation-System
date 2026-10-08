import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from src.models.taxonomy import map_label

class MLDatasetBuilder:
    def __init__(self, processed_data_dir: str = "datasets/processed", out_dir: str = "datasets/ml_ready"):
        self.processed_data_dir = processed_data_dir
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        
    def build_dataset(self):
        print("1. Loading processed feature datasets...")
        dfs = []
        for file in os.listdir(self.processed_data_dir):
            if file.endswith("_features.csv"):
                print(f"   Loading {file}...")
                path = os.path.join(self.processed_data_dir, file)
                df = pd.read_csv(path)
                dfs.append(df)
                
        if not dfs:
            print("No processed datasets found. Run the preprocessing pipeline first.")
            return
            
        master_df = pd.concat(dfs, ignore_index=True)
        print(f"   Total Windows Loaded: {len(master_df)}")
        
        print("2. Mapping raw labels to unified taxonomy...")
        master_df["standard_activity"] = master_df["activity"].apply(map_label)
        
        # Drop unknowns or calibrations if we don't want them in ML training
        master_df = master_df[~master_df["standard_activity"].isin(["Unknown", "Calibration", "Other"])]
        
        print("\nClass Distribution:")
        print(master_df["standard_activity"].value_counts())
        
        print("\n3. Performing Subject-Independent Split (70/15/15)...")
        # We need a unique subject identifier across datasets
        master_df["unique_subject_id"] = master_df["dataset_name"] + "_" + master_df["subject_id"].astype(str)
        
        X = master_df.drop(columns=["standard_activity", "unique_subject_id"]) # keep raw activity for reference if needed
        y = master_df["standard_activity"]
        groups = master_df["unique_subject_id"]
        
        # Split 1: Train (70%) vs Temp (30%)
        num_groups = groups.nunique()
        if num_groups < 5:
            print(f"Warning: Only {num_groups} subjects available. Doing standard 2-way train/test split.")
            gss1 = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
            train_idx, test_idx = next(gss1.split(X, y, groups))
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[test_idx], y.iloc[test_idx] # Val is Test
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        else:
            gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
            train_idx, temp_idx = next(gss1.split(X, y, groups))
            
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            groups_temp = groups.iloc[temp_idx]
            X_temp, y_temp = X.iloc[temp_idx], y.iloc[temp_idx]
            
            # Split 2: Validation (15%) vs Test (15%)
            gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
            val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups_temp))
            
            X_val, y_val = X_temp.iloc[val_idx], y_temp.iloc[val_idx]
            X_test, y_test = X_temp.iloc[test_idx], y_temp.iloc[test_idx]
        
        # Extract features vs metadata
        metadata_cols = ["dataset_name", "subject_id", "session_id", "trial_id", "sensor_id", "activity"]
        
        X_train_feats = X_train.drop(columns=metadata_cols, errors='ignore')
        X_val_feats = X_val.drop(columns=metadata_cols, errors='ignore')
        X_test_feats = X_test.drop(columns=metadata_cols, errors='ignore')
        
        print(f"\nSplit Results:")
        print(f"   Train: {len(X_train)} windows")
        print(f"   Validation: {len(X_val)} windows")
        print(f"   Test: {len(X_test)} windows")
        
        print("\n4. Saving ML-Ready Datasets...")
        # Save features
        X_train_feats.to_csv(os.path.join(self.out_dir, "X_train.csv"), index=False)
        X_val_feats.to_csv(os.path.join(self.out_dir, "X_val.csv"), index=False)
        X_test_feats.to_csv(os.path.join(self.out_dir, "X_test.csv"), index=False)
        
        # Save labels
        y_train.to_csv(os.path.join(self.out_dir, "y_train.csv"), index=False)
        y_val.to_csv(os.path.join(self.out_dir, "y_val.csv"), index=False)
        y_test.to_csv(os.path.join(self.out_dir, "y_test.csv"), index=False)
        
        # Save metadata for traceability
        X_train[metadata_cols].to_csv(os.path.join(self.out_dir, "meta_train.csv"), index=False)
        X_val[metadata_cols].to_csv(os.path.join(self.out_dir, "meta_val.csv"), index=False)
        X_test[metadata_cols].to_csv(os.path.join(self.out_dir, "meta_test.csv"), index=False)
        
        print(f"ML datasets generated successfully in {self.out_dir}/")

if __name__ == "__main__":
    builder = MLDatasetBuilder()
    builder.build_dataset()
