import os
import pandas as pd
from typing import Optional
from src.preprocessing.base_loader import BaseDatasetLoader

class PHYTMOLoader(BaseDatasetLoader):
    """
    Loader for the PHYTMO dataset.
    Extracts data from the 'inertial' CSV files and aligns them to the common schema.
    """
    
    def __init__(self, raw_data_path: str = "datasets/raw/PHYTMO"):
        super().__init__(raw_data_path)
    
    def load_data(self) -> pd.DataFrame:
        inertial_path = os.path.join(self.raw_data_path, "inertial")
        all_frames = []
        
        # Traverse the inertial directory structure
        # Structure is usually: inertial/lower/A/Lshin/*.csv
        for root, _, files in os.walk(inertial_path):
            for file in files:
                if not file.endswith(".csv"):
                    continue
                
                # File naming convention: e.g. A01GAT0_1.csv
                # subject: A01, activity: GAT0, trial: 1
                filename = os.path.splitext(file)[0]
                
                # We can extract subject and trial using string slicing assuming 'A' + 2 digits is subject
                if len(filename) >= 6:
                    subject_id = filename[:3]  # e.g., 'A01'
                    
                    # Split on '_' for trial id if it exists
                    parts = filename.split('_')
                    if len(parts) == 2:
                        activity_code = parts[0][3:] # GAT0
                        trial_id = parts[1]
                    else:
                        activity_code = filename[3:]
                        trial_id = "1"
                        
                    # Extract sensor location from the directory path (e.g. Lshin, Rshin)
                    sensor_id = os.path.basename(root)
                    
                    file_path = os.path.join(root, file)
                    
                    try:
                        # PHYTMO columns: Time (s),Gyroscope X (deg/s),...,Accelerometer X (g),...,Magnetometer Z (uT)
                        df_raw = pd.read_csv(file_path)
                        
                        df_unified = pd.DataFrame()
                        df_unified["dataset_name"] = ["PHYTMO"] * len(df_raw)
                        df_unified["subject_id"] = [subject_id] * len(df_raw)
                        df_unified["session_id"] = ["1"] * len(df_raw) # Session not explicitly in filename
                        df_unified["trial_id"] = [trial_id] * len(df_raw)
                        df_unified["timestamp"] = df_raw["Time (s)"]
                        df_unified["sensor_id"] = [sensor_id] * len(df_raw)
                        
                        # Mapping sensors
                        df_unified["ax"] = df_raw["Accelerometer X (g)"]
                        df_unified["ay"] = df_raw["Accelerometer Y (g)"]
                        df_unified["az"] = df_raw["Accelerometer Z (g)"]
                        df_unified["gx"] = df_raw["Gyroscope X (deg/s)"]
                        df_unified["gy"] = df_raw["Gyroscope Y (deg/s)"]
                        df_unified["gz"] = df_raw["Gyroscope Z (deg/s)"]
                        
                        df_unified["activity"] = [activity_code] * len(df_raw)
                        
                        all_frames.append(df_unified)
                        
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        
        if not all_frames:
            print(f"Warning: No valid CSV files found in {inertial_path}")
            return pd.DataFrame(columns=self.COMMON_COLUMNS)
            
        full_df = pd.concat(all_frames, ignore_index=True)
        self.validate_schema(full_df)
        return full_df

if __name__ == "__main__":
    # Test script
    print("Loading PHYTMO Dataset...")
    loader = PHYTMOLoader()
    df = loader.load_data()
    print(f"Successfully loaded {len(df)} samples.")
    print(df.head())
    print("\nUnique activities:", df["activity"].unique())
