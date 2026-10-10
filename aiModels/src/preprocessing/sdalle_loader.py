import os
import pandas as pd
from typing import Optional
from src.preprocessing.base_loader import BaseDatasetLoader, DataType

class SDALLELoader(BaseDatasetLoader):
    """
    Loader for the SDALLE dataset.
    Expects data to be extracted to datasets/raw/SDALLE/Dataset_SDALLE/
    or reads directly if possible.
    """
    
    def __init__(self, raw_data_path: str = "datasets/raw/SDALLE"):
        super().__init__(raw_data_path)
        
    @property
    def data_type(self) -> DataType:
        return DataType.IMU
    
    def load_data(self) -> pd.DataFrame:
        all_frames = []
        
        # Check for extracted folder
        extracted_path = os.path.join(self.raw_data_path, "Dataset_SDALLE")
        
        if not os.path.exists(extracted_path):
            print(f"Warning: SDALLE extracted directory not found at {extracted_path}.")
            print("Please extract the Dataset_SDALLE.rar file manually.")
            return pd.DataFrame(columns=self.get_expected_columns())
            
        print("Parsing SDALLE extracted directory...")
        for root, _, files in os.walk(extracted_path):
            for file in files:
                if not file.endswith('.csv'):
                    continue
                    
                path = os.path.join(root, file)
                parts = path.replace(extracted_path, "").strip("/\\").split(os.sep)
                
                if len(parts) >= 3:
                    subject_id = parts[0]
                    activity = parts[1]
                    trial_id = parts[2].replace(".csv", "")
                    
                    try:
                        # SDALLE contains Trigno IMU data without standard headers
                        # First 7 rows are metadata. There are 54 columns. 
                        # We'll take columns 1 to 6 as ax, ay, az, gx, gy, gz for the primary sensor.
                        # (Column 0 is an EMG channel)
                        df = pd.read_csv(path, skiprows=7, header=None)
                        
                        df["dataset_name"] = "SDALLE"
                        df["subject_id"] = subject_id
                        df["session_id"] = "1"
                        df["trial_id"] = trial_id
                        df["activity"] = activity
                        
                        # Trigno CSVs are aligned to the highest freq channel (EMG @ 1259.2593 Hz)
                        df["timestamp"] = [i * (1.0 / 1259.2593) for i in range(len(df))]
                        df["sensor_id"] = "IMU_1"
                        
                        # Map columns 1-6 to accelerometer and gyroscope and force numeric
                        if df.shape[1] >= 7:
                            df["ax"] = pd.to_numeric(df[1], errors='coerce')
                            df["ay"] = pd.to_numeric(df[2], errors='coerce')
                            df["az"] = pd.to_numeric(df[3], errors='coerce')
                            df["gx"] = pd.to_numeric(df[4], errors='coerce')
                            df["gy"] = pd.to_numeric(df[5], errors='coerce')
                            df["gz"] = pd.to_numeric(df[6], errors='coerce')
                        
                        # Downsample from ~1259 Hz to ~157 Hz to match IMU actual frequency 
                        # and prevent pipeline memory overflow / >1000Hz heuristic bugs
                        df = df.iloc[::8].reset_index(drop=True)
                        
                        expected = self.get_expected_columns()
                        available = [c for c in expected if c in df.columns]
                        
                        all_frames.append(df[available])
                    except Exception as e:
                        print(f"Error parsing {path}: {e}")
                        
        if not all_frames:
            return pd.DataFrame(columns=self.get_expected_columns())
            
        return pd.concat(all_frames, ignore_index=True)

if __name__ == "__main__":
    loader = SDALLELoader()
    df = loader.load_data()
    print(f"SDALLE Loaded {len(df)} rows.")
