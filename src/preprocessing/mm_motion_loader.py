import os
import pandas as pd
from typing import Optional
import zipfile
from src.preprocessing.base_loader import BaseDatasetLoader, DataType

class MMMotionLoader(BaseDatasetLoader):
    """
    Loader for the MM-Motion dataset.
    Streams directly from zip files (subject01.zip, etc.) to prevent massive disk usage.
    """
    
    def __init__(self, raw_data_path: str = "datasets/raw/MM-Motion"):
        super().__init__(raw_data_path)
        
    @property
    def data_type(self) -> DataType:
        return DataType.HIGH_DENSITY
    
    def load_data(self) -> pd.DataFrame:
        all_frames = []
        
        for i in range(1, 17):
            zip_filename = f"subject{i:02d}.zip"
            zip_path = os.path.join(self.raw_data_path, zip_filename)
            
            if not os.path.exists(zip_path):
                continue
                
            print(f"Processing {zip_filename}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    for name in z.namelist():
                        if not name.endswith('.csv'):
                            continue
                            
                        parts = name.split('/')
                        if len(parts) >= 3:
                            subject_str = parts[0]
                            pose_str = parts[1]
                            trial_str = parts[2]
                            
                            with z.open(name) as f:
                                # For development, we can limit nrows to avoid OOM if needed, but let's try full
                                df = pd.read_csv(f, nrows=1000) # Only load first 1000 per trial for speed right now
                                
                                df["dataset_name"] = "MM-Motion"
                                df["subject_id"] = subject_str
                                df["session_id"] = "1"
                                df["trial_id"] = trial_str
                                df["activity"] = pose_str
                                df["sensor_id"] = "mat"
                                
                                if "f_timestamp" in df.columns:
                                    df["timestamp"] = df["f_timestamp"]
                                else:
                                    df["timestamp"] = range(len(df))
                                    
                                rename_map = {f"R{j}(g)": f"R{j}" for j in range(1, 49)}
                                rename_map.update({f"L{j}(g)": f"L{j}" for j in range(1, 49)})
                                df = df.rename(columns=rename_map)
                                
                                expected = self.get_expected_columns()
                                available = [c for c in expected if c in df.columns]
                                all_frames.append(df[available])
            except Exception as e:
                print(f"Error reading {zip_path}: {e}")
                
        if not all_frames:
            return pd.DataFrame(columns=self.get_expected_columns())
            
        return pd.concat(all_frames, ignore_index=True)

if __name__ == "__main__":
    loader = MMMotionLoader()
    df = loader.load_data()
    print("MM-Motion loader initialized. Found schema:", df.columns.tolist())
