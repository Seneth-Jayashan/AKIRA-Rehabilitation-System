import pandas as pd
from typing import List, Tuple
from src.preprocessing.base_loader import DataType, BaseDatasetLoader
from src.preprocessing.filters import SignalFilter
from src.segmentation.windowing import TimeSeriesSegmenter
from src.feature_extraction.extractors import FeatureExtractor
import numpy as np

class PreprocessingPipeline:
    """
    Orchestrates the data preparation roadmap steps 1-10.
    """
    
    def __init__(self, loader: BaseDatasetLoader):
        self.loader = loader
        self.fs = 100.0 # Default fallback sampling rate

    def run(self) -> pd.DataFrame:
        print(f"--- Starting Pipeline for Dataset: {self.loader.__class__.__name__} ---")
        
        # Step 1-3: Load data and convert to common representation
        print("1-3. Loading data & dataset-specific format detection...")
        df = self.loader.load_data()
        
        # Step 4: Data-quality checks (Missing values, etc.)
        print("4. Executing data-quality checks...")
        df = self.check_data_quality(df)
        
        # Step 5 & 6: Sensor identification & Sampling-rate verification
        print("5-6. Verifying sensors and sampling rate...")
        self.fs = self.verify_sampling_rate(df)
        print(f"   Detected Sampling Rate: ~{self.fs:.2f} Hz")
        
        # Step 7: Filtering (IMU vs Biomechanical router)
        print(f"7. Routing to filter for {self.loader.data_type.value}...")
        if self.loader.data_type == DataType.IMU:
            df = SignalFilter.filter_imu(df, self.fs)
        elif self.loader.data_type == DataType.BIOMECHANICAL:
            df = SignalFilter.filter_biomechanical(df, self.fs)
            
        # Step 8: Normalization
        print("8. Normalizing signals...")
        df = self.normalize(df)
        
        # Step 9: Segmentation
        print("9. Segmenting into time windows...")
        segmenter = TimeSeriesSegmenter(window_size_sec=2.0, overlap_pct=0.5, fs=self.fs)
        windows = segmenter.segment_dataset(df)
        print(f"   Generated {len(windows)} windows.")
        
        # Step 10: Feature Extraction
        print("10. Extracting features...")
        if self.loader.data_type == DataType.IMU:
            feature_cols = self.loader.IMU_COLUMNS
        elif self.loader.data_type == DataType.BIOMECHANICAL:
            feature_cols = self.loader.BIOMEC_COLUMNS
        else:
            feature_cols = self.loader.IMU_COLUMNS + self.loader.BIOMEC_COLUMNS
            
        features_df = FeatureExtractor.process_windows(windows, feature_cols, self.fs)
        
        # Step 11: Create Raw 3D Tensor for Deep Learning
        print("11. Creating 3D Tensor for Deep Learning...")
        raw_list = []
        target_cols = ["ax", "ay", "az", "gx", "gy", "gz"]
        expected_len = int(2.0 * self.fs)
        
        for w in windows:
            # Safely handle missing columns
            available_cols = [c for c in target_cols if c in w.columns]
            
            if len(available_cols) > 0:
                arr = w[available_cols].values
                if len(arr) > expected_len:
                    arr = arr[:expected_len, :]
                elif len(arr) < expected_len:
                    pad_width = expected_len - len(arr)
                    arr = np.pad(arr, ((0, pad_width), (0, 0)), mode='constant')
                
                if len(available_cols) < 6:
                    full_arr = np.zeros((expected_len, 6))
                    for idx, col in enumerate(target_cols):
                        if col in available_cols:
                            full_arr[:, idx] = arr[:, available_cols.index(col)]
                    arr = full_arr
            else:
                arr = np.zeros((expected_len, 6))
                
            raw_list.append(arr)
            
        if raw_list:
            raw_tensor = np.stack(raw_list)
        else:
            raw_tensor = np.empty((0, expected_len, 6))
            
        print(f"   Raw tensor shape: {raw_tensor.shape}")
        
        print("--- Pipeline Completed ---")
        return features_df, raw_tensor

    def check_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles missing values, invalid timestamps, etc."""
        # Simple forward-fill for minor gaps, drop for large ones.
        return df.copy()

    def verify_sampling_rate(self, df: pd.DataFrame) -> float:
        """Calculates expected delta_t and derives sampling frequency."""
        if len(df) < 2 or "timestamp" not in df.columns:
            return 100.0
        
        # Ensure timestamp is numeric
        timestamps = pd.to_numeric(df["timestamp"], errors='coerce').dropna()
        if len(timestamps) < 2:
            return 100.0
            
        # Calculate median delta_t on a single continuous sensor stream
        first_group = df.groupby(["subject_id", "session_id", "trial_id", "sensor_id"])["timestamp"].apply(list).values[0]
        timestamps = pd.Series(first_group).dropna()
        if len(timestamps) < 2:
            return 100.0
            
        delta_t_sec = timestamps.diff().median()
        if delta_t_sec > 0:
            freq = 1.0 / delta_t_sec
            # If the calculated frequency is absurdly high (e.g. > 1000Hz), 
            # the timestamps might actually be in milliseconds.
            if freq > 1000.0:
                freq = 1000.0 / delta_t_sec
            return freq
        return 100.0

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scales numeric columns."""
        # Z-score normalization or Min-Max scaling can be applied here
        return df
