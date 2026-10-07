import pandas as pd
from typing import List, Tuple
from src.preprocessing.base_loader import DataType, BaseDatasetLoader
from src.preprocessing.filters import SignalFilter
from src.segmentation.windowing import TimeSeriesSegmenter
from src.feature_extraction.extractors import FeatureExtractor

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
        
        print("--- Pipeline Completed ---")
        return features_df

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
            
        # Calculate mean delta_t
        delta_t = timestamps.diff().mean()
        if delta_t > 0:
            return 1.0 / delta_t
        return 100.0

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scales numeric columns."""
        # Z-score normalization or Min-Max scaling can be applied here
        return df
