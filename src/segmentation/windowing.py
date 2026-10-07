import numpy as np
import pandas as pd
from typing import Tuple, List

class TimeSeriesSegmenter:
    """
    Handles sliding window segmentation for continuous time-series data.
    """
    def __init__(self, window_size_sec: float = 2.0, overlap_pct: float = 0.5, fs: float = 100.0):
        self.window_size_sec = window_size_sec
        self.overlap_pct = overlap_pct
        self.fs = fs
        
        self.window_samples = int(self.window_size_sec * self.fs)
        self.step_size = int(self.window_samples * (1.0 - self.overlap_pct))

    def segment(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Segments a continuous DataFrame into smaller windows.
        Assumes the DataFrame is a single continuous trial from one sensor.
        """
        windows = []
        n_samples = len(df)
        
        if n_samples < self.window_samples:
            # Drop segments that are too short to form a single window
            return windows
            
        for start_idx in range(0, n_samples - self.window_samples + 1, self.step_size):
            end_idx = start_idx + self.window_samples
            window_df = df.iloc[start_idx:end_idx].copy()
            windows.append(window_df)
            
        return windows

    def segment_dataset(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Groups data by trial/session/sensor to ensure windows do not cross trial boundaries.
        Returns a list of window DataFrames.
        """
        if df.empty:
            return []
            
        all_windows = []
        # Group by the unique identifiers of a continuous recording
        group_cols = ["subject_id", "session_id", "trial_id", "sensor_id"]
        
        # Verify columns exist
        missing = [c for c in group_cols if c not in df.columns]
        if missing:
            # If grouping columns are missing, treat as one continuous sequence
            return self.segment(df)
            
        for _, group_df in df.groupby(group_cols):
            group_df = group_df.sort_values(by="timestamp").reset_index(drop=True)
            group_windows = self.segment(group_df)
            all_windows.extend(group_windows)
            
        return all_windows
