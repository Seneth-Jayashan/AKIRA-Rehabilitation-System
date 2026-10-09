import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

class SignalFilter:
    """
    Handles dataset-specific signal filtering (IMU vs Biomechanical).
    """

    @staticmethod
    def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
        """
        Applies a zero-phase Butterworth low-pass filter.
        Used to remove high-frequency noise from sensors.
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        # Apply filter forwards and backwards to avoid phase shift (zero-phase)
        y = filtfilt(b, a, data)
        return y

    @classmethod
    def filter_imu(cls, df: pd.DataFrame, fs: float) -> pd.DataFrame:
        """
        Filters IMU data (accelerometer and gyroscope).
        Commonly uses a low-pass filter (e.g., 20Hz cutoff for human motion).
        """
        filtered_df = df.copy()
        cutoff_freq = 20.0 # Human motion is typically < 20Hz
        
        imu_cols = ["ax", "ay", "az", "gx", "gy", "gz"]
        for col in imu_cols:
            if col in filtered_df.columns and not filtered_df[col].isnull().all():
                # Fill missing temporarily for continuous filtering if any
                signal = filtered_df[col].interpolate().bfill().ffill().values
                filtered_df[col] = cls.butter_lowpass_filter(signal, cutoff=cutoff_freq, fs=fs)
                
        return filtered_df

    @classmethod
    def filter_biomechanical(cls, df: pd.DataFrame, fs: float) -> pd.DataFrame:
        """
        Filters Biomechanical data (Force Plate, EMG).
        EMG usually requires bandpass + rectification + lowpass envelope, 
        but we'll apply a standard low-pass for Force/CoP for now.
        """
        filtered_df = df.copy()
        
        # Force/CoP typically needs lower cutoff to remove impact vibrations
        force_cutoff = 15.0
        force_cols = ["fx", "fy", "fz", "mx", "my", "mz", "cop_x", "cop_y", "cop_z"]
        for col in force_cols:
            if col in filtered_df.columns and not filtered_df[col].isnull().all():
                signal = filtered_df[col].interpolate().bfill().ffill().values
                filtered_df[col] = cls.butter_lowpass_filter(signal, cutoff=force_cutoff, fs=fs)
                
        # EMG requires different handling (often high-pass to remove DC, then rectify, then low-pass)
        if "emg" in filtered_df.columns and not filtered_df[col].isnull().all():
            pass # TODO: Implement EMG-specific envelope filtering (e.g., 20-400Hz bandpass, rectify, 6Hz lowpass)

        return filtered_df
