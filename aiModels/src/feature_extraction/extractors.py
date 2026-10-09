import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.signal import welch

class FeatureExtractor:
    """
    Extracts time-domain and frequency-domain features from segmented data windows.
    """
    
    @staticmethod
    def extract_time_domain(signal: np.ndarray) -> dict:
        """
        Extracts standard time-domain statistical features.
        """
        if len(signal) == 0:
            return {}
            
        # Zero-crossing rate calculation
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        zcr = len(zero_crossings) / len(signal)
        
        # Root Mean Square
        rms = np.sqrt(np.mean(signal**2))
        
        return {
            "mean": np.mean(signal),
            "std": np.std(signal),
            "var": np.var(signal),
            "max": np.max(signal),
            "min": np.min(signal),
            "range": np.ptp(signal),
            "rms": rms,
            "zcr": zcr
        }

    @staticmethod
    def extract_frequency_domain(signal: np.ndarray, fs: float) -> dict:
        """
        Extracts frequency-domain features using Welch's method for Power Spectral Density (PSD).
        """
        if len(signal) < 256: 
            # Fallback for very small windows where Welch might fail
            freqs = np.fft.rfftfreq(len(signal), 1/fs)
            psd = np.abs(np.fft.rfft(signal))**2
        else:
            freqs, psd = welch(signal, fs=fs, nperseg=256)
            
        dominant_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0
        spectral_energy = np.sum(psd)
        
        # Frequency entropy
        psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
        freq_entropy = entropy(psd_norm) if len(psd_norm) > 0 else 0
        
        return {
            "dom_freq": dominant_freq,
            "spec_energy": spectral_energy,
            "freq_entropy": freq_entropy
        }

    @classmethod
    def extract_features_from_window(cls, window_df: pd.DataFrame, feature_cols: list, fs: float) -> dict:
        """
        Extracts features from a single Pandas DataFrame window for all specified columns.
        """
        features = {}
        
        # Preserve metadata (assuming constant within the window)
        metadata_cols = ["dataset_name", "subject_id", "session_id", "trial_id", "sensor_id", "activity"]
        for m in metadata_cols:
            if m in window_df.columns:
                # Use majority voting for activity, mode for others
                features[m] = window_df[m].mode()[0] if not window_df[m].empty else None
                
        for col in feature_cols:
            if col not in window_df.columns or window_df[col].isnull().all():
                continue
                
            signal = window_df[col].values
            
            # Time Domain
            td_feats = cls.extract_time_domain(signal)
            for k, v in td_feats.items():
                features[f"{col}_{k}"] = v
                
            # Frequency Domain
            fd_feats = cls.extract_frequency_domain(signal, fs=fs)
            for k, v in fd_feats.items():
                features[f"{col}_{k}"] = v
                
        return features

    @classmethod
    def process_windows(cls, windows: list, feature_cols: list, fs: float) -> pd.DataFrame:
        """
        Converts a list of window DataFrames into a single Feature DataFrame suitable for ML.
        """
        feature_rows = []
        for w in windows:
            feats = cls.extract_features_from_window(w, feature_cols, fs)
            if feats:
                feature_rows.append(feats)
                
        return pd.DataFrame(feature_rows)
