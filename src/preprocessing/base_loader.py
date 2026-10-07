import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional

class BaseDatasetLoader(ABC):
    """
    Abstract base class for all AKIRA dataset loaders.
    Enforces a common representation for downstream models.
    """
    
    COMMON_COLUMNS = [
        "dataset_name",
        "subject_id",
        "session_id",
        "trial_id",
        "timestamp",
        "sensor_id",
        "ax", "ay", "az",  # Accelerometer (g)
        "gx", "gy", "gz",  # Gyroscope (deg/s)
        "activity"         # Ground truth label
    ]

    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset and converts it to the common representation format.
        Must return a pandas DataFrame with columns matching COMMON_COLUMNS.
        """
        pass
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validates if the generated DataFrame conforms to the common schema.
        """
        missing_cols = set(self.COMMON_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in unified schema: {missing_cols}")
        return True
