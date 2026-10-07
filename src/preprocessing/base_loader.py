import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional
from enum import Enum

class DataType(Enum):
    IMU = "IMU"
    BIOMECHANICAL = "Biomechanical"
    MIXED = "Mixed"

class BaseDatasetLoader(ABC):
    """
    Abstract base class for all AKIRA dataset loaders.
    Enforces a common representation for downstream models and routing based on data type.
    """
    
    # Base columns applicable to ALL data types
    BASE_COLUMNS = [
        "dataset_name",
        "subject_id",
        "session_id",
        "trial_id",
        "timestamp",
        "sensor_id",
        "activity"         # Ground truth label
    ]

    # IMU-specific columns
    IMU_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]
    
    # Biomechanical-specific columns (Force plate, EMG, etc.)
    BIOMEC_COLUMNS = ["fx", "fy", "fz", "mx", "my", "mz", "cop_x", "cop_y", "cop_z", "emg"]

    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path

    @property
    @abstractmethod
    def data_type(self) -> DataType:
        """Returns the primary data type of this loader (IMU or Biomechanical)"""
        pass

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset and converts it to the common representation format.
        """
        pass
    
    def get_expected_columns(self) -> List[str]:
        if self.data_type == DataType.IMU:
            return self.BASE_COLUMNS + self.IMU_COLUMNS
        elif self.data_type == DataType.BIOMECHANICAL:
            return self.BASE_COLUMNS + self.BIOMEC_COLUMNS
        else:
            return self.BASE_COLUMNS + self.IMU_COLUMNS + self.BIOMEC_COLUMNS

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validates if the generated DataFrame conforms to the expected schema based on data_type.
        """
        expected = self.get_expected_columns()
        missing_cols = set(expected) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns for {self.data_type.value} schema: {missing_cols}")
        return True
