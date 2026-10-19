import os
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from src.preprocessing.base_loader import BaseDatasetLoader, DataType

class AkiraDatasetLoader(BaseDatasetLoader):
    """
    Loader and validation pipeline for the AKIRA 4-Member Rehabilitation Data Collection platform.
    Processes synchronized dual-IMU (12-channel) recordings exported from the
    AkiraDataCollector mobile app:
    - Member 1 (My Component - Ankle): Sensor 1 = Tibia,     Sensor 2 = Foot
    - Member 2 (Knee Component):       Sensor 1 = Thigh,     Sensor 2 = Shank
    - Member 3 (Wrist Component):      Sensor 1 = Wrist,     Sensor 2 = Hand
    - Member 4 (Elbow Component):      Sensor 1 = Upper Arm, Sensor 2 = Forearm
    """

    COMPONENT_SENSOR_MAP = {
        "ANKLE": ("tibia", "foot"),
        "KNEE":  ("thigh", "shank"),
        "WRIST": ("wrist", "hand"),
        "ELBOW": ("upper_arm", "forearm")
    }

    # 12 Motion channels captured by AKIRA hardware
    S1_CHANNELS = ["s1_ax", "s1_ay", "s1_az", "s1_gx", "s1_gy", "s1_gz"]
    S2_CHANNELS = ["s2_ax", "s2_ay", "s2_az", "s2_gx", "s2_gy", "s2_gz"]
    ALL_12_CHANNELS = S1_CHANNELS + S2_CHANNELS

    def __init__(
        self,
        raw_data_path: str = "datasets/raw/AKIRA",
        target_component: Optional[str] = "ANKLE",
        reshape_to_sensor_rows: bool = True
    ):
        """
        Args:
            raw_data_path: Path to an individual CSV file or directory containing exported AKIRA CSVs.
            target_component: "ANKLE" (Member 1 - default), "KNEE", "WRIST", "ELBOW", or None for all.
            reshape_to_sensor_rows: If True, splits into standard BaseDatasetLoader format with
                                   sensor_id=(proximal, distal) for pipeline compatibility.
        """
        super().__init__(raw_data_path)
        self.target_component = target_component.upper() if target_component else None
        self.reshape_to_sensor_rows = reshape_to_sensor_rows

    @property
    def data_type(self) -> DataType:
        return DataType.IMU

    def load_data(self) -> pd.DataFrame:
        csv_files = []
        if os.path.isfile(self.raw_data_path):
            csv_files = [self.raw_data_path]
        elif os.path.isdir(self.raw_data_path):
            csv_files = glob.glob(os.path.join(self.raw_data_path, "**", "*.csv"), recursive=True)
            csv_files = [f for f in csv_files if not f.endswith("_metadata.csv")]

        if not csv_files:
            print(f"Warning: No valid AKIRA CSV files found in {self.raw_data_path}")
            return pd.DataFrame(columns=self.get_expected_columns())

        frames = []
        for file_path in csv_files:
            try:
                df_raw = pd.read_csv(file_path)
                df_parsed = self._parse_single_file(df_raw, file_path)
                if df_parsed is not None and not df_parsed.empty:
                    frames.append(df_parsed)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        if not frames:
            return pd.DataFrame(columns=self.get_expected_columns())

        full_df = pd.concat(frames, ignore_index=True)
        if self.reshape_to_sensor_rows:
            self.validate_schema(full_df)
        return full_df

    def _parse_single_file(self, df_raw: pd.DataFrame, file_path: str) -> Optional[pd.DataFrame]:
        df_raw.columns = [c.strip().lower() for c in df_raw.columns]

        # Timestamp column check
        ts_col = "timestamp_ms" if "timestamp_ms" in df_raw.columns else "timestamp"
        if ts_col not in df_raw.columns:
            return None

        component = str(df_raw["component"].iloc[0]).upper() if "component" in df_raw.columns else "ANKLE"

        # Filter by target component if specified
        if self.target_component and component != self.target_component:
            return None

        subject_id = df_raw["participant_id"].iloc[0] if "participant_id" in df_raw.columns else "P001"
        session_id = str(df_raw["session_id"].iloc[0]) if "session_id" in df_raw.columns else os.path.basename(file_path).split('.')[0]
        trial_id = str(df_raw["trial_id"].iloc[0]) if "trial_id" in df_raw.columns else "Trial_1"
        activity = str(df_raw["exercise"].iloc[0]) if "exercise" in df_raw.columns else "UNKNOWN"

        ts_seconds = pd.to_numeric(df_raw[ts_col], errors='coerce') / 1000.0

        if not self.reshape_to_sensor_rows:
            df_wide = df_raw.copy()
            df_wide["dataset_name"] = "AKIRA"
            df_wide["subject_id"] = subject_id
            df_wide["session_id"] = session_id
            df_wide["trial_id"] = trial_id
            df_wide["timestamp"] = ts_seconds
            df_wide["activity"] = activity
            df_wide["component"] = component
            return df_wide

        # Resolve sensor IDs based on component
        s1_name, s2_name = self.COMPONENT_SENSOR_MAP.get(component, ("sensor_1", "sensor_2"))

        # Sensor 1 (Proximal: Tibia / Thigh / Wrist / Upper Arm)
        s1_df = pd.DataFrame()
        s1_df["dataset_name"] = ["AKIRA"] * len(df_raw)
        s1_df["subject_id"] = [subject_id] * len(df_raw)
        s1_df["session_id"] = [session_id] * len(df_raw)
        s1_df["trial_id"] = [trial_id] * len(df_raw)
        s1_df["timestamp"] = ts_seconds
        s1_df["sensor_id"] = [s1_name] * len(df_raw)
        s1_df["activity"] = [activity] * len(df_raw)

        s1_df["ax"] = pd.to_numeric(df_raw.get("s1_ax", df_raw.get("tibia_ax", df_raw.get("ax", 0.0))), errors='coerce')
        s1_df["ay"] = pd.to_numeric(df_raw.get("s1_ay", df_raw.get("tibia_ay", df_raw.get("ay", 0.0))), errors='coerce')
        s1_df["az"] = pd.to_numeric(df_raw.get("s1_az", df_raw.get("tibia_az", df_raw.get("az", 0.0))), errors='coerce')
        s1_df["gx"] = pd.to_numeric(df_raw.get("s1_gx", df_raw.get("tibia_gx", df_raw.get("gx", 0.0))), errors='coerce')
        s1_df["gy"] = pd.to_numeric(df_raw.get("s1_gy", df_raw.get("tibia_gy", df_raw.get("gy", 0.0))), errors='coerce')
        s1_df["gz"] = pd.to_numeric(df_raw.get("s1_gz", df_raw.get("tibia_gz", df_raw.get("gz", 0.0))), errors='coerce')

        # Sensor 2 (Distal: Foot / Shank / Hand / Forearm)
        s2_df = pd.DataFrame()
        s2_df["dataset_name"] = ["AKIRA"] * len(df_raw)
        s2_df["subject_id"] = [subject_id] * len(df_raw)
        s2_df["session_id"] = [session_id] * len(df_raw)
        s2_df["trial_id"] = [trial_id] * len(df_raw)
        s2_df["timestamp"] = ts_seconds
        s2_df["sensor_id"] = [s2_name] * len(df_raw)
        s2_df["activity"] = [activity] * len(df_raw)

        s2_df["ax"] = pd.to_numeric(df_raw.get("s2_ax", df_raw.get("foot_ax", 0.0)), errors='coerce')
        s2_df["ay"] = pd.to_numeric(df_raw.get("s2_ay", df_raw.get("foot_ay", 0.0)), errors='coerce')
        s2_df["az"] = pd.to_numeric(df_raw.get("s2_az", df_raw.get("foot_az", 0.0)), errors='coerce')
        s2_df["gx"] = pd.to_numeric(df_raw.get("s2_gx", df_raw.get("foot_gx", 0.0)), errors='coerce')
        s2_df["gy"] = pd.to_numeric(df_raw.get("s2_gy", df_raw.get("foot_gy", 0.0)), errors='coerce')
        s2_df["gz"] = pd.to_numeric(df_raw.get("s2_gz", df_raw.get("foot_gz", 0.0)), errors='coerce')

        return pd.concat([s1_df, s2_df], ignore_index=True)

    @staticmethod
    def validate_recording_quality(file_path: str) -> Dict:
        df = pd.read_csv(file_path)
        df.columns = [c.strip().lower() for c in df.columns]

        comp = df.get("component", pd.Series(["ANKLE"])).iloc[0]
        report = {
            "file": os.path.basename(file_path),
            "component": comp,
            "total_samples": len(df),
            "missing_values": int(df.isnull().sum().sum()),
            "s1_channels_present": any(c in df.columns for c in ["s1_ax", "tibia_ax"]),
            "s2_channels_present": any(c in df.columns for c in ["s2_ax", "foot_ax"]),
            "sampling_frequency_hz": 0.0,
            "packet_drop_estimate": 0,
            "max_timestamp_gap_ms": 0.0,
            "status": "PASS"
        }

        ts_col = "timestamp_ms" if "timestamp_ms" in df.columns else "timestamp"
        if ts_col in df.columns and len(df) > 1:
            ts = pd.to_numeric(df[ts_col], errors='coerce').dropna()
            diffs = ts.diff().dropna()
            median_interval = diffs.median()
            if median_interval > 0:
                report["sampling_frequency_hz"] = round(1000.0 / median_interval, 2)
                gap_threshold = median_interval * 1.5
                dropped_gaps = diffs[diffs > gap_threshold]
                estimated_drops = sum((g / median_interval) - 1 for g in dropped_gaps)
                report["packet_drop_estimate"] = int(round(estimated_drops))
                report["max_timestamp_gap_ms"] = float(diffs.max())

        accel_cols = [c for c in ["s1_ax", "s1_ay", "s1_az", "s2_ax", "s2_ay", "s2_az", "tibia_ax", "foot_ax"] if c in df.columns]
        for ac in accel_cols:
            s_num = pd.to_numeric(df[ac], errors='coerce')
            if (s_num.abs() >= 1.99).any():
                report["status"] = "WARNING_CLIPPING"

        if report["missing_values"] > 0 or not report["s1_channels_present"] or not report["s2_channels_present"]:
            report["status"] = "FAIL"

        return report

    @staticmethod
    def create_subject_independent_splits(
        df: pd.DataFrame,
        test_subjects: Optional[List[str]] = None,
        val_subjects: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        all_subjects = df["subject_id"].unique().tolist()
        if len(all_subjects) < 2:
            print(f"Warning: Only 1 subject ({all_subjects}) present. Cannot perform participant-independent split.")
            return df, pd.DataFrame(), pd.DataFrame()

        if test_subjects is None:
            test_subjects = [all_subjects[-1]]

        remaining = [s for s in all_subjects if s not in test_subjects]

        if val_subjects is None:
            if len(remaining) > 1:
                val_subjects = [remaining[-1]]
                train_subjects = [s for s in remaining if s not in val_subjects]
            else:
                val_subjects = []
                train_subjects = remaining
        else:
            train_subjects = [s for s in remaining if s not in val_subjects]

        train_df = df[df["subject_id"].isin(train_subjects)].reset_index(drop=True)
        val_df   = df[df["subject_id"].isin(val_subjects)].reset_index(drop=True)
        test_df  = df[df["subject_id"].isin(test_subjects)].reset_index(drop=True)

        return train_df, val_df, test_df
