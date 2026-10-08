TAXONOMY_MAPPING = {
    # Targets (Walking / Gait Proxy)
    "Walking": "Walking",
    "GAT": "Walking",
    "GAT0": "Walking",
    "GAT1": "Walking",
    "GHT": "Heel-Toe Gait",
    
    # Negatives (Squats)
    "SQT": "Squat",
    "SQT0": "Squat",
    "SQT1": "Squat",
    
    # Negatives (Knee Flexion)
    "KFE": "Knee Flexion",
    "KFEL0": "Knee Flexion",
    "KFEL1": "Knee Flexion",
    "KFER0": "Knee Flexion",
    "KFER1": "Knee Flexion",
    
    # Negatives (Hip Abduction)
    "HAA": "Hip Abduction",
    "HAAL0": "Hip Abduction",
    "HAAL1": "Hip Abduction",
    "HAAR0": "Hip Abduction",
    "HAAR1": "Hip Abduction",
    
    # Negatives (Other functional movements)
    "Jogging": "Jogging",
    "Stairs_up": "Stairs",
    "Stairs_down": "Stairs",
    "Stairs_Down": "Stairs",
    
    # Negatives (Static / Calibration / Miscellaneous)
    "Calib": "Calibration",
    "LCalib": "Calibration",
    "RCalib": "Calibration",
    "GIS": "Abnormal Gait",
    "EAH0": "Other",
    "EAH1": "Other",
    "EFE0": "Other",
    "EFE1": "Other",
    "SQZ0": "Other",
    "SQZ1": "Other"
}

def map_label(raw_label: str) -> str:
    """Maps a raw dataset label to the unified AKIRA exercise taxonomy."""
    # Convert to string just in case
    raw_label = str(raw_label).strip()
    return TAXONOMY_MAPPING.get(raw_label, "Unknown")
