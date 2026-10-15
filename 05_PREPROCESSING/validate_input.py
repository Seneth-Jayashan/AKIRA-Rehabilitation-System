def validate_imu_data(df):
    print("Validating input data...")
    required = ['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    return df
