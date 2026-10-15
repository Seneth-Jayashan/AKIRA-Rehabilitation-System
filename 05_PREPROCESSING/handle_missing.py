def handle_missing_values(df):
    print("Handling missing values...")
    # Do not fabricate missing IMU data. Just drop rows where raw accel/gyro is missing
    df_clean = df.dropna(subset=['ax', 'ay', 'az', 'gx', 'gy', 'gz']).copy()
    return df_clean
