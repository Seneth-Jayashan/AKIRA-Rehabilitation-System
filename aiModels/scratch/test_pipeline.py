from src.preprocessing.phytmo_loader import PHYTMOLoader
from src.preprocessing.ankle_image_loader import AnkleImageLoader
from src.preprocessing.pipeline import PreprocessingPipeline

print("=== Testing IMU Pipeline ===")
phytmo = PHYTMOLoader()
pipeline_imu = PreprocessingPipeline(phytmo)
# df_imu = pipeline_imu.run() # commented out to avoid loading 18M rows in test

print("\n=== Testing Biomechanical Pipeline ===")
ankle = AnkleImageLoader()
pipeline_bio = PreprocessingPipeline(ankle)
df_bio = pipeline_bio.run()
print(df_bio.head())
