import os
import pandas as pd
from src.preprocessing.phytmo_loader import PHYTMOLoader
from src.preprocessing.mm_motion_loader import MMMotionLoader
from src.preprocessing.sdalle_loader import SDALLELoader
from src.preprocessing.ankle_image_loader import AnkleImageLoader
from src.preprocessing.pipeline import PreprocessingPipeline

def process_dataset(loader, name):
    print(f"\n--- Starting Full Processing for {name} ---")
    pipeline = PreprocessingPipeline(loader)
    
    # Run the entire pipeline (loads, filters, normalizes, segments, extracts features)
    features_df = pipeline.run()
    
    # Save the processed ML-ready features
    out_dir = "datasets/processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name.lower()}_features.csv")
    
    features_df.to_csv(out_path, index=False)
    print(f"Successfully saved {len(features_df)} feature rows to {out_path}")

def main():
    datasets = [
        (PHYTMOLoader(), "PHYTMO"),
        (MMMotionLoader(), "MM-Motion"),
        (SDALLELoader(), "SDALLE"),
        (AnkleImageLoader(), "AnkleImage")
    ]
    
    for loader, name in datasets:
        process_dataset(loader, name)

if __name__ == "__main__":
    main()
