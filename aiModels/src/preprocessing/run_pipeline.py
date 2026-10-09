import os
import pandas as pd
import numpy as np
import numpy as np
from src.preprocessing.phytmo_loader import PHYTMOLoader
from src.preprocessing.mm_motion_loader import MMMotionLoader
from src.preprocessing.sdalle_loader import SDALLELoader
from src.preprocessing.ankle_image_loader import AnkleImageLoader
from src.preprocessing.pipeline import PreprocessingPipeline

def process_dataset(loader, name):
    print(f"\n--- Starting Full Processing for {name} ---")
    pipeline = PreprocessingPipeline(loader)
    try:
        # Run the entire pipeline (loads, filters, normalizes, segments, extracts features)
        features_df, raw_tensor = pipeline.run()
        
        if len(features_df) == 0:
            print(f"No windows generated for {name}.")
            return
            
        # Save the processed ML-ready features
        out_dir = "datasets/processed"
        os.makedirs(out_dir, exist_ok=True)
        out_path_csv = os.path.join(out_dir, f"{name.lower()}_features.csv")
        out_path_npy = os.path.join(out_dir, f"{name.lower()}_raw.npy")
        
        features_df.to_csv(out_path_csv, index=False)
        np.save(out_path_npy, raw_tensor)
        
        print(f"Successfully saved {len(features_df)} feature rows to {out_path_csv}")
        print(f"Successfully saved 3D tensor {raw_tensor.shape} to {out_path_npy}")
    except Exception as e:
        print(f"Failed to process {name}: {e}")

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
