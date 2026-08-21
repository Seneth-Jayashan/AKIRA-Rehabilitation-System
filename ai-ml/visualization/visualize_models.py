import os
import sys
import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Add the 'src' directory to the path so we can import the model architectures
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from train_model import build_model
    from train_compensation_model import build_binary_model
    from train_multiclass_compensation_model import build_multiclass_model
except ImportError as e:
    print(f"Error importing model architectures: {e}")
    sys.exit(1)

# Set up output directory for the images
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Visualizing Model 1
    print("Generating visualization for Model 1 (Exercise Recognition)...")
    model1 = build_model(num_classes=4)
    plot_model(model1, 
               to_file=os.path.join(OUTPUT_DIR, 'model1_exercise_recognition.png'), 
               show_shapes=True, 
               show_layer_names=True,
               dpi=96)
    
    # Visualizing Model 2
    print("Generating visualization for Model 2 (Binary Compensation)...")
    model2 = build_binary_model()
    plot_model(model2, 
               to_file=os.path.join(OUTPUT_DIR, 'model2_binary_compensation.png'), 
               show_shapes=True, 
               show_layer_names=True,
               dpi=96)
    
    # Visualizing Model 3
    print("Generating visualization for Model 3 (Multi-Class Compensation)...")
    model3 = build_multiclass_model(num_classes=7)
    plot_model(model3, 
               to_file=os.path.join(OUTPUT_DIR, 'model3_multiclass_compensation.png'), 
               show_shapes=True, 
               show_layer_names=True,
               dpi=96)
    
    print(f"Done! Visualizations saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
