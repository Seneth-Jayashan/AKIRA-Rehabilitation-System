import os
import joblib
import onnx
import onnxmltools
from skl2onnx.common.data_types import FloatTensorType as SklearnFloatTensorType
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType
import skl2onnx

def export_models():
    print("Exporting optimized models to ONNX for mobile inference...")
    os.makedirs("models/mobile", exist_ok=True)
    
    num_features = 66

    # 1. Export Random Forest
    rf_path = "models/optimized/best_random_forest.pkl"
    if os.path.exists(rf_path):
        print(f"Loading {rf_path}...")
        rf_model = joblib.load(rf_path)
        print("Converting Random Forest to ONNX...")
        initial_type = [('float_input', SklearnFloatTensorType([None, num_features]))]
        onnx_rf = skl2onnx.convert_sklearn(rf_model, initial_types=initial_type, target_opset=12)
        onnx.save_model(onnx_rf, "models/mobile/rehab_rf.onnx")
        print("Saved models/mobile/rehab_rf.onnx")
    
    # 2. Export XGBoost
    xgb_path = "models/optimized/best_xgboost.pkl"
    if os.path.exists(xgb_path):
        print(f"Loading {xgb_path}...")
        xgb_model = joblib.load(xgb_path)
        
        # ONNX conversion fails if XGBoost has string feature names like 'gx_spec_energy'
        # It expects f0, f1, f2, etc. So we rewrite the booster's feature names
        booster = xgb_model.get_booster()
        booster.feature_names = [f"f{i}" for i in range(num_features)]
        
        print("Converting XGBoost to ONNX...")
        # onnxmltools is required for XGBoost conversion
        initial_type = [('float_input', OnnxFloatTensorType([None, num_features]))]
        onnx_xgb = onnxmltools.convert_xgboost(xgb_model, initial_types=initial_type, target_opset=12)
        onnx.save_model(onnx_xgb, "models/mobile/rehab_xgb.onnx")
        print("Saved models/mobile/rehab_xgb.onnx")

    print("\nONNX Conversion Complete!")
    print("These .onnx files can now be embedded directly into the Android/iOS application")
    print("using ONNX Runtime for zero-latency, on-device offline inference.")

if __name__ == "__main__":
    export_models()
