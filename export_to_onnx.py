"""
Export Trained Models to ONNX Format
Converts scikit-learn and XGBoost models to ONNX for cross-platform deployment
"""

import joblib
import numpy as np
from pathlib import Path
import json

# Install required packages if not already installed
try:
    from skl2onnx import convert_sklearn, to_onnx
    from skl2onnx.common.data_types import FloatTensorType
    import onnx
    import onnxruntime as rt
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'skl2onnx', 'onnx', 'onnxruntime', 'onnxmltools'])
    from skl2onnx import convert_sklearn, to_onnx
    from skl2onnx.common.data_types import FloatTensorType
    import onnx
    import onnxruntime as rt

print("="*80)
print("EXPORTING MODELS TO ONNX FORMAT")
print("="*80)

# Paths
MODEL_DIR = Path("models")
ONNX_DIR = Path("models/onnx")
ONNX_DIR.mkdir(exist_ok=True)

# Load feature information
feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
n_features = len(feature_names)

print(f"\n📊 Number of features: {n_features}")
print(f"📁 Models directory: {MODEL_DIR}")
print(f"📁 ONNX output directory: {ONNX_DIR}")

# Define initial type for ONNX conversion
initial_type = [('float_input', FloatTensorType([None, n_features]))]

# Models to export
models_info = {
    'xgboost': {
        'file': 'best_model_xgboost.pkl',
        'name': 'XGBoost',
        'accuracy': 0.9865
    },
    'random_forest': {
        'file': 'model_random_forest.pkl',
        'name': 'Random Forest',
        'accuracy': 0.9855
    },
    'logistic_regression': {
        'file': 'model_logistic_regression.pkl',
        'name': 'Logistic Regression',
        'accuracy': 0.9520
    }
}

# Export each model
exported_models = []

for model_key, model_info in models_info.items():
    print(f"\n{'='*80}")
    print(f"Exporting: {model_info['name']}")
    print(f"{'='*80}")

    try:
        # Load model
        model_path = MODEL_DIR / model_info['file']
        print(f"Loading model from: {model_path}")
        model = joblib.load(model_path)

        # Convert to ONNX
        onnx_model_path = ONNX_DIR / f"{model_key}.onnx"
        print(f"Converting to ONNX...")

        # For XGBoost, we need special handling
        if 'xgboost' in model_key.lower():
            try:
                # XGBoost specific conversion using onnxmltools
                from onnxmltools.convert import convert_xgboost
                from onnxmltools.convert.common.data_types import FloatTensorType as OnnxmlFloatTensorType

                # Use onnxmltools FloatTensorType for XGBoost
                xgb_initial_type = [('float_input', OnnxmlFloatTensorType([None, n_features]))]

                onnx_model = convert_xgboost(
                    model,
                    initial_types=xgb_initial_type,
                    target_opset=12
                )
            except Exception as e:
                print(f"❌ XGBoost conversion failed: {e}")
                raise
        else:
            # For scikit-learn models
            onnx_model = convert_sklearn(
                model,
                initial_types=initial_type,
                target_opset=12
            )

        # Save ONNX model
        with open(onnx_model_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"✅ Saved ONNX model: {onnx_model_path}")

        # Verify the model
        print(f"Verifying ONNX model...")
        onnx_model_check = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model_check)
        print(f"✅ ONNX model is valid")

        # Test inference
        print(f"Testing inference...")
        session = rt.InferenceSession(str(onnx_model_path))
        input_name = session.get_inputs()[0].name

        # Create test input
        test_input = np.random.randn(1, n_features).astype(np.float32)

        # Run inference
        result = session.run(None, {input_name: test_input})
        print(f"✅ Inference test successful")
        print(f"   Output shape: {result[0].shape if len(result) > 0 else 'N/A'}")

        # Get model size
        model_size = onnx_model_path.stat().st_size / (1024 * 1024)  # MB
        print(f"📦 Model size: {model_size:.2f} MB")

        exported_models.append({
            'name': model_info['name'],
            'key': model_key,
            'onnx_file': f"onnx/{model_key}.onnx",
            'accuracy': model_info['accuracy'],
            'size_mb': round(model_size, 2),
            'input_name': input_name,
            'output_names': [output.name for output in session.get_outputs()]
        })

    except Exception as e:
        print(f"❌ Error exporting {model_info['name']}: {e}")
        import traceback
        traceback.print_exc()

# Export scaler to ONNX
print(f"\n{'='*80}")
print(f"Exporting: StandardScaler")
print(f"{'='*80}")

try:
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    scaler_onnx_path = ONNX_DIR / "scaler.onnx"

    print(f"Converting scaler to ONNX...")
    scaler_onnx = to_onnx(
        scaler,
        X=np.zeros((1, n_features), dtype=np.float32),
        target_opset=12
    )

    with open(scaler_onnx_path, "wb") as f:
        f.write(scaler_onnx.SerializeToString())

    print(f"✅ Saved scaler ONNX: {scaler_onnx_path}")

    # Verify
    onnx.checker.check_model(onnx.load(scaler_onnx_path))
    print(f"✅ Scaler ONNX is valid")

    scaler_size = scaler_onnx_path.stat().st_size / 1024  # KB
    print(f"📦 Scaler size: {scaler_size:.2f} KB")

except Exception as e:
    print(f"❌ Error exporting scaler: {e}")

# Save metadata
print(f"\n{'='*80}")
print(f"Saving Metadata")
print(f"{'='*80}")

metadata = {
    'export_date': '2025-11-10',
    'n_features': n_features,
    'feature_names': feature_names,
    'models': exported_models,
    'scaler': 'onnx/scaler.onnx',
    'usage': {
        'python': 'See examples/onnx_inference.py',
        'javascript': 'Use onnxruntime-web',
        'csharp': 'Use Microsoft.ML.OnnxRuntime',
        'java': 'Use ai.onnxruntime'
    }
}

metadata_path = ONNX_DIR / "models_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Saved metadata: {metadata_path}")

# Summary
print(f"\n{'='*80}")
print(f"✅ EXPORT COMPLETE!")
print(f"{'='*80}")

print(f"\n📁 ONNX Models Location: {ONNX_DIR.absolute()}")
print(f"\n📊 Exported Models:")
for model in exported_models:
    print(f"   • {model['name']}: {model['onnx_file']} ({model['size_mb']} MB)")

print(f"\n🚀 Next Steps:")
print(f"   1. Test ONNX models: python test_onnx_models.py")
print(f"   2. Use in API: Update main.py to use ONNX runtime")
print(f"   3. Deploy to edge: ONNX models work on mobile, web, IoT")
print(f"   4. Cross-platform: Use in C#, Java, JavaScript, etc.")

print(f"\n📖 ONNX Runtime Installation:")
print(f"   Python:     pip install onnxruntime")
print(f"   JavaScript: npm install onnxruntime-web")
print(f"   C#:         dotnet add package Microsoft.ML.OnnxRuntime")
print(f"   Java:       Maven/Gradle dependency")

print("\n" + "="*80)
