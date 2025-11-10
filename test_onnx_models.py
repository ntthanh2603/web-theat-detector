"""
Test ONNX Models
Compare ONNX model predictions with original models
"""

import numpy as np
import joblib
import onnxruntime as rt
from pathlib import Path
import json

print("="*80)
print("TESTING ONNX MODELS")
print("="*80)

# Paths
MODEL_DIR = Path("models")
ONNX_DIR = MODEL_DIR / "onnx"

# Load metadata
with open(ONNX_DIR / "models_metadata.json", 'r') as f:
    metadata = json.load(f)

print(f"\n📊 Number of features: {metadata['n_features']}")
print(f"📁 Testing {len(metadata['models'])} models")

# Load scaler
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
scaler_onnx_session = rt.InferenceSession(str(ONNX_DIR / "scaler.onnx"))

# Create test data
np.random.seed(42)
n_samples = 100
n_features = metadata['n_features']
X_test = np.random.randn(n_samples, n_features).astype(np.float32)

print(f"\n🧪 Test data shape: {X_test.shape}")

# Test each model
results = []

for model_info in metadata['models']:
    print(f"\n{'='*80}")
    print(f"Testing: {model_info['name']}")
    print(f"{'='*80}")

    model_key = model_info['key']

    try:
        # Load original model
        original_model = joblib.load(MODEL_DIR / f"model_{model_key}.pkl" if 'xgboost' not in model_key else MODEL_DIR / "best_model_xgboost.pkl")

        # Load ONNX model
        onnx_model = rt.InferenceSession(str(ONNX_DIR / f"{model_key}.onnx"))

        # Scale data
        X_scaled = scaler.transform(X_test)

        # Original model predictions
        original_pred = original_model.predict(X_scaled)
        original_proba = original_model.predict_proba(X_scaled)

        # ONNX model predictions
        input_name = onnx_model.get_inputs()[0].name
        onnx_output = onnx_model.run(None, {input_name: X_scaled.astype(np.float32)})

        # Extract predictions (format may vary)
        if len(onnx_output) >= 2:
            onnx_pred = onnx_output[0].flatten()
            onnx_proba = onnx_output[1] if isinstance(onnx_output[1], np.ndarray) else None
        else:
            onnx_pred = onnx_output[0].flatten()
            onnx_proba = None

        # Compare predictions
        if len(onnx_pred) == len(original_pred):
            match_rate = np.mean(onnx_pred == original_pred) * 100
            print(f"✅ Prediction match rate: {match_rate:.2f}%")
        else:
            print(f"⚠️  Shape mismatch - Original: {original_pred.shape}, ONNX: {onnx_pred.shape}")
            match_rate = 0.0

        # Compare probabilities if available
        if onnx_proba is not None and onnx_proba.shape == original_proba.shape:
            prob_diff = np.abs(original_proba - onnx_proba).mean()
            print(f"✅ Average probability difference: {prob_diff:.6f}")
        else:
            prob_diff = None
            print(f"⚠️  Probability comparison not available")

        # Performance test
        import time

        # Original model timing
        start = time.time()
        for _ in range(100):
            _ = original_model.predict(X_scaled[:10])
        original_time = (time.time() - start) / 100

        # ONNX model timing
        start = time.time()
        for _ in range(100):
            _ = onnx_model.run(None, {input_name: X_scaled[:10].astype(np.float32)})
        onnx_time = (time.time() - start) / 100

        speedup = original_time / onnx_time if onnx_time > 0 else 1.0

        print(f"\n⏱️  Performance Comparison (10 samples, 100 iterations):")
        print(f"   Original model: {original_time*1000:.3f} ms")
        print(f"   ONNX model:     {onnx_time*1000:.3f} ms")
        print(f"   Speedup:        {speedup:.2f}x {'(ONNX faster)' if speedup > 1 else '(Original faster)'}")

        results.append({
            'model': model_info['name'],
            'match_rate': match_rate,
            'prob_diff': prob_diff,
            'original_time_ms': round(original_time * 1000, 3),
            'onnx_time_ms': round(onnx_time * 1000, 3),
            'speedup': round(speedup, 2)
        })

        print(f"\n✅ {model_info['name']} test PASSED")

    except Exception as e:
        print(f"❌ Error testing {model_info['name']}: {e}")
        import traceback
        traceback.print_exc()

# Summary
print(f"\n{'='*80}")
print(f"TEST SUMMARY")
print(f"{'='*80}")

for result in results:
    print(f"\n{result['model']}:")
    print(f"  Match Rate:  {result['match_rate']:.2f}%")
    if result['prob_diff'] is not None:
        print(f"  Prob Diff:   {result['prob_diff']:.6f}")
    print(f"  Original:    {result['original_time_ms']:.3f} ms")
    print(f"  ONNX:        {result['onnx_time_ms']:.3f} ms")
    print(f"  Speedup:     {result['speedup']:.2f}x")

print(f"\n{'='*80}")
print(f"✅ All tests complete!")
print(f"{'='*80}")
