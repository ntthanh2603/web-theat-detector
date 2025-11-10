# ONNX Models Usage Guide

Complete guide for using the exported ONNX models across different platforms and programming languages.

## What is ONNX?

ONNX (Open Neural Network Exchange) is an open format for representing machine learning models. Benefits:

- ✅ **Cross-platform**: Use models in Python, JavaScript, C#, Java, C++, etc.
- ✅ **High Performance**: Optimized inference engines
- ✅ **Edge Deployment**: Run on mobile, web browsers, IoT devices
- ✅ **Framework Independent**: No need for scikit-learn or XGBoost at runtime

## Exported Models

The following models are available in ONNX format:

| Model | File | Accuracy | Size | Status |
|-------|------|----------|------|--------|
| **Random Forest** | `models/onnx/random_forest.onnx` | **98.55%** | 2.86 MB | ✅ Exported |
| Logistic Regression | `models/onnx/logistic_regression.onnx` | 95.20% | <1 MB | ✅ Exported |
| StandardScaler | `models/onnx/scaler.onnx` | - | 0.67 KB | ✅ Exported |
| XGBoost | `models/best_model_xgboost.pkl` | 98.65% | 218 KB | ⚠️ Python only (PKL) |

**Note:** XGBoost ONNX export has compatibility issues with current library versions. Use the **Random Forest ONNX model** (98.55% accuracy) for cross-platform deployment, or the pickled XGBoost model for Python-only deployments.

## Quick Start

### 1. Export Models to ONNX

```bash
# Install dependencies
pip install skl2onnx onnx onnxruntime onnxmltools

# Export models
python export_to_onnx.py

# Test exported models
python test_onnx_models.py
```

### 2. Python Inference

```python
import onnxruntime as rt
import numpy as np

# Load Random Forest model (98.55% accuracy)
session = rt.InferenceSession("models/onnx/random_forest.onnx")
input_name = session.get_inputs()[0].name

# Prepare input (48 features)
features = np.array([[...]], dtype=np.float32)  # Shape: (1, 48)

# Run inference
output = session.run(None, {input_name: features})
prediction = output[0][0]  # 0 = legitimate, 1 = phishing
```

See `examples/onnx_inference.py` for complete example.

---

## Platform-Specific Usage

### Python (onnxruntime)

**Installation:**
```bash
pip install onnxruntime
```

**Usage:**
```python
import onnxruntime as rt
import numpy as np

# Load model
session = rt.InferenceSession("models/onnx/xgboost.onnx")

# Get input/output names
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

# Inference
features = np.random.randn(1, 48).astype(np.float32)
result = session.run(output_names, {input_name: features})

print(f"Prediction: {result[0]}")
print(f"Probabilities: {result[1]}")
```

---

### JavaScript/TypeScript (ONNX Runtime Web)

**Installation:**
```bash
npm install onnxruntime-web
```

**Usage:**
```javascript
import * as ort from 'onnxruntime-web';

// Load Random Forest model (98.55% accuracy)
const session = await ort.InferenceSession.create('./models/onnx/random_forest.onnx');

// Prepare input
const features = new Float32Array(48);  // Your 48 features
const tensor = new ort.Tensor('float32', features, [1, 48]);

// Run inference
const feeds = { float_input: tensor };
const results = await session.run(feeds);

const prediction = results.label.data[0];
const probabilities = results.probabilities.data;

console.log(`Phishing: ${prediction === 1}`);
console.log(`Confidence: ${probabilities[1]}`);
```

**Browser Example:**
```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
</head>
<body>
    <button onclick="detectPhishing()">Check URL</button>
    <script>
        async function detectPhishing() {
            // Load Random Forest model (98.55% accuracy)
            const session = await ort.InferenceSession.create('random_forest.onnx');

            // Extract features from URL (48 features)
            const features = new Float32Array(48);
            // ... feature extraction logic ...

            const tensor = new ort.Tensor('float32', features, [1, 48]);
            const results = await session.run({ float_input: tensor });

            const isPhishing = results.label.data[0] === 1;
            alert(isPhishing ? 'Phishing!' : 'Safe');
        }
    </script>
</body>
</html>
```

---

### C# (.NET)

**Installation:**
```bash
dotnet add package Microsoft.ML.OnnxRuntime
```

**Usage:**
```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// Load model
using var session = new InferenceSession("models/onnx/xgboost.onnx");

// Prepare input
var features = new float[48];  // Your 48 features
var tensor = new DenseTensor<float>(features, new[] { 1, 48 });

// Create input
var inputs = new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor("float_input", tensor)
};

// Run inference
using var results = session.Run(inputs);
var prediction = results.First(v => v.Name == "label").AsEnumerable<long>().First();
var probabilities = results.First(v => v.Name == "probabilities").AsEnumerable<float>().ToArray();

Console.WriteLine($"Phishing: {prediction == 1}");
Console.WriteLine($"Confidence: {probabilities[1]}");
```

---

### Java

**Maven dependency:**
```xml
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime</artifactId>
    <version>1.16.0</version>
</dependency>
```

**Usage:**
```java
import ai.onnxruntime.*;

// Load model
OrtEnvironment env = OrtEnvironment.getEnvironment();
OrtSession session = env.createSession("models/onnx/xgboost.onnx");

// Prepare input
float[] features = new float[48];  // Your 48 features
long[] shape = new long[]{1, 48};
OnnxTensor tensor = OnnxTensor.createTensor(env, features, shape);

// Run inference
Map<String, OnnxTensor> inputs = Map.of("float_input", tensor);
OrtSession.Result result = session.run(inputs);

// Get output
long[] predictions = (long[]) result.get(0).getValue();
float[][] probabilities = (float[][]) result.get(1).getValue();

System.out.println("Phishing: " + (predictions[0] == 1));
System.out.println("Confidence: " + probabilities[0][1]);
```

---

### C++ (Native)

**CMake:**
```cmake
find_package(onnxruntime REQUIRED)
target_link_libraries(your_app onnxruntime)
```

**Usage:**
```cpp
#include <onnxruntime_cxx_api.h>
#include <vector>

// Initialize
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "phishing_detector");
Ort::SessionOptions session_options;
Ort::Session session(env, "models/onnx/xgboost.onnx", session_options);

// Prepare input
std::vector<float> features(48);  // Your 48 features
std::vector<int64_t> input_shape = {1, 48};

auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info, features.data(), features.size(),
    input_shape.data(), input_shape.size()
);

// Run inference
const char* input_names[] = {"float_input"};
const char* output_names[] = {"label", "probabilities"};

auto output_tensors = session.Run(
    Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
    output_names, 2
);

// Get results
int64_t* prediction = output_tensors[0].GetTensorMutableData<int64_t>();
float* probabilities = output_tensors[1].GetTensorMutableData<float>();

std::cout << "Phishing: " << (prediction[0] == 1) << std::endl;
std::cout << "Confidence: " << probabilities[1] << std::endl;
```

---

### React Native (Mobile)

**Installation:**
```bash
npm install onnxruntime-react-native
```

**Usage:**
```javascript
import { InferenceSession, Tensor } from 'onnxruntime-react-native';

const detectPhishing = async (url) => {
    // Load model
    const session = await InferenceSession.create(
        require('./models/xgboost.onnx')
    );

    // Extract features
    const features = extractFeatures(url);  // 48 features
    const tensor = new Tensor('float32', new Float32Array(features), [1, 48]);

    // Run inference
    const feeds = { float_input: tensor };
    const results = await session.run(feeds);

    const isPhishing = results.label.data[0] === 1;
    const confidence = results.probabilities.data[1];

    return { isPhishing, confidence };
};
```

---

### Python FastAPI Integration

Update your API to use ONNX models:

```python
import onnxruntime as rt
from fastapi import FastAPI

app = FastAPI()

# Load ONNX models at startup
xgboost_session = rt.InferenceSession("models/onnx/xgboost.onnx")
scaler_session = rt.InferenceSession("models/onnx/scaler.onnx")

@app.post("/predict")
async def predict(url: str):
    # Extract features
    features = extract_features(url)  # Returns (1, 48) array

    # Scale features
    features_scaled = scaler_session.run(
        None,
        {"float_input": features.astype(np.float32)}
    )[0]

    # Predict
    output = xgboost_session.run(
        None,
        {"float_input": features_scaled.astype(np.float32)}
    )

    prediction = int(output[0][0])
    probability = float(output[1][0][1])

    return {
        "is_phishing": bool(prediction),
        "confidence": probability
    }
```

---

## Feature Extraction

All models expect 48 features in this order:

```python
feature_names = [
    'NumDots', 'SubdomainLevel', 'PathLevel', 'UrlLength', 'NumDash',
    'NumDashInHostname', 'AtSymbol', 'TildeSymbol', 'NumUnderscore',
    'NumPercent', 'NumQueryComponents', 'NumAmpersand', 'NumHash',
    'NumNumericChars', 'NoHttps', 'RandomString', 'IpAddress',
    'DomainInSubdomains', 'DomainInPaths', 'HttpsInHostname',
    'HostnameLength', 'PathLength', 'QueryLength', 'DoubleSlashInPath',
    'NumSensitiveWords', 'EmbeddedBrandName', 'PctExtHyperlinks',
    'PctExtResourceUrls', 'ExtFavicon', 'InsecureForms',
    'RelativeFormAction', 'ExtFormAction', 'AbnormalFormAction',
    'PctNullSelfRedirectHyperlinks', 'FrequentDomainNameMismatch',
    'FakeLinkInStatusBar', 'RightClickDisabled', 'PopUpWindow',
    'SubmitInfoToEmail', 'IframeOrFrame', 'MissingTitle',
    'ImagesOnlyInForm', 'SubdomainLevelRT', 'UrlLengthRT',
    'PctExtResourceUrlsRT', 'AbnormalExtFormActionR',
    'ExtMetaScriptLinkRT', 'PctExtNullSelfRedirectHyperlinksRT'
]
```

See `api/feature_extractor.py` for complete feature extraction logic.

---

## Performance Tips

1. **Model Loading**: Load models once at startup, not per request
2. **Batching**: Process multiple URLs in one inference call
3. **GPU Acceleration**: Use CUDA execution provider for large batches
4. **Quantization**: Convert to INT8 for mobile/edge devices
5. **Model Size**: Use XGBoost for best accuracy/size tradeoff

### GPU Acceleration (Python)

```bash
pip install onnxruntime-gpu
```

```python
import onnxruntime as rt

# Use CUDA execution provider
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = rt.InferenceSession("model.onnx", providers=providers)
```

### Batch Inference

```python
# Instead of 1 URL at a time
features_batch = np.array([
    [url1_features],
    [url2_features],
    [url3_features]
]).astype(np.float32)  # Shape: (3, 48)

output = session.run(None, {input_name: features_batch})
```

---

## Troubleshooting

### Model Loading Issues

```python
import onnx

# Check model validity
model = onnx.load("model.onnx")
onnx.checker.check_model(model)

# View model info
print(model.graph.input)
print(model.graph.output)
```

### Input Shape Mismatch

```python
# Check expected input shape
for input in session.get_inputs():
    print(f"{input.name}: {input.shape}")

# Ensure your input matches
features = features.reshape(1, 48).astype(np.float32)
```

### Performance Issues

```python
# Enable profiling
session_options = rt.SessionOptions()
session_options.enable_profiling = True
session = rt.InferenceSession("model.onnx", session_options)
```

---

## Additional Resources

- **ONNX Documentation**: https://onnx.ai/
- **ONNX Runtime**: https://onnxruntime.ai/
- **Tutorials**: https://onnxruntime.ai/docs/get-started/
- **Model Zoo**: https://github.com/onnx/models

---

## Support

For issues specific to this phishing detection system:
- Check `examples/onnx_inference.py` for complete working example
- Review `test_onnx_models.py` for testing
- See `api/feature_extractor.py` for feature extraction

For ONNX Runtime issues:
- GitHub: https://github.com/microsoft/onnxruntime
- Issues: https://github.com/microsoft/onnxruntime/issues
