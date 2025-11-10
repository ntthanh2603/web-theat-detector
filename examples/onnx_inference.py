"""
ONNX Model Inference Example
Demonstrates how to use ONNX models for phishing detection
"""

import numpy as np
import onnxruntime as rt
from pathlib import Path
import json
import sys

sys.path.append('..')
from api.feature_extractor import URLFeatureExtractor

class ONNXPhishingDetector:
    """Phishing detector using ONNX models"""

    def __init__(self, model_dir="models/onnx"):
        """
        Initialize ONNX phishing detector

        Args:
            model_dir: Directory containing ONNX models
        """
        self.model_dir = Path(model_dir)

        # Load metadata
        with open(self.model_dir / "models_metadata.json", 'r') as f:
            self.metadata = json.load(f)

        # Load feature extractor
        self.feature_extractor = URLFeatureExtractor()

        # Load scaler
        self.scaler_session = rt.InferenceSession(
            str(self.model_dir / "scaler.onnx")
        )
        self.scaler_input_name = self.scaler_session.get_inputs()[0].name

        # Load models
        self.models = {}
        for model_info in self.metadata['models']:
            model_key = model_info['key']
            session = rt.InferenceSession(
                str(self.model_dir / f"{model_key}.onnx")
            )
            self.models[model_key] = {
                'session': session,
                'input_name': session.get_inputs()[0].name,
                'name': model_info['name'],
                'accuracy': model_info['accuracy']
            }

        print(f"✅ Loaded {len(self.models)} ONNX models")

    def predict(self, url: str, model_name: str = "random_forest"):
        """
        Predict if URL is phishing

        Args:
            url: URL to analyze
            model_name: Model to use (random_forest, logistic_regression)
                       Note: XGBoost is not available in ONNX format

        Returns:
            dict: Prediction results
        """
        # Extract features
        features = self.feature_extractor.get_feature_vector(url)

        # Scale features
        features_scaled = self.scaler_session.run(
            None,
            {self.scaler_input_name: features.reshape(1, -1).astype(np.float32)}
        )[0]

        # Get model
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")

        model = self.models[model_name]

        # Predict
        output = model['session'].run(
            None,
            {model['input_name']: features_scaled.astype(np.float32)}
        )

        # Parse output
        prediction = int(output[0][0]) if hasattr(output[0], 'shape') and len(output[0].shape) == 1 else int(output[0][0][0])

        # Get probabilities if available
        if len(output) > 1:
            # Handle different output formats
            if isinstance(output[1], list):
                proba = output[1][0] if len(output[1]) > 0 else [0.0, 1.0]
            elif hasattr(output[1], 'shape') and len(output[1].shape) == 2:
                proba = output[1][0]
            else:
                proba = output[1]

            # Extract phishing probability
            if isinstance(proba, (list, np.ndarray)) and len(proba) > 1:
                phishing_prob = float(proba[1])
            else:
                phishing_prob = float(proba[0]) if hasattr(proba, '__getitem__') else 1.0
        else:
            phishing_prob = 1.0 if prediction == 1 else 0.0

        return {
            'url': url,
            'is_phishing': bool(prediction),
            'phishing_probability': phishing_prob,
            'confidence': phishing_prob if prediction == 1 else (1 - phishing_prob),
            'model_used': model['name'],
            'model_accuracy': model['accuracy']
        }

    def predict_ensemble(self, url: str):
        """
        Predict using ensemble of all models

        Args:
            url: URL to analyze

        Returns:
            dict: Ensemble prediction results
        """
        # Extract and scale features once
        features = self.feature_extractor.get_feature_vector(url)
        features_scaled = self.scaler_session.run(
            None,
            {self.scaler_input_name: features.reshape(1, -1).astype(np.float32)}
        )[0]

        # Get predictions from all models
        predictions = {}
        votes = []
        probabilities = []

        for model_key, model_info in self.models.items():
            output = model_info['session'].run(
                None,
                {model_info['input_name']: features_scaled.astype(np.float32)}
            )

            pred = int(output[0][0]) if hasattr(output[0], 'shape') and len(output[0].shape) == 1 else int(output[0][0][0])

            if len(output) > 1:
                # Handle different output formats
                if isinstance(output[1], list):
                    proba = output[1][0] if len(output[1]) > 0 else [0.0, 1.0]
                elif hasattr(output[1], 'shape') and len(output[1].shape) == 2:
                    proba = output[1][0]
                else:
                    proba = output[1]

                # Extract phishing probability
                if isinstance(proba, (list, np.ndarray)) and len(proba) > 1:
                    phishing_prob = float(proba[1])
                else:
                    phishing_prob = float(proba[0]) if hasattr(proba, '__getitem__') else 1.0
            else:
                phishing_prob = 1.0 if pred == 1 else 0.0

            predictions[model_info['name']] = {
                'prediction': pred,
                'phishing_probability': phishing_prob
            }

            votes.append(pred)
            probabilities.append(phishing_prob)

        # Weighted average
        weights = [model_info['accuracy'] for model_info in self.models.values()]
        weighted_prob = sum(p * w for p, w in zip(probabilities, weights)) / sum(weights)

        # Majority voting
        final_prediction = 1 if sum(votes) >= len(votes) / 2 else 0

        return {
            'url': url,
            'is_phishing': bool(final_prediction),
            'ensemble_probability': weighted_prob,
            'confidence': weighted_prob if final_prediction else (1 - weighted_prob),
            'individual_predictions': predictions,
            'votes': f"{sum(votes)}/{len(votes)} models predict phishing"
        }


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("ONNX PHISHING DETECTION - INFERENCE EXAMPLE")
    print("="*80)

    # Initialize detector
    detector = ONNXPhishingDetector("../models/onnx")

    # Test URLs
    test_urls = [
        "https://www.google.com",
        "http://secure-login-verify.suspicious-domain.com/account/update",
        "https://www.github.com",
        "http://paypal-verify-account.phishing-site.com/login"
    ]

    print("\n" + "="*80)
    print("SINGLE MODEL PREDICTIONS (Random Forest - 98.55% accuracy)")
    print("="*80)

    for url in test_urls:
        print(f"\n🔍 Analyzing: {url}")
        result = detector.predict(url, model_name="random_forest")

        status = "⚠️ PHISHING" if result['is_phishing'] else "✅ SAFE"
        print(f"   {status}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Phishing Probability: {result['phishing_probability']:.2%}")

    print("\n" + "="*80)
    print("ENSEMBLE PREDICTIONS (All Models)")
    print("="*80)

    for url in test_urls:
        print(f"\n🔍 Analyzing: {url}")
        result = detector.predict_ensemble(url)

        status = "⚠️ PHISHING" if result['is_phishing'] else "✅ SAFE"
        print(f"   {status}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Votes: {result['votes']}")
        print(f"   Individual predictions:")
        for model_name, pred in result['individual_predictions'].items():
            pred_text = "Phishing" if pred['prediction'] == 1 else "Safe"
            print(f"      - {model_name}: {pred_text} ({pred['phishing_probability']:.2%})")

    print("\n" + "="*80)
    print("✅ Inference examples complete!")
    print("="*80)
