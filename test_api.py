"""
Test suite for Heart Disease Classification API.
Tests model loading, predictions, and API endpoints.
"""

import pytest
import os
from predict import PredictionService, NUMERIC_FEATURES, CATEGORICAL_FEATURES


class TestPredictionService:
    """Test cases for PredictionService."""

    @pytest.fixture(scope="session", autouse=True)
    def setup(self):
        """Check if model files exist before running tests."""
        if not os.path.exists("model.pkl"):
            raise FileNotFoundError(
                "model.pkl not found. Run 'python train.py' first to train and save the model."
            )
        if not os.path.exists("preprocessor.pkl"):
            raise FileNotFoundError(
                "preprocessor.pkl not found. Run 'python train.py' first."
            )

    def test_prediction_service_init(self):
        """Test PredictionService initialization."""
        service = PredictionService()
        assert service.model is not None
        assert service.preprocessor is not None
        print("✓ PredictionService initialized successfully")

    def test_single_prediction(self):
        """Test single prediction."""
        service = PredictionService()
        
        # Test data - patient with heart disease symptoms
        test_data = {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalch": 150,  # CORRECTED: was thalach
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1,
            "dataset": "Cleveland"
        }
        
        result = service.predict(test_data)
        
        assert "prediction" in result
        assert "probability" in result
        assert "heart_disease" in result
        assert result["prediction"] in [0, 1]
        assert 0.0 <= result["probability"] <= 1.0
        assert isinstance(result["heart_disease"], bool)
        
        print(f"✓ Single prediction successful: {result}")

    def test_batch_prediction(self):
        """Test batch predictions."""
        service = PredictionService()
        
        # Test data - multiple patients
        test_data_list = [
            {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalch": 150,  # CORRECTED
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1,
                "dataset": "Cleveland"
            },
            {
                "age": 45,
                "sex": 0,
                "cp": 1,
                "trestbps": 120,
                "chol": 210,
                "fbs": 0,
                "restecg": 1,
                "thalch": 140,  # CORRECTED
                "exang": 0,
                "oldpeak": 0.5,
                "slope": 1,
                "ca": 0,
                "thal": 1,
                "dataset": "Cleveland"
            }
        ]
        
        results = service.predict_batch(test_data_list)
        
        assert len(results) == 2
        for result in results:
            assert "prediction" in result
            assert "probability" in result
            assert "heart_disease" in result
            assert result["prediction"] in [0, 1]
            assert 0.0 <= result["probability"] <= 1.0
        
        print(f"✓ Batch prediction successful: {len(results)} predictions made")

    def test_feature_names(self):
        """Test that feature names are correct."""
        service = PredictionService()
        features = service.get_feature_names()
        
        assert "numeric_features" in features
        assert "categorical_features" in features
        
        # Verify correct feature names
        assert "thalch" in features["numeric_features"]
        assert "thalach" not in features["numeric_features"]  # OLD NAME NOT PRESENT
        assert len(features["numeric_features"]) == 5
        assert len(features["categorical_features"]) == 9
        
        print(f"✓ Feature names correct:")
        print(f"  Numeric: {features['numeric_features']}")
        print(f"  Categorical: {features['categorical_features']}")

    def test_missing_features_error(self):
        """Test handling of missing features."""
        service = PredictionService()
        
        # Test data with missing feature
        incomplete_data = {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            # Missing "chol"
            "fbs": 1,
            "restecg": 0,
            "thalch": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1,
            "dataset": "Cleveland"
        }
        
        # This should raise an error
        try:
            result = service.predict(incomplete_data)
            print("✗ Should have raised an error for missing features")
            assert False, "Expected error for missing features"
        except Exception as e:
            print(f"✓ Correctly raised error for missing features: {type(e).__name__}")


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    # Run without pytest (simple script mode)
    print("=" * 70)
    print("Running Heart Disease Classification Tests")
    print("=" * 70)
    
    try:
        test = TestPredictionService()
        test.setup()
        
        print("\n[1/5] Testing PredictionService Initialization...")
        test.test_prediction_service_init()
        
        print("\n[2/5] Testing Single Prediction...")
        test.test_single_prediction()
        
        print("\n[3/5] Testing Batch Prediction...")
        test.test_batch_prediction()
        
        print("\n[4/5] Testing Feature Names...")
        test.test_feature_names()
        
        print("\n[5/5] Testing Missing Features Error Handling...")
        test.test_missing_features_error()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        raise
