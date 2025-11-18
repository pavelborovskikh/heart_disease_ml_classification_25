"""
Prediction service for Heart Disease Classification.
Handles single and batch predictions with proper preprocessing.
"""

import logging
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Configure logging
logger = logging.getLogger(__name__)

# Feature definitions
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "dataset", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_PATH = "model.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"


class PredictionService:
    """
    Service for loading trained model and making predictions.
    Handles both single and batch predictions.
    """

    def __init__(self):
        """Initialize the prediction service by loading model and preprocessor."""
        self.model = None
        self.preprocessor = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model and preprocessor from pickle files."""
        try:
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from {MODEL_PATH}")
            
            # Try to load preprocessor
            try:
                with open(PREPROCESSOR_PATH, "rb") as f:
                    self.preprocessor = pickle.load(f)
                logger.info(f"Preprocessor loaded from {PREPROCESSOR_PATH}")
            except FileNotFoundError:
                logger.warning(f"Preprocessor file not found: {PREPROCESSOR_PATH}")
                self.preprocessor = None
                
        except FileNotFoundError:
            logger.error(f"Model file not found: {MODEL_PATH}")
            raise

    def _validate_input(self, data: Dict[str, Any]) -> None:
        """Validate that input contains all required features."""
        missing_features = set(ALL_FEATURES) - set(data.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

    def _prepare_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare a pandas DataFrame from input data.
        
        Args:
            data: Dictionary containing feature values
            
        Returns:
            DataFrame with features in correct order
        """
        # Validate input
        self._validate_input(data)
        
        # Create DataFrame with features in correct order
        df = pd.DataFrame([data])
        
        # Select only the required features in correct order
        df = df[ALL_FEATURES]
        
        return df

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a single prediction.
        
        Args:
            data: Dictionary with features for one sample
                 Expected keys: age, sex, cp, trestbps, chol, fbs, restecg,
                               thalch, exang, oldpeak, slope, ca, thal, dataset
        
        Returns:
            Dictionary with prediction and probability
        """
        try:
            # Prepare data
            df = self._prepare_dataframe(data)
            
            # Use model's built-in preprocessing pipeline
            prediction = self.model.predict(df)[0]
            probability = self.model.predict_proba(df)[0][1]
            
            return {
                "prediction": int(prediction),
                "probability": float(probability),
                "heart_disease": bool(prediction),
                "confidence": float(max(self.model.predict_proba(df)[0]))
            }
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def predict_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple samples.
        
        Args:
            data_list: List of dictionaries, each containing features for one sample
        
        Returns:
            List of prediction dictionaries
        """
        try:
            # Prepare data
            dfs = [self._prepare_dataframe(data) for data in data_list]
            df_batch = pd.concat(dfs, ignore_index=True)
            
            # Make predictions
            predictions = self.model.predict(df_batch)
            probabilities = self.model.predict_proba(df_batch)[:, 1]
            confidences = np.max(self.model.predict_proba(df_batch), axis=1)
            
            results = []
            for i, (pred, prob, conf) in enumerate(zip(predictions, probabilities, confidences)):
                results.append({
                    "sample": i,
                    "prediction": int(pred),
                    "probability": float(prob),
                    "heart_disease": bool(pred),
                    "confidence": float(conf)
                })
            
            return results
        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            raise

    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Get the names of features the model expects.
        
        Returns:
            Dictionary with numeric and categorical feature names
        """
        return {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "all_features": ALL_FEATURES
        }
