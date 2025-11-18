"""
FastAPI application for Heart Disease Classification ML model.
Handles predictions via REST API endpoints.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from predict import PredictionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global prediction service instance
service: PredictionService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle - startup and shutdown.
    """
    global service
    # Startup
    logger.info("FastAPI application starting up...")
    service = PredictionService()
    logger.info(f"Numeric features: {service.get_feature_names()['numeric_features']}")
    logger.info(f"Categorical features: {service.get_feature_names()['categorical_features']}")
    
    yield
    
    # Shutdown
    logger.info("FastAPI application shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Classification API",
    description="ML API for predicting heart disease presence using Random Forest",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# Pydantic Models
# ============================================================================

class HeartPredictionRequest(BaseModel):
    """
    Request schema for heart disease prediction.
    All fields are required.
    """
    model_config = ConfigDict(protected_namespaces=())
    
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalch: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int
    dataset: str


class PredictionResponse(BaseModel):
    """Response schema for prediction results."""
    model_config = ConfigDict(protected_namespaces=())
    
    prediction: int
    probability: float
    heart_disease: bool
    confidence: float


class HealthResponse(BaseModel):
    """Response schema for health check."""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    model_loaded: bool
    model_type: str


class FeaturesResponse(BaseModel):
    """Response schema for feature information."""
    model_config = ConfigDict(protected_namespaces=())
    
    numeric_features: List[str]
    categorical_features: List[str]
    all_features: List[str]


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - welcome message."""
    return {
        "message": "Heart Disease Classification API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Verifies that the model is loaded and ready.
    """
    return {
        "status": "healthy" if service.model is not None else "unhealthy",
        "model_loaded": service.model is not None,
        "model_type": "RandomForestClassifier with preprocessing pipeline"
    }


@app.get("/features", response_model=FeaturesResponse, tags=["Info"])
async def get_features():
    """
    Get expected feature names for predictions.
    """
    features = service.get_feature_names()
    return {
        "numeric_features": features["numeric_features"],
        "categorical_features": features["categorical_features"],
        "all_features": features["all_features"]
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: HeartPredictionRequest):
    """
    Make a single prediction for heart disease presence.
    
    **Parameters:**
    - **age**: Patient age (int)
    - **sex**: Sex (0=female, 1=male)
    - **cp**: Chest pain type (0-3)
    - **trestbps**: Resting blood pressure (mm Hg)
    - **chol**: Serum cholesterol (mg/dl)
    - **fbs**: Fasting blood sugar > 120 mg/dl (0/1)
    - **restecg**: Resting electrocardiographic results (0-2)
    - **thalch**: Maximum heart rate achieved
    - **exang**: Exercise induced angina (0/1)
    - **oldpeak**: ST depression induced by exercise
    - **slope**: Slope of ST segment (0-2)
    - **ca**: Number of major vessels (0-4)
    - **thal**: Thalassemia (0-3)
    - **dataset**: Dataset source (string)
    
    **Returns:**
    - **prediction**: Binary prediction (0=no disease, 1=disease present)
    - **probability**: Probability of disease (0-1)
    - **heart_disease**: Boolean interpretation of prediction
    - **confidence**: Model confidence score
    """
    try:
        result = service.predict(request.model_dump())
        return result
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Predictions"])
async def predict_batch(requests: List[HeartPredictionRequest]):
    """
    Make predictions for multiple patients at once.
    
    **Parameters:**
    - List of HeartPredictionRequest objects
    
    **Returns:**
    - List of PredictionResponse objects
    """
    try:
        data_list = [req.model_dump() for req in requests]
        results = service.predict_batch(data_list)
        return results
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/info", tags=["Info"])
async def model_info():
    """
    Get detailed information about the model and API.
    """
    return {
        "model_type": "Random Forest Classifier",
        "framework": "scikit-learn",
        "preprocessing": "StandardScaler + OneHotEncoder via ColumnTransformer",
        "features_total": 14,
        "numeric_features": 5,
        "categorical_features": 9,
        "target": "Heart disease presence (binary classification)",
        "endpoints": {
            "health": "GET /health",
            "features": "GET /features",
            "predict": "POST /predict",
            "batch_predict": "POST /predict/batch",
            "model_info": "GET /info",
            "docs": "GET /docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
