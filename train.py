"""
ML Zoomcamp Capstone: House Price Prediction Training Pipeline
Uses Random Forest Regressor with preprocessing pipeline for production deployment.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Feature definitions
NUMERIC_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "sqft_above", "sqft_basement", "yr_built", "lat", "long",
    "sqft_living15", "sqft_lot15"
]
CATEGORICAL_FEATURES = [
    "waterfront", "view", "condition", "grade", "zipcode"
]

DATA_FILE = "kc_house_data.csv"
MODEL_OUTPUT_FILE = "model.pkl"
PREPROCESSOR_OUTPUT_FILE = "preprocessor.pkl"


def load_and_prepare_data(filepath: str) -> tuple:
    """Load data, handle missing values, create features."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    logger.info("Starting data preprocessing...")

    # Handle duplicates
    initial_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Removed {initial_rows - len(df)} duplicate rows")

    # Feature engineering
    df['house_age'] = 2015 - df['yr_built']
    df['renovated'] = (df['yr_renovated'] > 0).astype(int)
    df['price_per_sqft'] = df['price'] / df['sqft_living']
    
    logger.info(f"Price statistics:")
    logger.info(f"  Mean: ${df['price'].mean():,.2f}")
    logger.info(f"  Median: ${df['price'].median():,.2f}")
    logger.info(f"  Std: ${df['price'].std():,.2f}")
    logger.info(f"  Min: ${df['price'].min():,.2f}")
    logger.info(f"  Max: ${df['price'].max():,.2f}")

    logger.info(f"Numeric features: {NUMERIC_FEATURES}")
    logger.info(f"Categorical features: {CATEGORICAL_FEATURES}")

    return df


def build_preprocessor() -> ColumnTransformer:
    """Build sklearn ColumnTransformer for preprocessing."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    """Evaluate a trained model and return metrics."""
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"\n{name} Results:")
    logger.info(f"  RMSE: ${rmse:,.2f}")
    logger.info(f"  MAE: ${mae:,.2f}")
    logger.info(f"  R² Score: {r2:.4f}")
    
    return {
        "name": name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> tuple:
    """Train Linear Regression baseline model."""
    logger.info("Training Linear Regression baseline...")
    
    preprocessor = build_preprocessor()
    
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    
    model.fit(X_train, y_train)
    metrics = evaluate_model("Linear Regression", model, X_test, y_test)
    
    return model, metrics


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> tuple:
    """Train Random Forest with hyperparameter tuning."""
    logger.info("\nBuilding Random Forest pipeline...")

    preprocessor = build_preprocessor()

    # Baseline Random Forest
    logger.info("Training baseline Random Forest...")
    rf_baseline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    verbose=0,
                ),
            ),
        ]
    )

    rf_baseline.fit(X_train, y_train)
    baseline_metrics = evaluate_model("Random Forest (Baseline)", rf_baseline, X_test, y_test)

    # Hyperparameter tuning
    logger.info("\nStarting hyperparameter tuning (RandomizedSearchCV)...")

    rf_param_dist = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    }

    rf_random_search = RandomizedSearchCV(
        estimator=Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestRegressor(random_state=42, n_jobs=-1, verbose=0),
                ),
            ]
        ),
        param_distributions=rf_param_dist,
        n_iter=15,
        scoring="neg_root_mean_squared_error",
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )

    rf_random_search.fit(X_train, y_train)

    logger.info(f"Best RF params: {rf_random_search.best_params_}")
    logger.info(f"Best RF CV RMSE: ${-rf_random_search.best_score_:,.2f}")

    best_rf = rf_random_search.best_estimator_
    tuned_metrics = evaluate_model("Random Forest (Tuned)", best_rf, X_test, y_test)

    return best_rf, preprocessor, tuned_metrics


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> tuple:
    """Train XGBoost Regressor."""
    logger.info("\nTraining XGBoost Regressor...")
    
    preprocessor = build_preprocessor()
    
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=7,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                ),
            ),
        ]
    )
    
    model.fit(X_train, y_train)
    metrics = evaluate_model("XGBoost", model, X_test, y_test)
    
    return model, metrics


def save_model_and_preprocessor(
    model: Pipeline, preprocessor: ColumnTransformer
) -> None:
    """Save trained model and preprocessor to disk."""
    logger.info(f"\nSaving model to {MODEL_OUTPUT_FILE}...")
    with open(MODEL_OUTPUT_FILE, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Saving preprocessor to {PREPROCESSOR_OUTPUT_FILE}...")
    with open(PREPROCESSOR_OUTPUT_FILE, "wb") as f:
        pickle.dump(preprocessor, f)

    logger.info("✅ Model and preprocessor saved successfully!")


def main():
    """Main training pipeline."""
    try:
        logger.info("=" * 60)
        logger.info("Starting ML Zoomcamp House Price Prediction Training")
        logger.info("=" * 60)

        # 1. Load and prepare data
        df = load_and_prepare_data(DATA_FILE)

        # 2. Prepare features and target
        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
        y = df["price"].values

        # 3. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info(
            f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples"
        )

        # 4. Train multiple models
        all_metrics = []
        
        # Linear Regression
        lr_model, lr_metrics = train_linear_regression(X_train, y_train, X_test, y_test)
        all_metrics.append(lr_metrics)
        
        # Random Forest with tuning
        rf_model, preprocessor, rf_metrics = train_random_forest(
            X_train, y_train, X_test, y_test
        )
        all_metrics.append(rf_metrics)
        
        # XGBoost
        xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
        all_metrics.append(xgb_metrics)

        # 5. Select best model based on R² score
        logger.info("\n" + "=" * 60)
        logger.info("Model Comparison Summary")
        logger.info("=" * 60)
        for metrics in all_metrics:
            logger.info(
                f"{metrics['name']}: "
                f"RMSE=${metrics['rmse']:,.2f}, "
                f"MAE=${metrics['mae']:,.2f}, "
                f"R²={metrics['r2']:.4f}"
            )
        
        best_model_metrics = max(all_metrics, key=lambda x: x['r2'])
        logger.info(f"\n✅ Best Model: {best_model_metrics['name']} (R²={best_model_metrics['r2']:.4f})")
        
        # Select the actual best model
        if best_model_metrics['name'] == 'Linear Regression':
            best_model = lr_model
        elif best_model_metrics['name'] == 'XGBoost':
            best_model = xgb_model
        else:  # Random Forest (Tuned)
            best_model = rf_model

        # 6. Save model and preprocessor
        save_model_and_preprocessor(best_model, preprocessor)

        logger.info("=" * 60)
        logger.info("✅ Training pipeline completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
