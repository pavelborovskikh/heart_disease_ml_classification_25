"""
ML Zoomcamp 25 Midterm: Heart Disease Classification Training Pipeline
Uses Random Forest with preprocessing pipeline for production deployment.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Feature definitions - CORRECTED: 'thalch' not 'thalach'
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex",
    "dataset",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

DATA_FILE = "heart_disease_uci.csv"
MODEL_OUTPUT_FILE = "model.pkl"
PREPROCESSOR_OUTPUT_FILE = "preprocessor.pkl"


def load_and_prepare_data(filepath: str) -> tuple:
    """Load data, create binary target, handle missing values."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    logger.info("Starting data preprocessing...")

    # Create binary target: 0 = no disease, 1 = presence of disease
    df["target"] = (df["num"] > 0).astype(int)
    logger.info(f"Target distribution:\n{df['target'].value_counts()}")

    # Handle duplicates
    initial_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Removed {initial_rows - len(df)} duplicate rows")

    # Handle missing values (convert "?" to NaN if present)
    df = df.replace("?", np.nan)

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


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> tuple:
    """Train Random Forest with hyperparameter tuning."""
    logger.info("Building Random Forest pipeline...")

    preprocessor = build_preprocessor()

    # Baseline Random Forest
    logger.info("Training baseline Random Forest...")
    rf_baseline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                    verbose=0,
                ),
            ),
        ]
    )

    rf_baseline.fit(X_train, y_train)

    y_pred_baseline = rf_baseline.predict(X_test)
    y_proba_baseline = rf_baseline.predict_proba(X_test)[:, 1]

    baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
    baseline_roc_auc = roc_auc_score(y_test, y_proba_baseline)

    logger.info(f"Baseline Random Forest - Accuracy: {baseline_accuracy:.4f}")
    logger.info(f"Baseline Random Forest - ROC AUC: {baseline_roc_auc:.4f}")
    logger.info(
        f"\nBaseline Classification Report:\n{classification_report(y_test, y_pred_baseline)}"
    )

    # Hyperparameter tuning
    logger.info("\nStarting hyperparameter tuning (RandomizedSearchCV)...")

    rf_param_dist = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 8, 10],
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
                    RandomForestClassifier(random_state=42, n_jobs=-1, verbose=0),
                ),
            ]
        ),
        param_distributions=rf_param_dist,
        n_iter=15,
        scoring="roc_auc",
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )

    rf_random_search.fit(X_train, y_train)

    logger.info(f"Best RF params: {rf_random_search.best_params_}")
    logger.info(f"Best RF CV ROC AUC: {rf_random_search.best_score_:.4f}")

    best_rf = rf_random_search.best_estimator_

    y_pred_tuned = best_rf.predict(X_test)
    y_proba_tuned = best_rf.predict_proba(X_test)[:, 1]

    tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
    tuned_roc_auc = roc_auc_score(y_test, y_proba_tuned)

    logger.info(f"\nTuned Random Forest - Accuracy: {tuned_accuracy:.4f}")
    logger.info(f"Tuned Random Forest - ROC AUC: {tuned_roc_auc:.4f}")
    logger.info(
        f"\nTuned Classification Report:\n{classification_report(y_test, y_pred_tuned)}"
    )
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_tuned)}")

    return best_rf, preprocessor


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
        logger.info("Starting ML Zoomcamp Heart Disease Classification Training")
        logger.info("=" * 60)

        # 1. Load and prepare data
        df = load_and_prepare_data(DATA_FILE)

        # 2. Prepare features and target
        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
        y = df["target"].values

        # 3. Train-test split (stratified for balanced classes)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(
            f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples"
        )

        # 4. Train Random Forest with tuning
        best_model, preprocessor = train_random_forest(
            X_train, y_train, X_test, y_test
        )

        # 5. Save model and preprocessor
        save_model_and_preprocessor(best_model, preprocessor)

        logger.info("=" * 60)
        logger.info("✅ Training pipeline completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
