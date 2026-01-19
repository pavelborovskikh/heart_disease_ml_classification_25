# 🚀 Quick Start Guide - House Price Prediction API

Get the House Price Prediction API up and running in 5 minutes!

## Prerequisites

- Python 3.11 or higher
- pip or uv package manager
- Git (optional, for cloning)

## 5-Step Setup

### Step 1: Clone or Download

```bash
git clone https://github.com/pavelborovskikh/heart_disease_ml_classification_25.git
cd heart_disease_ml_classification_25
```

### Step 2: Install Dependencies

**Option A: Using pip**
```bash
pip install pandas numpy scikit-learn xgboost fastapi uvicorn pydantic joblib
```

**Option B: Using uv (faster)**
```bash
pip install uv
uv pip install --system pandas numpy scikit-learn xgboost fastapi uvicorn pydantic joblib
```

### Step 3: Train the Model

```bash
python train.py
```

Expected output:
```
============================================================
Starting ML Zoomcamp House Price Prediction Training
============================================================
...
✅ Training pipeline completed successfully!
```

This creates:
- `model.pkl` - Trained Random Forest model
- `preprocessor.pkl` - Fitted preprocessing pipeline

### Step 4: Start the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 5: Test the API

Open a new terminal and run:

```bash
python test.py
```

Or test with cURL:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 3,
    "bathrooms": 2.0,
    "sqft_living": 2000,
    "sqft_lot": 5000,
    "floors": 1.0,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "grade": 7,
    "sqft_above": 1500,
    "sqft_basement": 500,
    "yr_built": 1990,
    "zipcode": 98001,
    "lat": 47.3073,
    "long": -122.2108,
    "sqft_living15": 1900,
    "sqft_lot15": 4800
  }'
```

## 🐳 Docker Quick Start (Alternative)

If you prefer Docker:

```bash
# Build image
docker build -t house-price-api .

# Run container
docker run -p 8000:8000 house-price-api
```

Done! The API is at `http://localhost:8000`

## 📚 Next Steps

- Visit http://localhost:8000/docs for interactive API documentation
- Check out `notebook.ipynb` for exploratory data analysis
- Run `pytest test_api.py -v` for unit tests
- Read `README.md` for detailed documentation

## ❓ Troubleshooting

### Port 8000 already in use
```bash
# Use a different port
uvicorn app:app --host 0.0.0.0 --port 8080
```

### ModuleNotFoundError
```bash
# Make sure all dependencies are installed
pip install pandas numpy scikit-learn xgboost fastapi uvicorn pydantic joblib
```

### Model file not found
```bash
# Train the model first
python train.py
```

## �� Quick Commands Reference

```bash
# Train model
python train.py

# Start API server
uvicorn app:app --reload

# Run tests
python test.py
python test_api.py
pytest test_api.py -v

# Docker build & run
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api

# Check API health
curl http://localhost:8000/health
```

---

**Need help?** Check the full [README.md](README.md) or open an issue on GitHub.
