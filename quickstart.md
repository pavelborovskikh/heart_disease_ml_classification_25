# Quick Start Guide

Get your Heart Disease ML API running in 5 minutes!

## Prerequisites

- Python 3.10+ installed
- `uv` package manager (or pip)
- Git (optional)

## 1. Install Dependencies

### Using uv (Recommended - Fast!)
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Using pip (Alternative)
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn xgboost fastapi uvicorn pydantic joblib requests
```

## 2. Train the Model

```bash
# Using uv
uv run python train.py

# Using pip
python train.py
```

This creates `model.pkl` and `preprocessor.pkl` files.

## 3. Run the API

```bash
# Using uv
uv run uvicorn app:app --reload

# Using pip
uvicorn app:app --reload
```

The API will be available at: **http://localhost:8000**

## 4. Test the API

### Option 1: Interactive Docs
Open your browser: **http://localhost:8000/docs**

### Option 2: Test Script
```bash
# Local testing
uv run python test.py

# For deployed API (update URL in test.py first)
uv run python test.py
```

### Option 3: curl
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalch": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1,
    "dataset": "Cleveland"
  }'
```

## 5. Deploy to Fly.io (Optional)

```bash
# Install Fly CLI
# Windows: iwr https://fly.io/install.ps1 -useb | iex
# Mac/Linux: curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Deploy
fly deploy

# Open your deployed API
fly open /docs
```

Your API will be live at: `https://heart-disease-ml.fly.dev`

## Troubleshooting

### Port Already in Use
```bash
# Kill the process using port 8000
# Windows: netstat -ano | findstr :8000
# Linux/Mac: lsof -ti:8000 | xargs kill -9
```

### Model Files Missing
```bash
# Retrain the model
uv run python train.py
```

### Fly.io Timeout on First Request
```bash
# Wake up the machine first
curl https://heart-disease-ml.fly.dev/health

# Wait 10 seconds, then test
uv run python test.py
```

### Import Errors
```bash
# Reinstall dependencies
uv sync --force
```

## Next Steps

1. **Explore the API**: Visit `/docs` for interactive documentation
2. **Check Performance**: Review model metrics in training output
3. **Customize**: Modify `train.py` to experiment with different models
4. **Monitor**: Use `fly logs` to monitor your deployed API

## Quick Reference

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies |
| `uv run python train.py` | Train model |
| `uv run uvicorn app:app --reload` | Start API locally |
| `uv run python test.py` | Run tests |
| `fly deploy` | Deploy to Fly.io |
| `fly logs` | View deployment logs |
| `fly status` | Check deployment status |

## Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Review [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) for project requirements
- Visit the Swagger UI at `/docs` for API documentation

---

**Ready in 5 minutes!** 🚀