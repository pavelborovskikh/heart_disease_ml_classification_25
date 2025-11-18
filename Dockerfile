FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN uv pip install --system --no-cache pandas numpy scikit-learn xgboost fastapi uvicorn pydantic joblib

# Copy application files
COPY train.py predict.py app.py ./
COPY heart_disease_uci.csv ./

# Train model during build (bake into image)
RUN python train.py

# Expose port
EXPOSE 8000

# Run the application (NOT train.py)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]