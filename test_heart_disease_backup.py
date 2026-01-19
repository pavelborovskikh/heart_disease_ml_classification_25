#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for Heart Disease Prediction API
ML Zoomcamp Midterm Project
"""

import requests

# Configuration
# Change this URL to your deployed Fly.io URL after deployment
# url = "http://localhost:8000/predict"
# For Fly.io deployment, use:
url = "https://heart-disease-ml.fly.dev/predict"

# Sample patient data - High risk patient
patient_high_risk = {
    "age": 63,
    "sex": 1,  # Male
    "cp": 3,  # Chest pain type
    "trestbps": 145,  # Resting blood pressure
    "chol": 233,  # Cholesterol
    "fbs": 1,  # Fasting blood sugar > 120 mg/dl
    "restecg": 0,
    "thalch": 150,  # Maximum heart rate achieved
    "exang": 0,  # Exercise induced angina
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,  # Number of major vessels
    "thal": 1,
    "dataset": "Cleveland"
}

# Sample patient data - Low risk patient
patient_low_risk = {
    "age": 45,
    "sex": 0,  # Female
    "cp": 1,
    "trestbps": 120,
    "chol": 210,
    "fbs": 0,
    "restecg": 1,
    "thalch": 165,
    "exang": 0,
    "oldpeak": 0.5,
    "slope": 1,
    "ca": 0,
    "thal": 2,
    "dataset": "Cleveland"
}

def test_prediction(patient_data, patient_name):
    """Test the prediction endpoint with patient data."""
    print(f"\n{'='*60}")
    print(f"Testing: {patient_name}")
    print(f"{'='*60}")
    
    try:
        response = requests.post(url, json=patient_data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            print(f"\nResults:")
            print(f"  - Prediction: {result['prediction']}")
            print(f"  - Heart Disease: {result['heart_disease']}")
            print(f"  - Probability: {result['probability']:.2%}")
            print(f"  - Confidence: {result['confidence']:.2%}")
            
            if result['heart_disease']:
                print(f"\n⚠️  WARNING: Patient shows signs of heart disease")
            else:
                print(f"\n✅ Patient appears healthy")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Could not connect to {url}")
        print("   Make sure the API is running!")
    except requests.exceptions.Timeout:
        print(f"❌ Timeout: Request took too long")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Heart Disease Prediction API - Test Script")
    print("ML Zoomcamp Midterm Project")
    print("="*60)
    print(f"\nTesting endpoint: {url}")
    
    # Test with high-risk patient
    test_prediction(patient_high_risk, "High Risk Patient (63-year-old male)")
    
    # Test with low-risk patient
    test_prediction(patient_low_risk, "Low Risk Patient (45-year-old female)")
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print("="*60 + "\n")
