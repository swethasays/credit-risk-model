# Credit Risk Prediction Model

End-to-end ML pipeline for predicting loan default probability using customer financial data.

## Tech stack
![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)

## Problem
A retail bank faces increasing loan defaults due to ineffective rule-based credit
assessment. This project builds a machine learning pipeline to predict default
probability, improve risk classification accuracy, and provide interpretable
explanations aligned with regulatory requirements.

## Architecture
EDA → Feature Engineering → XGBoost Model → SHAP Explainability
→ FastAPI Endpoint → LangChain Explanation Layer → Streamlit Dashboard

## Results
- ROC-AUC: **0.86** on held-out test set
- Reduced false approvals of high-risk customers by ~20%
- Real-time risk scoring via REST API with human-readable explanations

## Run locally
```bash
# Train model
python src/train.py

# Start API
uvicorn api.main:app --reload

# Launch dashboard
streamlit run app/dashboard.py
```

## Dataset
[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — 150k customer records, 11 financial features.