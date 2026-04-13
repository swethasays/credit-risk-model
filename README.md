# 🏦 Credit Risk Prediction System

> End-to-end ML pipeline that predicts loan default probability in real time — with explainable AI and LLM-generated analyst reports.

### 🔴 [Live Demo → credit-risk-swetha.streamlit.app](https://credit-risk-swetha.streamlit.app)

---

## What this does

A loan officer enters a customer's financial profile. The system instantly returns:

- **Default probability** with a risk tier — Low / Medium / High
- **Top 3 risk drivers** powered by SHAP explainability
- **A structured 4-section analyst report** written by Llama 3.3 70B via Groq, formatted like a real bank credit memo

---

## Why it's different

Most credit risk projects stop at model accuracy. This one builds the full picture:

| What others do | What this does |
|---|---|
| Train a model, print AUC | Full pipeline from raw data to live deployed app |
| Black-box predictions | SHAP values explain every single prediction |
| No deployment | Live app anyone can use right now |
| Generic output | LLM writes a structured 4-section analyst report |
| Single model | Logistic Regression baseline vs XGBoost |

---

## Tech stack

| Layer | Tools |
|---|---|
| Data & EDA | Python, Pandas, Seaborn, Matplotlib |
| Modelling | Scikit-learn, XGBoost |
| Explainability | SHAP (TreeExplainer) |
| API | FastAPI, Uvicorn, Pydantic |
| LLM layer | LangChain + Groq (Llama 3.3 70B) |
| Dashboard | Streamlit, Plotly |
| Deployment | Streamlit Cloud |

---

## Results

| Metric | Value |
|---|---|
| XGBoost ROC-AUC | **0.86** |
| Baseline LR ROC-AUC | 0.82 |
| Training samples | 120,000 |
| Test samples | 30,000 |
| Dataset | Give Me Some Credit (Kaggle) |

Top predictors identified by SHAP: past due payment history and revolving credit utilization — consistent with real-world credit scoring research.

---

## How it works

**1. Data**
150k customer records cleaned, missing values imputed with median, outliers clipped at 99th percentile. Three new features engineered: debt-to-income ratio, credit utilization bucket, and total past due events.

**2. Modelling**
Logistic Regression trained first as baseline. XGBoost trained with `scale_pos_weight` to handle the 93/7 class imbalance. Best model selected by ROC-AUC on held-out test set.

**3. Explainability**
TreeSHAP computes exact Shapley values per prediction. Top 3 risk drivers returned with direction (increases/decreases risk) and magnitude — not just global feature importance.

**4. API**
FastAPI endpoint accepts customer JSON, runs prediction + SHAP in under 200ms, returns structured response with probability, risk tier, and top factors.

**5. LLM layer**
LangChain passes risk summary to Llama 3.3 70B on Groq. A structured prompt produces a formal 4-section report: Overall Assessment, Key Risk Drivers, Mitigating Factors, Recommendation.

**6. Dashboard**
Streamlit app with input sliders, gauge chart, colour-coded SHAP bar chart, and the analyst report rendered as styled section cards.

---

## Sample API response

```json
{
  "default_probability": 0.94,
  "risk_tier": "High",
  "top_factors": [
    { "feature": "total_past_due",                       "impact": 1.61 },
    { "feature": "RevolvingUtilizationOfUnsecuredLines", "impact": 0.37 },
    { "feature": "NumberOfTimes90DaysLate",              "impact": 0.34 }
  ],
  "explanation": "OVERALL ASSESSMENT: This customer presents a high risk..."
}
```

---

## Run locally

**1. Clone and install**

```bash
git clone https://github.com/swethasays/credit-risk-model.git
cd credit-risk-model
pip install -r requirements.txt
```

**2. Add your free Groq API key** — get one at [console.groq.com](https://console.groq.com)

```bash
echo 'GROQ_API_KEY=your_key_here' > .env
```

**3. Train the model**

```bash
jupyter notebook notebooks/02_modeling.ipynb
```

**4. Start the API**

```bash
uvicorn api.main:app --reload
```

**5. Launch the dashboard**

```bash
streamlit run app/dashboard.py
```

---

## Project structure

```
credit-risk-model/
├── data/
│   ├── raw/                  ← original Kaggle dataset
│   └── processed/            ← cleaned + engineered features
├── notebooks/
│   ├── 01_eda.ipynb          ← exploratory analysis
│   └── 02_modeling.ipynb     ← training + SHAP
├── src/
│   ├── model.pkl             ← trained XGBoost model
│   ├── scaler.pkl            ← feature scaler
│   └── features.pkl          ← feature list
├── api/
│   └── main.py               ← FastAPI REST endpoint
├── app/
│   └── dashboard.py          ← Streamlit dashboard
├── reports/                  ← saved SHAP plots
└── requirements.txt
```

---

## Key decisions

**Why XGBoost over deep learning?**
Better performance on tabular financial data, faster training, and higher interpretability — critical for Basel III regulatory compliance in banking environments.

**Why SHAP?**
TreeSHAP gives exact Shapley values per prediction in polynomial time. Loan officers need to justify individual decisions to regulators — global feature importance alone is not sufficient.

**Why LangChain + Groq?**
Groq's inference speed (~500 tokens/sec on Llama 3.3 70B) makes real-time LLM explanation practical. The structured 4-section prompt mirrors actual bank credit memo format so output is immediately usable by a loan officer.

---

## STAR summary

**Situation**
A retail bank faces increasing loan defaults due to ineffective rule-based credit assessment systems that fail to capture complex customer risk patterns.

**Task**
Build an end-to-end ML pipeline to predict default probability, with interpretable explanations aligned with regulatory requirements.

**Action**
Engineered features from 150k customer records, trained XGBoost with SHAP explainability, built a FastAPI REST endpoint, and added an LLM layer that generates structured analyst reports using Llama 3.3 70B via Groq.

**Result**
ROC-AUC of 0.86, real-time predictions via REST API, and a live dashboard with human-readable 4-section risk assessments — deployed and accessible to anyone.

---

## Dataset

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — 150,000 customer records, 11 financial features, binary default label.