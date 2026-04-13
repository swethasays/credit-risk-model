import joblib
import numpy as np
import shap
import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Credit Risk API", version="1.0")

model    = joblib.load("src/model.pkl")
scaler   = joblib.load("src/scaler.pkl")
features = joblib.load("src/features.pkl")

explainer = shap.TreeExplainer(model)


class CustomerData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: float
    debt_to_income_ratio: float
    total_past_due: int
    credit_util_bucket: float


def get_risk_tier(prob: float) -> str:
    if prob < 0.15:
        return "Low"
    elif prob < 0.40:
        return "Medium"
    return "High"


def get_top_shap_factors(shap_vals, feature_names, top_n=3):
    pairs = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)
    return [{"feature": f, "impact": round(float(v), 4)} for f, v in pairs[:top_n]]


def generate_explanation(prob: float, risk_tier: str, top_factors: list) -> str:
    factors_text = ", ".join(
        [f"{f['feature']} ({'increases' if f['impact'] > 0 else 'decreases'} risk by {abs(f['impact']):.2f})"
         for f in top_factors]
    )

    groq_key = os.getenv("GROQ_API_KEY")

    if groq_key:
        try:
            llm = ChatGroq(model="llama3-8b-8192", temperature=0.3, api_key=groq_key)
            prompt = PromptTemplate.from_template(
                "You are a credit risk analyst at a bank. "
                "A customer has a {risk_tier} risk level "
                "with a {prob:.0%} probability of defaulting on their loan. "
                "The top contributing factors are: {factors}. "
                "Write a 2-sentence professional explanation for a loan officer."
            )
            chain = prompt | llm | StrOutputParser()
            return chain.invoke({
                "risk_tier": risk_tier,
                "prob":      prob,
                "factors":   factors_text
            })
        except Exception as e:
            print(f"Groq error: {e}")

    return (
        f"This customer presents a {risk_tier.lower()} credit risk with a "
        f"{prob:.0%} estimated probability of default. "
        f"Key contributing factors include {factors_text}."
    )


@app.get("/")
def root():
    return {"message": "Credit Risk API is running"}


@app.post("/predict")
def predict(customer: CustomerData):
    input_data = np.array([[
        customer.RevolvingUtilizationOfUnsecuredLines,
        customer.age,
        customer.NumberOfTime30_59DaysPastDueNotWorse,
        customer.DebtRatio,
        customer.MonthlyIncome,
        customer.NumberOfOpenCreditLinesAndLoans,
        customer.NumberOfTimes90DaysLate,
        customer.NumberRealEstateLoansOrLines,
        customer.NumberOfTime60_89DaysPastDueNotWorse,
        customer.NumberOfDependents,
        customer.debt_to_income_ratio,
        customer.total_past_due,
        customer.credit_util_bucket
    ]])

    prob        = float(model.predict_proba(input_data)[0][1])
    risk_tier   = get_risk_tier(prob)
    shap_vals   = explainer.shap_values(input_data)[0]
    top_factors = get_top_shap_factors(shap_vals, features)
    explanation = generate_explanation(prob, risk_tier, top_factors)

    return {
        "default_probability": round(prob, 4),
        "risk_tier":           risk_tier,
        "top_factors":         top_factors,
        "explanation":         explanation
    }