import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib
import shap
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model     = joblib.load("src/model.pkl")
    scaler    = joblib.load("src/scaler.pkl")
    features  = joblib.load("src/features.pkl")
    explainer = shap.TreeExplainer(model)
    return model, scaler, features, explainer

model, scaler, features, explainer = load_artifacts()


def get_risk_tier(prob):
    if prob < 0.15:
        return "Low"
    elif prob < 0.40:
        return "Medium"
    return "High"


def get_top_shap_factors(shap_vals, feature_names, top_n=3):
    pairs = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)
    return [{"feature": f, "impact": round(float(v), 4)} for f, v in pairs[:top_n]]


def generate_explanation(prob, risk_tier, top_factors):
    factors_text = ", ".join(
        [f"{f['feature']} ({'increases' if f['impact'] > 0 else 'decreases'} risk by {abs(f['impact']):.2f})"
         for f in top_factors]
    )

    groq_key = os.getenv("GROQ_API_KEY")

    if groq_key:
        try:
            from langchain_groq import ChatGroq
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            llm    = ChatGroq(model="llama3-8b-8192", temperature=0.3, api_key=groq_key)
            prompt = PromptTemplate.from_template(
                "You are a credit risk analyst at a bank. "
                "A customer has a {risk_tier} risk level "
                "with a {prob:.0%} probability of defaulting on their loan. "
                "The top contributing factors are: {factors}. "
                "Write a 2-sentence professional explanation for a loan officer."
            )
            chain = prompt | llm | StrOutputParser()
            return chain.invoke({"risk_tier": risk_tier, "prob": prob, "factors": factors_text})
        except Exception as e:
            print(f"Groq error: {e}")

    return (
        f"This customer presents a {risk_tier.lower()} credit risk with a "
        f"{prob:.0%} estimated probability of default. "
        f"Key contributing factors include {factors_text}."
    )


st.title("🏦 Credit Risk Prediction Dashboard")
st.markdown("Real-time loan default prediction powered by XGBoost + SHAP + Groq LLM")

tab1, tab2 = st.tabs(["Predict", "Model Performance"])

with tab1:
    st.subheader("Customer Profile")
    st.caption("Fill in the customer's financial details and click Predict. Hover over the ? icons to learn what each field means.")

    col1, col2, col3 = st.columns(3)

    with col1:
        revolving_util = st.slider(
            "Revolving Utilization",
            0.0, 1.0, 0.5,
            help="How much of the credit card limit is being used. 0.9 = using 90% of limit. Above 0.6 is risky.")
        age = st.slider(
            "Age",
            18, 100, 45,
            help="Customer's age. Younger customers tend to have higher default rates.")
        monthly_income = st.number_input(
            "Monthly Income ($)",
            0, 100000, 5000,
            help="Gross monthly income in dollars. Higher income generally means lower risk.")
        debt_ratio = st.slider(
            "Debt Ratio",
            0.0, 1.0, 0.3,
            help="Monthly debt payments divided by monthly income. Above 0.5 is concerning.")

    with col2:
        past_due_30_59 = st.number_input(
            "Times 30-59 Days Past Due",
            0, 20, 0,
            help="How many times the customer was 30-59 days late on a payment. Even 1-2 is a warning sign.")
        past_due_60_89 = st.number_input(
            "Times 60-89 Days Past Due",
            0, 20, 0,
            help="How many times the customer was 60-89 days late. More serious than 30-59 days.")
        times_90_late = st.number_input(
            "Times 90+ Days Late",
            0, 20, 0,
            help="How many times the customer was 90+ days late. This is a major red flag for default risk.")
        dependents = st.number_input(
            "Number of Dependents",
            0, 20, 1,
            help="Number of people financially dependent on this customer (children, elderly parents etc).")

    with col3:
        open_credits = st.number_input(
            "Open Credit Lines",
            0, 50, 5,
            help="Total number of open loans and credit cards. Very high numbers can indicate financial stress.")
        real_estate = st.number_input(
            "Real Estate Loans",
            0, 20, 1,
            help="Number of mortgage or real estate loans. Having 1-2 is normal and generally a positive signal.")

    # Risk guide
    with st.expander("How to interpret the results"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.success("**Low Risk** (< 15%)\nCustomer is unlikely to default. Loan can likely be approved.")
        with c2:
            st.warning("**Medium Risk** (15–40%)\nCustomer shows some risk factors. Consider further review or conditions.")
        with c3:
            st.error("**High Risk** (> 40%)\nCustomer has significant default risk. Careful assessment recommended.")
            
with tab2:
    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("XGBoost ROC-AUC",     "0.86")
        st.metric("Baseline LR ROC-AUC", "0.82")
        st.metric("Training samples",    "120,000")
        st.metric("Test samples",        "30,000")

    with col2:
        try:
            st.image("reports/model_performance.png",
                     caption="ROC curve + confusion matrix",
                     use_column_width=True)
        except:
            pass

    st.divider()
    st.markdown("**SHAP explainability**")

    c1, c2, c3 = st.columns(3)
    with c1:
        try:
            st.image("reports/shap_bar.png",
                     caption="Global feature importance",
                     use_column_width=True)
        except:
            st.info("Run modeling notebook first.")
    with c2:
        try:
            st.image("reports/shap_beeswarm.png",
                     caption="Beeswarm plot",
                     use_column_width=True)
        except:
            pass
    with c3:
        try:
            st.image("reports/shap_waterfall.png",
                     caption="Single prediction waterfall",
                     use_column_width=True)
        except:
            pass