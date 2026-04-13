import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib
import shap

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide"
)

model    = joblib.load("src/model.pkl")
scaler   = joblib.load("src/scaler.pkl")
features = joblib.load("src/features.pkl")
explainer = shap.TreeExplainer(model)

st.title("🏦 Credit Risk Prediction Dashboard")
st.markdown("Real-time loan default prediction powered by XGBoost + SHAP + Groq LLM")

tab1, tab2 = st.tabs(["Predict", "Model Performance"])

with tab1:
    st.subheader("Customer Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        revolving_util   = st.slider("Revolving Utilization", 0.0, 1.0, 0.5)
        age              = st.slider("Age", 18, 100, 45)
        monthly_income   = st.number_input("Monthly Income ($)", 0, 100000, 5000)
        debt_ratio       = st.slider("Debt Ratio", 0.0, 1.0, 0.3)

    with col2:
        past_due_30_59   = st.number_input("Times 30-59 Days Past Due", 0, 20, 0)
        past_due_60_89   = st.number_input("Times 60-89 Days Past Due", 0, 20, 0)
        times_90_late    = st.number_input("Times 90+ Days Late",        0, 20, 0)
        dependents       = st.number_input("Number of Dependents",       0, 20, 1)

    with col3:
        open_credits     = st.number_input("Open Credit Lines", 0, 50, 5)
        real_estate      = st.number_input("Real Estate Loans", 0, 20, 1)

    # Derived features
    debt_to_income   = debt_ratio * monthly_income
    total_past_due   = past_due_30_59 + past_due_60_89 + times_90_late
    if revolving_util < 0.3:
        credit_util_bucket = 0.0
    elif revolving_util < 0.6:
        credit_util_bucket = 1.0
    elif revolving_util < 0.9:
        credit_util_bucket = 2.0
    else:
        credit_util_bucket = 3.0

    if st.button("Predict Default Risk", type="primary"):
        payload = {
            "RevolvingUtilizationOfUnsecuredLines": revolving_util,
            "age":                                  age,
            "NumberOfTime30_59DaysPastDueNotWorse": int(past_due_30_59),
            "DebtRatio":                            debt_ratio,
            "MonthlyIncome":                        float(monthly_income),
            "NumberOfOpenCreditLinesAndLoans":      int(open_credits),
            "NumberOfTimes90DaysLate":              int(times_90_late),
            "NumberRealEstateLoansOrLines":         int(real_estate),
            "NumberOfTime60_89DaysPastDueNotWorse": int(past_due_60_89),
            "NumberOfDependents":                   float(dependents),
            "debt_to_income_ratio":                 float(debt_to_income),
            "total_past_due":                       int(total_past_due),
            "credit_util_bucket":                   float(credit_util_bucket)
        }

        with st.spinner("Analysing customer profile..."):
            try:
                response = requests.post("http://127.0.0.1:8000/predict", json=payload)
                result   = response.json()

                prob      = result["default_probability"]
                tier      = result["risk_tier"]
                factors   = result["top_factors"]
                explanation = result["explanation"]

                # Colour by tier
                color = {"Low": "green", "Medium": "orange", "High": "red"}[tier]

                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("Default Probability", f"{prob:.1%}")
                m2.metric("Risk Tier", tier)
                m3.metric("Total Past Due Events", total_past_due)

                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode  = "gauge+number",
                    value = prob * 100,
                    title = {"text": "Default Probability (%)"},
                    gauge = {
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": color},
                        "steps": [
                            {"range": [0,  15], "color": "#d4edda"},
                            {"range": [15, 40], "color": "#fff3cd"},
                            {"range": [40, 100],"color": "#f8d7da"},
                        ]
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                # SHAP factors bar
                st.subheader("Top Risk Factors")
                factor_df = pd.DataFrame(factors)
                colors    = ["red" if x > 0 else "green" for x in factor_df["impact"]]
                fig2, ax  = plt.subplots(figsize=(8, 3))
                ax.barh(factor_df["feature"], factor_df["impact"], color=colors)
                ax.set_xlabel("SHAP Impact")
                ax.set_title("Feature contributions to this prediction")
                ax.axvline(0, color="black", linewidth=0.8)
                st.pyplot(fig2)

                # LLM explanation
                st.subheader("Analyst Explanation")
                st.info(explanation)

            except Exception as e:
                st.error(f"API error: {e}. Make sure the FastAPI server is running.")

with tab2:
    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("XGBoost ROC-AUC", "0.86")
        st.metric("Baseline LR ROC-AUC", "0.82")
        st.metric("Training samples", "120,000")
        st.metric("Test samples", "30,000")

    with col2:
        st.markdown("**SHAP plots**")
        try:
            st.image("reports/shap_bar.png",       caption="Global feature importance")
            st.image("reports/shap_beeswarm.png",  caption="SHAP beeswarm")
            st.image("reports/shap_waterfall.png", caption="Single prediction waterfall")
        except:
            st.info("Run the modeling notebook first to generate SHAP plots.")

    try:
        st.image("reports/model_performance.png", caption="ROC curve + confusion matrix")
    except:
        pass