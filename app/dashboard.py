import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib
import shap
import os

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide"
)

# Load Groq key — direct from st.secrets, fallback to .env
GROQ_API_KEY = None
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    except:
        pass

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

    if GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=GROQ_API_KEY)
            prompt = PromptTemplate.from_template("""
You are a senior credit risk analyst. Write a credit assessment using EXACTLY this format.
Each label must appear exactly once, at the start of its line, followed by a colon and a space, then the content.
Do not repeat any label. Do not add any extra text before OVERALL.

OVERALL: [one sentence verdict on the customer based on {risk_tier} risk and {prob:.0%} default probability]
DRIVERS: [2-3 sentences explaining these factors in plain English: {factors}]
MITIGATING: [one sentence on what reduces this customer risk, or state: No significant mitigating factors identified]
RECOMMENDATION: [one sentence - approve, decline, or conditional approval with conditions]

Rules:
- Output exactly 4 lines, one per label
- No bullet points, no HTML, no markdown, no code blocks, no backticks
- Do not repeat labels
- Professional banking language only
""")
            chain = prompt | llm | StrOutputParser()
            return chain.invoke({
                "risk_tier": risk_tier,
                "prob":      prob,
                "factors":   factors_text
            })
        except Exception as e:
            st.warning(f"Groq error: {e}")

    return (
        f"OVERALL: This customer presents a {risk_tier.lower()} credit risk with a {prob:.0%} probability of default.\n"
        f"DRIVERS: Key factors include {factors_text}.\n"
        f"MITIGATING: No significant mitigating factors identified.\n"
        f"RECOMMENDATION: Further manual review is recommended before making a final decision."
    )


def render_explanation(explanation):
    section_map = {
        "OVERALL":        ("🔵 Overall Assessment", "info"),
        "DRIVERS":        ("🔴 Key Risk Drivers",    "error"),
        "MITIGATING":     ("🟢 Mitigating Factors",  "success"),
        "RECOMMENDATION": ("🟡 Recommendation",      "warning"),
    }

    # Clean any markdown or HTML artifacts
    clean = explanation
    for artifact in ["**", "```", "</div>", "<div>", "`"]:
        clean = clean.replace(artifact, "")
    clean = clean.strip()

    # Parse each line — match KEY: content format, ignore duplicates
    found = {key: None for key in section_map}
    for line in clean.split("\n"):
        line = line.strip()
        if not line:
            continue
        for key in section_map:
            if line.upper().startswith(key + ":"):
                body = line[len(key)+1:].strip()
                if found[key] is None:
                    found[key] = body
                break

    # Render in order using native Streamlit components
    for key, (label, stype) in section_map.items():
        body = found[key] or "No information available."
        msg  = f"**{label}**\n\n{body}"
        if stype == "info":
            st.info(msg)
        elif stype == "error":
            st.error(msg)
        elif stype == "success":
            st.success(msg)
        elif stype == "warning":
            st.warning(msg)


st.title("🏦 Credit Risk Prediction Dashboard")
st.markdown("Real-time loan default prediction powered by XGBoost + SHAP + Groq LLM")

tab1, tab2 = st.tabs(["Predict", "Model Performance"])

with tab1:
    st.subheader("Customer Profile")
    st.caption("Fill in the customer's financial details and click Predict. Hover over the ? icons to learn what each field means.")

    col1, col2, col3 = st.columns(3)

    with col1:
        revolving_util = st.slider(
            "Revolving Utilization", 0.0, 1.0, 0.5,
            help="How much of the credit card limit is being used. 0.9 = using 90% of limit. Above 0.6 is risky.")
        age = st.slider(
            "Age", 18, 100, 45,
            help="Customer's age. Younger customers tend to have higher default rates.")
        monthly_income = st.number_input(
            "Monthly Income ($)", 0, 100000, 5000,
            help="Gross monthly income in dollars. Higher income generally means lower risk.")
        debt_ratio = st.slider(
            "Debt Ratio", 0.0, 1.0, 0.3,
            help="Monthly debt payments divided by monthly income. Above 0.5 is concerning.")

    with col2:
        past_due_30_59 = st.number_input(
            "Times 30-59 Days Past Due", 0, 20, 0,
            help="How many times the customer was 30-59 days late on a payment.")
        past_due_60_89 = st.number_input(
            "Times 60-89 Days Past Due", 0, 20, 0,
            help="How many times the customer was 60-89 days late.")
        times_90_late = st.number_input(
            "Times 90+ Days Late", 0, 20, 0,
            help="How many times 90+ days late. This is a major red flag.")
        dependents = st.number_input(
            "Number of Dependents", 0, 20, 1,
            help="Number of people financially dependent on this customer.")

    with col3:
        open_credits = st.number_input(
            "Open Credit Lines", 0, 50, 5,
            help="Total number of open loans and credit cards.")
        real_estate = st.number_input(
            "Real Estate Loans", 0, 20, 1,
            help="Number of mortgage or real estate loans. Having 1-2 is normal and positive.")

    with st.expander("How to interpret the results"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.success("**Low Risk** (< 15%)\nCustomer is unlikely to default. Loan can likely be approved.")
        with c2:
            st.warning("**Medium Risk** (15–40%)\nCustomer shows some risk factors. Consider further review.")
        with c3:
            st.error("**High Risk** (> 40%)\nCustomer has significant default risk. Careful assessment recommended.")

    debt_to_income = debt_ratio * monthly_income
    total_past_due = int(past_due_30_59 + past_due_60_89 + times_90_late)
    if revolving_util < 0.3:
        credit_util_bucket = 0.0
    elif revolving_util < 0.6:
        credit_util_bucket = 1.0
    elif revolving_util < 0.9:
        credit_util_bucket = 2.0
    else:
        credit_util_bucket = 3.0

    if st.button("Predict Default Risk", type="primary"):
        input_data = np.array([[
            revolving_util, age, int(past_due_30_59), debt_ratio,
            float(monthly_income), int(open_credits), int(times_90_late),
            int(real_estate), int(past_due_60_89), float(dependents),
            float(debt_to_income), total_past_due, credit_util_bucket
        ]])

        with st.spinner("Analysing customer profile..."):
            prob        = float(model.predict_proba(input_data)[0][1])
            risk_tier   = get_risk_tier(prob)
            shap_vals   = explainer.shap_values(input_data)[0]
            top_factors = get_top_shap_factors(shap_vals, features)
            explanation = generate_explanation(prob, risk_tier, top_factors)

            color = {"Low": "green", "Medium": "orange", "High": "red"}[risk_tier]

            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Default Probability", f"{prob:.1%}")
            m2.metric("Risk Tier", risk_tier)
            m3.metric("Total Past Due Events", total_past_due)

            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = prob * 100,
                title = {"text": "Default Probability (%)"},
                gauge = {
                    "axis":  {"range": [0, 100]},
                    "bar":   {"color": color},
                    "steps": [
                        {"range": [0,  15], "color": "#d4edda"},
                        {"range": [15, 40], "color": "#fff3cd"},
                        {"range": [40, 100],"color": "#f8d7da"},
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Top Risk Factors")
            factor_df = pd.DataFrame(top_factors)
            colors    = ["red" if x > 0 else "green" for x in factor_df["impact"]]
            fig2, ax  = plt.subplots(figsize=(8, 3))
            ax.barh(factor_df["feature"], factor_df["impact"], color=colors)
            ax.set_xlabel("SHAP Impact")
            ax.set_title("Feature contributions to this prediction")
            ax.axvline(0, color="black", linewidth=0.8)
            st.pyplot(fig2)

            st.subheader("Analyst Explanation")
            render_explanation(explanation)

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
                     caption="ROC curve + confusion matrix")
        except Exception as e:
            st.error(f"model_performance.png error: {e}")

    st.divider()
    st.markdown("**SHAP explainability**")
    c1, c2, c3 = st.columns(3)
    with c1:
        try:
            st.image("reports/shap_bar.png",
                     caption="Global feature importance")
        except Exception as e:
            st.error(f"shap_bar.png error: {e}")
    with c2:
        try:
            st.image("reports/shap_beeswarm.png",
                     caption="Beeswarm plot")
        except Exception as e:
            st.error(f"shap_beeswarm.png error: {e}")
    with c3:
        try:
            st.image("reports/shap_waterfall.png",
                     caption="Single prediction waterfall")
        except Exception as e:
            st.error(f"shap_waterfall.png error: {e}")