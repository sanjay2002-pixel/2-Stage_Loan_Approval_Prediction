# app.py

import streamlit as st
from src.predict import LoanPredictor

st.set_page_config(page_title="2-Stage Loan Approval System", layout="centered")

st.title("🏦 2-Stage Loan Approval & Valuation System")

st.write("Enter applicant details below:")

# --------------------------
# User Input Fields
# --------------------------

no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Requested Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (months)", min_value=0)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

# --------------------------
# Predict Button
# --------------------------

if st.button("Predict Loan Decision"):

    input_data = {
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    }

    predictor = LoanPredictor()
    result = predictor.predict(input_data)

    st.subheader("📊 Result")

    if result["Loan Status"] == "Approved":
        st.success("✅ Loan Approved")
        st.write(f"💰 Predicted Loan Amount: {result['Predicted Loan Amount']}")
    else:
        st.error("❌ Loan Rejected")