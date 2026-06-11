from __future__ import annotations

import os

import requests
import streamlit as st


API_URL = os.getenv("LOAN_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("LOAN_API_KEY", "loan123")


st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
)

st.title("Loan Approval Prediction")
st.caption("Applicant, credit, and loan details are scored by the FastAPI model service.")

with st.sidebar:
    st.header("Service")

    try:
        health_response = requests.get(f"{API_URL}/health", timeout=3)
        if health_response.ok and health_response.json().get("model_loaded"):
            st.success("API online")
        elif health_response.ok:
            st.warning("API online, model missing")
        else:
            st.error("API unavailable")
    except requests.RequestException:
        st.error("API unavailable")

    st.text_input("API URL", value=API_URL, disabled=True)

st.divider()

st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["Yes", "No"], format_func=lambda x: "Married" if x == "Yes" else "Not Married")
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

with col2:
    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    self_employed = st.selectbox("Employment Type", ["No", "Yes"], format_func=lambda x: "Self Employed" if x == "Yes" else "Salaried")
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

st.divider()
st.subheader("Income Information")

col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input(
        "Applicant Monthly Income",
        min_value=0.0,
        value=5000.0,
        step=500.0,
    )

with col2:
    coapplicant_income = st.number_input(
        "Co-Applicant Monthly Income",
        min_value=0.0,
        value=0.0,
        step=500.0,
    )

st.divider()
st.subheader("Loan Information")

col1, col2 = st.columns(2)

with col1:
    loan_amount = st.number_input(
        "Loan Amount (in thousands)",
        min_value=1.0,
        value=120.0,
        step=5.0,
    )

with col2:
    loan_term = st.selectbox(
        "Loan Term (Months)",
        [12, 36, 60, 120, 180, 240, 300, 360, 480],
        index=7,
    )

st.divider()
st.subheader("Credit Information")

credit_history = st.selectbox(
    "Credit History",
    [1.0, 0.0],
    format_func=lambda x: "Good" if x == 1.0 else "Poor",
)

st.divider()

payload = {
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": float(loan_term),
    "Credit_History": credit_history,
    "Property_Area": property_area,
}

if st.button("Predict Loan Approval", use_container_width=True):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers={"X-API-KEY": API_KEY},
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()

        st.subheader("Prediction Result")

        probability = result["approval_probability"]
        if result["prediction"] == "Approved":
            st.success(f"Loan is likely to be approved. Approval probability: {probability:.1%}")
        else:
            st.error(f"Loan is likely to be rejected. Approval probability: {probability:.1%}")

        st.progress(probability)

        with st.expander("Response details"):
            st.json(result)

    except requests.HTTPError as exc:
        st.error(f"API error: {exc.response.status_code}")
        st.code(exc.response.text)
    except requests.RequestException as exc:
        st.error("Unable to connect to the FastAPI server.")
        st.info("Start the API with: uvicorn api.main:app --reload")
        st.code(str(exc))
