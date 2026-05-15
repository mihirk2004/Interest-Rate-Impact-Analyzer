import streamlit as st
import joblib
import pandas as pd

# Load model once
model = joblib.load("inflation_loan_xgboost_model.pkl")

# UI
st.title("📊 Loan Prediction Dashboard")
st.write("Use the inflation rate slider to predict future loan rates.")

# Input
inflation_rate = st.slider("Inflation Rate (%)", 0.0, 20.0, 5.0, 0.1)

# Generate lag values dynamically with decay
def simulate_lags(current, decay=0.9):
    return {
        "Lag_1": current * decay,
        "Lag_3": current * (decay ** 2),
        "Lag_6": current * (decay ** 3),
        "Lag_12": current * (decay ** 4)
    }

# Build input data
infl_lags = simulate_lags(inflation_rate)
loan_lags = simulate_lags(inflation_rate * 0.8)  # simulate loan following inflation

input_data = {
    "Inflation_Rate": inflation_rate,
    "Inflation_Rate_Lag_1": infl_lags["Lag_1"],
    "Inflation_Rate_Lag_3": infl_lags["Lag_3"],
    "Inflation_Rate_Lag_6": infl_lags["Lag_6"],
    "Inflation_Rate_Lag_12": infl_lags["Lag_12"],
    "Loan_Rate_Lag_1": loan_lags["Lag_1"],
    "Loan_Rate_Lag_3": loan_lags["Lag_3"],
    "Loan_Rate_Lag_6": loan_lags["Lag_6"],
    "Loan_Rate_Lag_12": loan_lags["Lag_12"]
}

input_df = pd.DataFrame([input_data])

# Predict on button click
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.subheader("📈 Prediction")
    st.metric("Predicted Loan Rate", round(prediction, 2))
