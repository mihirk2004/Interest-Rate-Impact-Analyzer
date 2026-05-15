import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import joblib

# Load the saved model
@st.cache_resource
def load_model():
    model = joblib.load('savings_rate_model.pkl')
    return model

model = load_model()

# Set up the app
st.title('Savings Rate Prediction Model')
st.write("""
This app predicts the savings rate based on interest rates and other economic indicators.
Adjust the sliders to see how changes in interest rates affect the predicted savings rate.
""")

# Sidebar with input controls
st.sidebar.header('Input Parameters')

# Get default values (these would ideally come from your training data statistics)
default_values = {
    '1 Year': 2.5,
    '2 Year': 2.8,
    '3 Year': 3.0,
    '5 Year': 3.3,
    'Year': 2023,
    'Month': 6
}

# Create sliders for each interest rate
st.sidebar.subheader('Interest Rates')
one_year = 2.5
two_year = 2.8
three_year = 3.0
five_year = 3.3

# Date inputs
st.sidebar.subheader('Date Information')
year = st.sidebar.slider('Year', 2000, 2030, default_values['Year'])
month = st.sidebar.slider('Month', 1, 12, default_values['Month'])

# Create input dataframe
input_data = pd.DataFrame({
    '1 Year': [one_year],
    '2 Year': [two_year],
    '3 Year': [three_year],
    '5 Year': [five_year],
    'Year': [year],
    'Month': [month]
})

# Display input parameters
st.subheader('Current Input Parameters')
st.write(input_data)

# Make prediction
if st.button('Predict Savings Rate'):
    prediction = model.predict(input_data)
    st.subheader('Predicted Savings Rate')
    st.write(f"{prediction[0]:.2f}%")
    
    # Add some interpretation
    st.subheader('Interpretation')
    st.write(f"Based on the current interest rates and time period, the model predicts a savings rate of {prediction[0]:.2f}%.")
    
    # Add some basic analysis
    if prediction[0] < 5:
        st.warning("Low savings rate predicted. This may indicate strong consumer spending.")
    elif prediction[0] > 10:
        st.success("High savings rate predicted. This may indicate cautious consumer behavior.")
    else:
        st.info("Moderate savings rate predicted. The economy appears to be in a balanced state.")

# Add some model information
st.sidebar.subheader('Model Information')
st.sidebar.write("""
- **Model Type**: XGBoost Regressor
- **Features Used**: 
  - 1-5 Year Interest Rates
  - Year and Month
- **Target**: Savings Rate (%)
""")

# Add feature importance plot
st.subheader('Feature Importance')
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    xgb.plot_importance(model, ax=ax)
    st.pyplot(fig)
except:
    st.write("Could not display feature importance plot. Make sure matplotlib is installed.")