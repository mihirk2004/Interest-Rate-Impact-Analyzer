import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# Load the saved model
@st.cache_resource
def load_model():
    model = joblib.load(r'C:\Users\Admin\Documents\Mihir Codes\Sem 4\AI Assignments\AI PBL\dependencies\interest rate and savings\savings_rate_model.pkl')
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

# Create sliders for each interest rate
st.sidebar.subheader('Interest Rates')
one_year = st.sidebar.slider('Interest Rate (%)', 0.0, 15.0, 2.5, 0.1)
two_year = st.sidebar.slider('2 Year Interest Rate (%)', 0.0, 15.0, 2.8, 0.1)
three_year = st.sidebar.slider('3 Year Interest Rate (%)', 0.0, 15.0, 3.0, 0.1)
five_year = st.sidebar.slider('5 Year Interest Rate (%)', 0.0, 15.0, 3.3, 0.1)

# Date inputs
st.sidebar.subheader('Date Information')
year = st.sidebar.slider('Year', 2000, 2030, 2023)
month = st.sidebar.slider('Month', 1, 12, 6)

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

# Visualization section
st.subheader('Interest Rate vs Savings Rate Relationship')

# Generate a range of interest rates for visualization
interest_rates = np.linspace(0, 15, 30)
savings_predictions = []

# Create a base dataframe for predictions
base_data = input_data.copy()
for rate in interest_rates:
    base_data['1 Year'] = rate
    base_data['2 Year'] = rate + 0.3  # Keep some spread between rates
    base_data['3 Year'] = rate + 0.5
    base_data['5 Year'] = rate + 0.8
    pred = model.predict(base_data)
    savings_predictions.append(pred[0])

# Plot the relationship
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(interest_rates, savings_predictions, 'b-', linewidth=2)
ax.set_xlabel('1 Year Interest Rate (%)', fontsize=12)
ax.set_ylabel('Predicted Savings Rate (%)', fontsize=12)
ax.set_title('Impact of Interest Rates on Savings Rate', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig)

# Add explanation
st.write("""
The chart above shows how changes in the 1-year interest rate (while maintaining proportional spreads for other rates) 
affect the predicted savings rate. Generally, higher interest rates tend to encourage more savings as they offer 
better returns on savings accounts and fixed-income investments.
""")

# Add some model information
st.sidebar.subheader('Model Information')
st.sidebar.write("""
- **Model Type**: XGBoost Regressor
- **Features Used**: 
  - 1-5 Year Interest Rates
  - Year and Month
- **Target**: Savings Rate (%)
""")

# # Add feature importance plot
# st.subheader('Feature Importance')
# try:
#     fig, ax = plt.subplots()
#     xgb.plot_importance(model, ax=ax)
#     st.pyplot(fig)
# except:
#     st.write("Could not display feature importance plot. Make sure matplotlib is installed.")