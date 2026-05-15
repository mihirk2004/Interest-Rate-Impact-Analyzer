import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# Load the model from the binary data
@st.cache_resource
def load_model():
    model = joblib.load('economic_growth_xgboost_model.pkl')
    return model

# Function to make predictions
def predict_growth(inflation_rate):
    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'Inflation Rate': [inflation_rate],
        'Inflation Rate_lag1': [inflation_rate * 0.95],  # These can be adjusted as per your model's feature engineering
        'Inflation Rate_lag2': [inflation_rate * 0.90],
        'Inflation Rate_lag3': [inflation_rate * 0.85]
    })
    
    # Ensure that the input data has the same feature names as during model training
    model_columns = model.get_booster().feature_names  # Get the feature names expected by the model
    input_data = input_data[model_columns]  # Align the input data columns with model's expected columns

    # Make prediction
    prediction = model.predict(input_data)  # Directly pass the DataFrame without xgb.DMatrix
    return prediction[0]


# Load the model
model = load_model()

# Streamlit app
st.title('Economic Growth Prediction Dashboard')
st.markdown("""This app predicts economic growth based on inflation rate using an XGBoost model.
Adjust the inflation rate slider to see how it affects the predicted growth.""")

# Sidebar with user input
st.sidebar.header('User Input Parameters')

# Inflation rate slider
inflation_rate = st.sidebar.slider(
    'Inflation Rate (%)', 
    min_value=0.0, 
    max_value=10.0, 
    value=2.5, 
    step=0.1
)

# Make prediction
if st.sidebar.button('Predict Economic Growth'):
    growth_prediction = predict_growth(inflation_rate)
    st.success(f'Predicted Economic Growth: {growth_prediction:.2f}%')

# Main content
st.header('How Inflation Affects Economic Growth')
st.write("""
The relationship between inflation and economic growth is complex. Moderate inflation 
is often associated with economic growth, while high inflation can be detrimental.
""")

# Model information
st.header('Model Information')
st.write("""
- **Model Type**: XGBoost Regressor
- **Target Variable**: Economic Growth (%)
- **Main Feature**: Inflation Rate (%)
- **Additional Features**: Lagged inflation rates (1-3 periods)
""")

# Feature importance (if available)
try:
    st.subheader('Feature Importance')
    importance = model.get_booster().get_score(importance_type='weight')
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values('Importance', ascending=False)
    
    st.bar_chart(importance_df.set_index('Feature'))
except:
    st.info("Feature importance data not available for this model.")

# Download button for sample data
st.sidebar.header('Data')
st.sidebar.download_button(
    label="Download Sample Data Format",
    data=pd.DataFrame({
        'Inflation Rate': [2.5],
        'Inflation Rate_lag1': [2.4],
        'Inflation Rate_lag2': [2.3],
        'Inflation Rate_lag3': [2.2]
    }).to_csv(index=False).encode('utf-8'),
    file_name='economic_growth_sample_input.csv',
    mime='text/csv'
)

# Add some spacing
st.sidebar.markdown("---")
st.sidebar.info("Adjust the parameters and click 'Predict' to see results.")
