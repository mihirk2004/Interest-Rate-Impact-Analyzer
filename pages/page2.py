import streamlit as st
import pandas as pd

import numpy as np
import joblib
import xgboost as xgb

# Load the model from the binary data
@st.cache_resource
def load_model():
    model = joblib.load(r'E:\Admin\User\Mihir\Documents\Mihir Codes\Sem 4\AI Assignments\AI PBL\dependencies\inflation and economic growth\economic_growth_xgboost_model.pkl')
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
# try:
#     st.subheader('Feature Importance')
#     importance = model.get_booster().get_score(importance_type='weight')
#     importance_df = pd.DataFrame({
#         'Feature': list(importance.keys()),
#         'Importance': list(importance.values())
#     }).sort_values('Importance', ascending=False)
    
#     st.bar_chart(importance_df.set_index('Feature'))
# except:
#     st.info("Feature importance data not available for this model.")

# # Download button for sample data
# st.sidebar.header('Data')
# st.sidebar.download_button(
#     label="Download Sample Data Format",
#     data=pd.DataFrame({
#         'Inflation Rate': [2.5],
#         'Inflation Rate_lag1': [2.4],
#         'Inflation Rate_lag2': [2.3],
#         'Inflation Rate_lag3': [2.2]
#     }).to_csv(index=False).encode('utf-8'),
#     file_name='economic_growth_sample_input.csv',
#     mime='text/csv'
# )

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load and prepare data
data = pd.read_csv(r'E:\Admin\User\Mihir\Documents\Mihir Codes\Sem 4\AI Assignments\AI PBL\dependencies\inflation and economic growth\inflation and economic growth.csv')
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data.set_index('Date', inplace=True)

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 7))

# Plot data
ax.plot(data.index, data['Inflation Rate'], label='Inflation Rate', color='#E63946', linewidth=2)
ax.plot(data.index, data['Economic Growth Rate'], label='Economic Growth', color='#1D3557', linewidth=2)

# Formatting
ax.set_title('Inflation and Economic Growth Trends (2000-2025)', pad=20, fontsize=14)
ax.set_xlabel('Year', labelpad=10)
ax.set_ylabel('Percentage (%)', labelpad=10)
ax.grid(True, linestyle='--', alpha=0.7)

# Legend
ax.legend(framealpha=1, shadow=True)

# Format x-axis
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate()

# Highlight key periods
ax.axvspan('2008-01-01', '2009-12-31', color='gray', alpha=0.2, label='Financial Crisis')
ax.axvspan('2020-01-01', '2021-12-31', color='red', alpha=0.1, label='COVID-19 Pandemic')

plt.tight_layout()
st.pyplot(fig)

# Add some spacing
st.sidebar.markdown("---")
st.sidebar.info("Adjust the parameters and click 'Predict' to see results.")
