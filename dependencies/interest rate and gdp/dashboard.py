import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained model and scaler
@st.cache_resource
def load_assets():
    model = load_model('lstm_gdp_predictor.h5')
    scaler = joblib.load('scaler.save')
    return model, scaler

model, scaler = load_assets()

# Function to create input sequence for prediction
def prepare_input(interest_rate, last_n_values=12):
    # Load historical data to get the last sequence
    df = pd.read_csv('Interest Rate and Gdp.csv', parse_dates=['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)
    df = df.resample('M').last().ffill()
    
    # Get the last SEQ_LENGTH-1 values
    last_sequence = df[['Average', 'Gdp Per Capita']].iloc[-(last_n_values-1):].values
    
    # Create new row with user input and dummy GDP (will be scaled properly)
    new_row = np.array([[interest_rate, 0]])  # GDP will be replaced during scaling
    
    # Combine to make full sequence
    full_sequence = np.vstack([last_sequence, new_row])
    
    # Scale the data
    scaled_sequence = scaler.transform(full_sequence)
    
    # Reshape for LSTM (1 sample, SEQ_LENGTH timesteps, n_features)
    return scaled_sequence.reshape(1, last_n_values, 2)

# Streamlit app
st.title('GDP Per Capita Predictor')
st.markdown("""
This app predicts GDP Per Capita based on interest rate input using a pre-trained LSTM model.
""")

# Sidebar for user input
st.sidebar.header('User Input Parameters')
interest_rate = st.sidebar.slider(
    'Select Interest Rate (%)',
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.1
)

# Main panel
st.subheader('Prediction')

# Prepare input and make prediction
input_sequence = prepare_input(interest_rate)
scaled_prediction = model.predict(input_sequence)

# Inverse transform the prediction
dummy = np.zeros((1, 2))
dummy[0, 1] = scaled_prediction[0, 0]  # GDP is the second feature
predicted_gdp = scaler.inverse_transform(dummy)[0, 1]

st.success(f'Predicted GDP Per Capita: **${predicted_gdp:,.2f}**')

# Visualization section
st.subheader('Historical Trend vs Prediction')

# Load historical data
@st.cache_data
def load_historical_data():
    df = pd.read_csv('Interest Rate and Gdp.csv', parse_dates=['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)
    return df.resample('M').last().ffill()

df = load_historical_data()

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot historical interest rates
ax1.plot(df['Average'], label='Historical Interest Rates', color='tab:blue')
ax1.axhline(y=interest_rate, color='r', linestyle='--', label='Selected Interest Rate')
ax1.set_ylabel('Interest Rate (%)')
ax1.set_title('Interest Rate Trend')
ax1.legend()
ax1.grid(True)

# Plot historical GDP
ax2.plot(df['Gdp Per Capita'], label='Historical GDP Per Capita', color='tab:green')
ax2.axhline(y=predicted_gdp, color='r', linestyle='--', label='Predicted GDP')
ax2.set_ylabel('GDP Per Capita ($)')
ax2.set_title('GDP Per Capita Trend')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
st.pyplot(fig)

# Model information
st.sidebar.subheader('Model Information')
st.sidebar.info("""
- Model: LSTM Neural Network
- Features: 12-month sequence of interest rates and GDP values
- Target: Next month's GDP Per Capita
""")

# How to use
st.sidebar.subheader('How to Use')
st.sidebar.info("""
1. Adjust the interest rate slider
2. View the predicted GDP value
3. Examine the historical trends
""")