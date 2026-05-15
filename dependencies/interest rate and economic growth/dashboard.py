import streamlit as st
import numpy as np
import tensorflow as tf

# Load the LSTM model
@st.cache_resource
def load_lstm_model():
    model = tf.keras.models.load_model('improved_economic_growth_lstm.h5')
    return model

model = load_lstm_model()

# Streamlit UI
st.title("Economic Feature Prediction Based on Interest Rate")
st.markdown("""
This dashboard uses an LSTM model to predict an economic feature based on the input **Interest Rate**.
""")

# Interest Rate Slider
interest_rate = st.slider(
    "Select Interest Rate (%)",
    min_value=0.0,
    max_value=15.0,
    value=2.5,
    step=0.1
)

# Prepare input for LSTM (reshape to 3D: [samples, timesteps, features])
# Let's assume we pass one timestep with one feature
input_data = np.array([[[interest_rate]]])  # Shape: (1, 1, 1)

# Make prediction
prediction = model.predict(input_data)
predicted_value = prediction[0][0]  # Assuming single output

# Display prediction
st.success(f"Predicted Economic Output: {predicted_value:.2f}")
