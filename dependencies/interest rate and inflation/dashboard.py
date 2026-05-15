import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError

# Solution 1: Register the metric properly when loading
@st.cache_resource
def load_lstm_model():
    try:
        # Option 1: Map 'mse' to the actual MSE function
        custom_objects = {'mse': MeanSquaredError(name='mse')}
        
        # Option 2: Alternatively use the string name
        # custom_objects = {'mse': 'mse'}
        
        model = load_model(r'C:\Users\Admin\Documents\Mihir Codes\Sem 4\AI Assignments\AI PBL\dependencies\interest rate and inflation\optimized_inflation_model.pkl', 
                         custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# Load the scaler that was used during training
@st.cache_resource
def load_scaler():
    try:
        # If you saved the scaler during training:
        # from joblib import dump
        # dump(scaler, 'scaler.joblib')
        return joblib.load(r'C:\Users\Admin\Documents\Mihir Codes\Sem 4\AI Assignments\AI PBL\dependencies\interest rate and inflation\scaler.pkl')
    except:
        # Fallback - create new scaler (won't work properly with real data)
        st.warning("Using new scaler - predictions may be inaccurate")
        return MinMaxScaler(feature_range=(0, 1))

def prepare_input(interest_rate, scaler, look_back=10):
    """Create input sequence for the model"""
    dummy_data = np.zeros((look_back, 4))  # 4 features like in training
    dummy_data[:, 0] = interest_rate  # Interest Rate
    dummy_data[:, 2] = interest_rate  # Interest_MA
    scaled_data = scaler.transform(dummy_data)
    return scaled_data[:, :2].reshape(1, look_back, 2)  # Only first 2 features

def main():
    st.title("Economic Growth Rate Predictor")
    st.markdown("""
    This app predicts economic growth rate based on interest rate inputs.
    Adjust the slider below to see predictions.
    """)

    # Load model and scaler
    model = load_lstm_model()
    scaler = load_scaler()

    # UI Controls
    interest_rate = st.slider(
        "Interest Rate (%)",
        min_value=0.1,
        max_value=20.0,
        value=5.0,
        step=0.1,
        key='rate_slider'
    )

    # Make prediction
    if st.button("Predict") or True:  # Auto-update
        try:
            # Prepare input and predict
            sequence = prepare_input(interest_rate, scaler)
            prediction = model.predict(sequence, verbose=0)
            
            # Inverse transform the prediction
            dummy = np.zeros((1, 4))
            dummy[0, 1] = prediction[0][0]  # Growth rate is at position 1
            predicted_growth = scaler.inverse_transform(dummy)[0, 1]

            # Display results
            st.subheader("Prediction Result")
            col1, col2 = st.columns(2)
            col1.metric("Input Interest Rate", f"{interest_rate}%")
            col2.metric("Predicted Growth Rate", f"{predicted_growth:.2f}%")

            # Visualization
            st.subheader("Interest Rate vs Economic Growth Relationship")
            rates = np.linspace(0.5, 15.0, 50)
            growth_rates = []
            
            for rate in rates:
                seq = prepare_input(rate, scaler)
                pred = model.predict(seq, verbose=0)
                dummy = np.zeros((1, 4))
                dummy[0, 1] = pred[0][0]
                growth_rates.append(scaler.inverse_transform(dummy)[0, 1])
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(rates, growth_rates, 'b-', label='Predicted Relationship')
            ax.scatter([interest_rate], [predicted_growth], 
                      c='red', s=100, label='Current Prediction')
            ax.set_xlabel("Interest Rate (%)")
            ax.set_ylabel("Economic Growth Rate (%)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()