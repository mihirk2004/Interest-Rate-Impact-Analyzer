import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import pickle
from tensorflow.keras.metrics import MeanSquaredError

# Load feature names
@st.cache_resource
def load_feature_names():
    try:
        with open(r'C:\Users\Admin\Documents\Mihir Codes\Sem 4\AI Assignments\AI PBL\dependencies\interest rate and inflation\feature_names.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load feature names: {str(e)}")
        return None

# Load the XGBoost model
@st.cache_resource
def load_xgb_model():
    try:
        model = joblib.load(r'C:\Users\Admin\Documents\Mihir Codes\Sem 4\AI Assignments\AI PBL\dependencies\interest rate and inflation\optimized_inflation_model.pkl')
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# Load the scaler
@st.cache_resource
def load_scaler():
    try:
        return joblib.load(r'C:\Users\Admin\Documents\Mihir Codes\Sem 4\AI Assignments\AI PBL\dependencies\interest rate and inflation\scaler.pkl')
    except:
        st.warning("Using new scaler - predictions may be inaccurate")
        return MinMaxScaler(feature_range=(0, 1))

def prepare_input(interest_rate, inflation_rate=2.0):
    """Create input features for the XGBoost model"""
    data = {
        'Average': interest_rate,
        'Interest_Lag_1': interest_rate,
        'Inflation_Lag_1': inflation_rate,
        'Interest_Lag_2': interest_rate,
        'Inflation_Lag_2': inflation_rate,
        'Interest_Lag_3': interest_rate,
        'Inflation_Lag_3': inflation_rate,
        'Interest_Lag_4': interest_rate,
        'Inflation_Lag_4': inflation_rate,
        'Interest_Lag_5': interest_rate,
        'Inflation_Lag_5': inflation_rate,
        'Interest_Lag_6': interest_rate,
        'Inflation_Lag_6': inflation_rate,
        'Interest_Lag_7': interest_rate,
        'Inflation_Lag_7': inflation_rate,
        'Interest_Lag_12': interest_rate,
        'Inflation_Lag_12': inflation_rate,
        'Interest_MA_3': interest_rate,
        'Interest_MA_7': interest_rate,
        'Inflation_MA_3': inflation_rate,
        'Month': 1,  # Default month (January)
        'Quarter': 1,  # Default quarter (Q1)
        'Year': 2023,  # Default year
        'Interest_Diff': 0  # Default difference
    }
    
    # Convert to DataFrame and return as numpy array
    df = pd.DataFrame([data])
    return df.values

def main():
    st.title("Economic Growth Rate Predictor")
    st.markdown("""
    This app predicts economic growth rate based on interest rate and inflation inputs.
    Adjust the sliders below to see predictions.
    """)

    # Load model, scaler, and feature names
    model = load_xgb_model()
    scaler = load_scaler()
    feature_names = load_feature_names()
    
    # if feature_names is not None:
    #     st.sidebar.subheader("Feature Information")
    #     st.sidebar.write("Model uses these features:")
    #     st.sidebar.write(feature_names)

    # UI Controls
    col1, col2 = st.columns(2)
    with col1:
        interest_rate = st.slider(
            "Interest Rate (%)",
            min_value=0.1,
            max_value=20.0,
            value=5.0,
            step=0.1
        )
    with col2:
        inflation_rate = st.slider(
            "Inflation Rate (%)",
            min_value=0.1,
            max_value=15.0,
            value=2.0,
            step=0.1
        )

    # Make prediction
    if st.button("Predict") or True:  # Auto-update
        try:
            # Prepare input features
            features = prepare_input(interest_rate, inflation_rate)
            
            # Scale the input data
            scaled_features = scaler.transform(features)
            
            # Make prediction (removed verbose parameter)
            prediction = model.predict(scaled_features)
            
            # Inverse transform the prediction
            dummy = np.zeros((1, 24))
            dummy[0, 1] = prediction[0]  # Assuming growth rate is at position 1
            predicted_growth = scaler.inverse_transform(dummy)[0, 1]

            # Display results
            st.subheader("Prediction Result")
            col1, col2, col3 = st.columns(3)
            col1.metric("Input Interest Rate", f"{interest_rate}%")
            col2.metric("Input Inflation Rate", f"{inflation_rate}%")
            col3.metric("Predicted Growth Rate", f"{predicted_growth:.2f}%")

            # Visualization
            st.subheader("Interest Rate vs Economic Growth Relationship")
            rates = np.linspace(0.5, 15.0, 50)
            growth_rates = []
            
            for rate in rates:
                features = prepare_input(rate, inflation_rate)
                scaled_feat = scaler.transform(features)
                pred = model.predict(scaled_feat)
                dummy = np.zeros((1, 24))
                dummy[0, 1] = pred[0]
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