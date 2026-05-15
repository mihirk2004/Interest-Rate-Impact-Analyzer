import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib

# Set page config
st.set_page_config(page_title="Economic Growth Predictor", layout="wide")

# Title and description
st.title("Economic Growth Rate Prediction using LSTM")
st.markdown("""
This app predicts economic growth rate based on interest rate using a pre-trained LSTM model.
- Adjust the interest rate using the slider to see predicted growth rate
- View historical data visualization
""")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

# Load data function
@st.cache_data
def load_data():
    df = pd.read_csv('Interest Rate and Economic Growth Rate.csv', parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    return df

# Prediction function
def predict_growth(model, scaler, interest_rate, look_back=10):
    # Create a sequence with the given interest rate
    # We need to create a realistic sequence for prediction
    # Here we'll use the average growth rate from training data as a baseline
    
    # Get scaler parameters
    min_interest = scaler.data_min_[0]
    max_interest = scaler.data_max_[0]
    min_growth = scaler.data_min_[1]
    max_growth = scaler.data_max_[1]
    
    # Create a sequence with constant interest rate and average growth
    avg_growth = (min_growth + max_growth) / 2
    sequence = np.zeros((look_back, 2))
    sequence[:, 0] = interest_rate  # Set interest rate
    sequence[:, 1] = avg_growth    # Set baseline growth rate
    
    # Scale the sequence
    dummy = np.zeros((look_back, 4))
    dummy[:, 0] = sequence[:, 0]
    dummy[:, 1] = sequence[:, 1]
    # Calculate moving averages for the sequence
    dummy[:, 2] = np.mean(sequence[:, 0])
    dummy[:, 3] = np.mean(sequence[:, 1])
    
    scaled_sequence = scaler.transform(dummy)[:, :2].reshape(1, look_back, 2)
    
    # Make prediction
    scaled_pred = model.predict(scaled_sequence, verbose=0)
    
    # Inverse scale the prediction
    dummy = np.zeros((1, 4))
    dummy[:, 1] = scaled_pred
    pred = scaler.inverse_transform(dummy)[0, 1]
    
    return pred

# Main app
def main():
    df = load_data()
    
    # Check if model exists
    try:
        model = load_model('improved_economic_growth_lstm.keras')
        scaler = joblib.load('scaler.pkl')
        trained = True
    except:
        st.error("Model files not found. Please ensure 'improved_economic_growth_lstm.keras' and 'scaler.pkl' are in the correct directory.")
        trained = False
    
    if trained:
        # Prediction section
        st.sidebar.subheader("Predict Economic Growth")
        interest_rate = st.sidebar.slider(
            "Select Interest Rate", 
            min_value=float(df['Interest Rate'].min()), 
            max_value=float(df['Interest Rate'].max()),
            value=float(df['Interest Rate'].mean()),
            step=0.1
        )
        
        if st.sidebar.button('Predict Growth Rate'):
            growth_rate = predict_growth(model, scaler, interest_rate)
            st.sidebar.success(f"Predicted Economic Growth Rate: {growth_rate:.2f}%")
        
        # Display historical data visualization
        st.subheader("Historical Interest Rate and Economic Growth Rate")
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:red'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Interest Rate', color=color)
        ax1.plot(df['Date'], df['Interest Rate'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Economic Growth Rate', color=color)
        ax2.plot(df['Date'], df['Economic Growth Rate'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Historical Interest Rate and Economic Growth Rate')
        st.pyplot(fig)
        
        # Show raw data
        if st.checkbox('Show Raw Data'):
            st.subheader('Raw Data')
            st.write(df)

if __name__ == '__main__':
    main()