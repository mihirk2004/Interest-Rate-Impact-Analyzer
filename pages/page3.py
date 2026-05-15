import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Set page config
st.set_page_config(page_title="Loan Rate Predictor", page_icon="📊")

# Title and description
st.title("📊 Loan Rate Prediction Dashboard")
st.write("Predict future loan rates based on current and historical inflation rates.")

# Load model, scaler, and feature order
try:
    model = joblib.load(r'C:\Users\Admin\Documents\Mihir Codes\Sem 4\AI Assignments\AI PBL\dependencies\inflation and loan\inflation_loan_xgboost_model.pkl')
    scaler = joblib.load(r'C:\Users\Admin\Documents\Mihir Codes\Sem 4\AI Assignments\AI PBL\dependencies\inflation and loan\inflation_loan_scaler.pkl')
    feature_order = joblib.load(r'C:\Users\Admin\Documents\Mihir Codes\Sem 4\AI Assignments\AI PBL\dependencies\inflation and loan\inflation_loan_feature_order.pkl')
    use_real_model = True
except Exception as e:
    st.warning(f"Couldn't load the trained model. Error: {str(e)}")
    st.warning("Using simplified calculation instead.")
    use_real_model = False

# Create a function to generate features with correct naming
def generate_features(current_inflation):
    """Generate lag features with correct naming convention"""
    features = {
        'Inflation Rate': current_inflation,
        'Inflation_Rate_lag_1': current_inflation * 0.95,
        'Inflation_Rate_lag_3': current_inflation * 0.90,
        'Inflation_Rate_lag_6': current_inflation * 0.85,
        'Inflation_Rate_lag_12': current_inflation * 0.80,
        'Loan_Rate_lag_1': current_inflation * 0.85,
        'Loan_Rate_lag_3': current_inflation * 0.80,
        'Loan_Rate_lag_6': current_inflation * 0.75,
        'Loan_Rate_lag_12': current_inflation * 0.70
    }
    return features

# User input
inflation_rate = st.slider(
    "Current Inflation Rate (%)", 
    min_value=0.0, 
    max_value=20.0, 
    value=5.0, 
    step=0.1
)

# Generate features
input_features = generate_features(inflation_rate)
input_df = pd.DataFrame([input_features])

# Ensure correct feature order and names
if use_real_model:
    try:
        # Reorder columns to match training data exactly
        input_df = input_df[feature_order]
        
        # Verify all expected features are present
        missing_features = set(feature_order) - set(input_df.columns)
        if missing_features:
            st.error(f"Missing features: {missing_features}")
    except Exception as e:
        st.error(f"Feature ordering failed: {str(e)}")
        use_real_model = False

# Predict button
if st.button("Predict Loan Rate"):
    if use_real_model:
        try:
            # Scale the input data
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            use_real_model = False
    else:
        # Fallback calculation
        prediction = inflation_rate * 1.2
    
    # Display results
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Inflation Rate", f"{inflation_rate:.1f}%")
    with col2:
        st.metric("Predicted Loan Rate", f"{prediction:.1f}%")


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data = pd.read_csv(r'C:\Users\Admin\Documents\Mihir Codes\Sem 4\AI Assignments\AI PBL\dependencies\inflation and loan\inflation and loan.csv')
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data.set_index('Date', inplace=True)
# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 7))

# Plot data
ax.plot(data.index, data['Inflation Rate'], label='Inflation Rate', color='#E63946', linewidth=2)
ax.plot(data.index, data['Loan Rate'], label='Loan Rate', color='#1D3557', linewidth=2)

# Formatting
ax.set_title('Inflation and Loan Rate Trends (2000-2025)', pad=20, fontsize=14)
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

# Show the generated features (for transparency)
with st.expander("Show generated input features"):
    st.write("These features were used for the prediction:")
    st.dataframe(input_df.style.format("{:.2f}"))