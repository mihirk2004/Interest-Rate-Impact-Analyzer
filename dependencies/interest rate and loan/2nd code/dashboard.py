import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
import joblib
from tensorflow.keras.models import load_model

# Load the trained model and preprocessing objects
model = load_model('final_loan_model.h5')
poly = joblib.load('poly_transformer.pkl')
scaler = joblib.load('robust_scaler.pkl')

# Streamlit app
st.title("Loan Rate Predictor")
st.markdown("""
This app predicts the loan rate based on the interest rate using a trained neural network model.
""")

# Sidebar slider for interest rate
st.sidebar.header("Input Parameters")
interest_rate = st.sidebar.slider(
    "Average Interest Rate (%)", 
    min_value=0.0, 
    max_value=20.0, 
    value=5.0, 
    step=0.1
)

# Create a range of values for visualization
interest_range = np.linspace(0, 20, 100).reshape(-1, 1)

# Function to make predictions
def predict_loan_rate(interest_rates):
    # Transform the input
    interest_poly = poly.transform(interest_rates)
    interest_scaled = scaler.transform(interest_poly)
    # Make prediction
    predictions = model.predict(interest_scaled).flatten()
    return predictions

# Predict for the slider value
current_prediction = predict_loan_rate(np.array([[interest_rate]]))

# Predict for the range of values
predictions = predict_loan_rate(interest_range)

# Display the prediction
st.subheader(f"Predicted Loan Rate: {current_prediction[0]:.2f}%")
st.markdown(f"For an interest rate of **{interest_rate}%**, the predicted loan rate is **{current_prediction[0]:.2f}%**")

# Create the visualization
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(interest_range, predictions, 'b-', linewidth=2, label='Predicted Loan Rate')
ax.scatter([interest_rate], current_prediction, c='red', s=100, label='Current Prediction')
ax.set_xlabel('Average Interest Rate (%)')
ax.set_ylabel('Predicted Loan Rate (%)')
ax.set_title('Interest Rate vs Predicted Loan Rate')
ax.grid(True)
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

# Show feature importance (optional)
if st.checkbox("Show Feature Importance"):
    st.subheader("Feature Importance")
    weights = model.layers[0].get_weights()[0]
    poly_features = poly.get_feature_names_out(['Average'])
    feature_importance = pd.DataFrame({
        'Feature': poly_features,
        'Weight': weights[:, 0]
    }).sort_values('Weight', key=abs, ascending=False)
    
    st.dataframe(feature_importance)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.barh(feature_importance['Feature'], np.abs(feature_importance['Weight']))
    ax2.set_title('Feature Importance (Absolute Weights)')
    ax2.set_xlabel('Absolute Weight Value')
    st.pyplot(fig2)

# Add some model info
st.sidebar.markdown("""
### Model Information
- Model: Neural Network with 2 hidden layers
- Preprocessing: Polynomial Features (degree=3) + Robust Scaling
- Training MAE: {your_mae_value:.2f}%
- Training R²: {your_r2_value:.2f}
""".format(your_mae_value=0.42, your_r2_value=0.95))  # Replace with your actual values