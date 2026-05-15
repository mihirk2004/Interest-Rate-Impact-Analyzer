import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler

# Load the model
@st.cache_resource
def load_model():
    # with open('economic_growth_model.pkl', 'rb') as f:
    #     model = pickle.load(f)
    model = joblib.load('economic_growth_model.pkl')
    return model

model = load_model()

# Create the Streamlit app
st.title('Economic Growth Prediction Dashboard')
st.write('This app predicts economic growth based on GDP-related parameters')

# Sidebar inputs
st.sidebar.header('Input GDP Parameters')

# Create input widgets for all features
gdp_per_capita = st.sidebar.slider('GDP Per Capita', 0.0, 100000.0, 50000.0)
gdp_lag1 = st.sidebar.slider('GDP Lag 1', 0.0, 100000.0, 50000.0)
growth_lag1 = st.sidebar.slider('Growth Lag 1', -10.0, 10.0, 2.0)
gdp_lag2 = st.sidebar.slider('GDP Lag 2', 0.0, 100000.0, 50000.0)
growth_lag2 = st.sidebar.slider('Growth Lag 2', -10.0, 10.0, 2.0)
gdp_lag3 = st.sidebar.slider('GDP Lag 3', 0.0, 100000.0, 50000.0)
growth_lag3 = st.sidebar.slider('Growth Lag 3', -10.0, 10.0, 2.0)
gdp_rolling_mean = st.sidebar.slider('GDP Rolling Mean', 0.0, 100000.0, 50000.0)
gdp_rolling_std = st.sidebar.slider('GDP Rolling Std', 0.0, 50000.0, 10000.0)

# Create a DataFrame with the input data
input_data = pd.DataFrame({
    'Gdp Per Capita': [gdp_per_capita],
    'GDP_lag1': [gdp_lag1],
    'Growth_lag1': [growth_lag1],
    'GDP_lag2': [gdp_lag2],
    'Growth_lag2': [growth_lag2],
    'GDP_lag3': [gdp_lag3],
    'Growth_lag3': [growth_lag3],
    'GDP_rolling_mean': [gdp_rolling_mean],
    'GDP_rolling_std': [gdp_rolling_std]
})

# Display the input data
st.subheader('Input Parameters')
st.write(input_data)

# Make prediction
if st.button('Predict Economic Growth'):
    prediction = model.predict(input_data)
    
    st.subheader('Prediction Result')
    st.write(f'Predicted Economic Growth Rate: {prediction[0]:.2f}%')
    
    # Interpretation
    st.subheader('Interpretation')
    if prediction[0] > 3:
        st.success('The economy is predicted to grow at a healthy rate.')
    elif prediction[0] > 0:
        st.warning('The economy is predicted to grow at a modest rate.')
    else:
        st.error('The economy is predicted to contract.')

# Model information
st.sidebar.header('Model Information')
st.sidebar.write('Model Type: Gradient Boosting Regressor')
st.sidebar.write('Number of Estimators: 300')
st.sidebar.write('Learning Rate: 0.1')
st.sidebar.write('Max Depth: 3')

# Add some visualizations
st.subheader('Feature Importance')
try:
    # Get feature importances from the model
    if hasattr(model, 'named_steps'):
        gb_model = model.named_steps['regressor']
    else:
        gb_model = model
    
    if hasattr(gb_model, 'feature_importances_'):
        feature_importance = gb_model.feature_importances_
        features = input_data.columns
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(importance_df.set_index('Feature'))
    else:
        st.write("Feature importance not available for this model.")
except Exception as e:
    st.write(f"Could not display feature importance: {str(e)}")

# Add some explanation
st.subheader('How to Use This Dashboard')
st.write("""
1. Adjust the sliders in the sidebar to input your GDP parameters
2. Click the 'Predict Economic Growth' button
3. View the predicted growth rate and interpretation
4. Examine the feature importance to understand which factors most influence the prediction
""")

# Add a footer
st.markdown("---")
st.markdown("""
**Note**: This is a demonstration dashboard using a pre-trained model. 
For accurate economic predictions, consult with professional economists.
""")