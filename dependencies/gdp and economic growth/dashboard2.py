import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load('economic_growth_model.pkl')
    return model

model = load_model()

# App title and intro
st.title('Economic Growth Prediction Dashboard')
st.write('This app predicts economic growth based on **GDP Per Capita**, assuming average values for other features.')

# Sidebar input
st.sidebar.header('Input GDP Per Capita')
gdp_per_capita = st.sidebar.slider('GDP Per Capita(in dollor $)', 0.0, 10000.0, 5000.0)

# Create a DataFrame with required features (only GDP Per Capita is user input)
input_data = pd.DataFrame({
    'Gdp Per Capita': [gdp_per_capita],
    'GDP_lag1': [50000.0],
    'Growth_lag1': [2.0],
    'GDP_lag2': [50000.0],
    'Growth_lag2': [2.0],
    'GDP_lag3': [50000.0],
    'Growth_lag3': [2.0],
    'GDP_rolling_mean': [50000.0],
    'GDP_rolling_std': [10000.0]
})

# Display the input
st.subheader('Input Parameter')
st.write(input_data[['Gdp Per Capita']])

# Predict economic growth
if st.button('Predict Economic Growth'):
    prediction = model.predict(input_data)

    st.subheader('Prediction Result')
    st.write(f'**Predicted Economic Growth Rate:** `{prediction[0]:.2f}%`')

    # Interpretation
    st.subheader('Interpretation')
    if prediction[0] > 3:
        st.success('The economy is predicted to grow at a healthy rate.')
    elif prediction[0] > 0:
        st.warning('The economy is predicted to grow at a modest rate.')
    else:
        st.error('The economy is predicted to contract.')

# # Dependency graph
# st.subheader('Dependency: GDP Per Capita vs Economic Growth')
# gdp_range = np.linspace(0, 100000, 200)
# gdp_df = pd.DataFrame({
#     'Gdp Per Capita': gdp_range,
#     'GDP_lag1': 50000.0,
#     'Growth_lag1': 2.0,
#     'GDP_lag2': 50000.0,
#     'Growth_lag2': 2.0,
#     'GDP_lag3': 50000.0,
#     'Growth_lag3': 2.0,
#     'GDP_rolling_mean': 50000.0,
#     'GDP_rolling_std': 10000.0
# })
# growth_preds = model.predict(gdp_df)

# fig, ax = plt.subplots()
# ax.plot(gdp_range, growth_preds, color='blue')
# ax.set_xlabel('GDP Per Capita')
# ax.set_ylabel('Predicted Growth Rate (%)')
# ax.set_title('Dependency Plot')
# st.pyplot(fig)

# Actual graph

# Plot actual data from the dataset
st.subheader('Actual Data: GDP Per Capita vs Economic Growth')

# Load dataset
data = pd.read_csv('gdp and economic growth.csv')

# Sort by GDP Per Capita for a smoother line plot
data_sorted = data.sort_values(by='Gdp Per Capita')

# Line chart
fig2, ax2 = plt.subplots()
ax2.plot(data_sorted['Gdp Per Capita'], data_sorted['Economic Growth Rate'], color='red', marker='x')
ax2.set_xlabel('GDP Per Capita')
ax2.set_ylabel('Economic Growth Rate (%)')
ax2.set_title('Actual GDP vs Economic Growth')
st.pyplot(fig2)


# Footer
st.markdown("---")
st.markdown("""
**Note**: This demo uses only GDP Per Capita for user input. Other variables are set to default values.
For more precise predictions, provide full economic data.
""")
