import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scale.pkl')

st.write("<h1>House Price Prediction App</h1>", unsafe_allow_html=True)

# Collect user input for all the features used during training
Street = st.number_input("Street", min_value=0, max_value=1, value=0)
LotShape = st.number_input("Lot Shape", min_value=0, max_value=3, value=0)
Utilities = st.number_input("Utilities", min_value=0, max_value=1, value=0)
LandSlope = st.number_input("Land Slope", min_value=0, max_value=2, value=0)
Neighborhood = st.number_input("Neighborhood", value=0)
HouseStyle = st.number_input("House Style", value=0)
RoofMatl = st.number_input("Roof Material", value=0)
Exterior2nd = st.number_input("Exterior 2nd", value=0)
ExterCond = st.number_input("Exterior Condition", value=0)
Foundation = st.number_input("Foundation", value=0)
HeatingQC = st.number_input("Heating Quality and Condition", value=0)
CentralAir = st.number_input("Central Air", min_value=0, max_value=1, value=0)
KitchenQual = st.number_input("Kitchen Quality", value=0)
Functional = st.number_input("Functional", value=0)
GarageType = st.number_input("Garage Type", value=0)
GarageCond = st.number_input("Garage Condition", value=0)
PoolQC = st.number_input("Pool Quality", value=0)
Fence = st.number_input("Fence", value=0)
MiscFeature = st.number_input("Miscellaneous Feature", value=0)
SaleType = st.number_input("Sale Type", value=0)
SaleCondition = st.number_input("Sale Condition", value=0)

# Create a DataFrame with the input values
input_features = pd.DataFrame({
    'Street': [Street],
    'LotShape': [LotShape],
    'Utilities': [Utilities],
    'LandSlope': [LandSlope],
    'Neighborhood': [Neighborhood],
    'HouseStyle': [HouseStyle],
    'RoofMatl': [RoofMatl],
    'Exterior2nd': [Exterior2nd],
    'ExterCond': [ExterCond],
    'Foundation': [Foundation],
    'HeatingQC': [HeatingQC],
    'CentralAir': [CentralAir],
    'KitchenQual': [KitchenQual],
    'Functional': [Functional],
    'GarageType': [GarageType],
    'GarageCond': [GarageCond],
    'PoolQC': [PoolQC],
    'Fence': [Fence],
    'MiscFeature': [MiscFeature],
    'SaleType': [SaleType],
    'SaleCondition': [SaleCondition]
})

# Scale the input features using the same scaler used during model training
input_features_scaled = scaler.transform(input_features)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_features_scaled)
    st.write(f"Predicted House Price: ${prediction[0]:,.2f}")