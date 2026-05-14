import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# Load Model
# =========================

model = pickle.load(open('model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title='House Price Prediction',
    page_icon='🏠',
    layout='centered'
)

# =========================
# Custom CSS
# =========================

st.markdown("""
<style>

body {
    background-color: #0f172a;
}

.main {
    background: linear-gradient(to bottom right, #0f172a, #111827);
    color: white;
}

h1, h2, h3, h4 {
    color: white;
}

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    font-size: 18px;
    font-weight: 600;
    transition: 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0px 0px 15px rgba(124, 58, 237, 0.5);
}

.stNumberInput, .stSlider {
    background-color: transparent;
}

.result-box {
    background: linear-gradient(135deg, #1e293b, #111827);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
    margin-top: 25px;
}

.result-price {
    font-size: 42px;
    font-weight: bold;
    color: #22c55e;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 18px;
    margin-bottom: 30px;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================

st.markdown("""
<h1 style='text-align: center;'>🏠 House Price Prediction Dashboard</h1>
<p class='subtitle'>
Predict house prices using Machine Learning
</p>
""", unsafe_allow_html=True)

# =========================
# User Inputs
# =========================

col1, col2 = st.columns(2)

with col1:
    OverallQual = st.slider('Overall Quality', 1, 10, 5)
    GarageCars = st.slider('Garage Capacity', 0, 5, 2)
    FullBath = st.slider('Full Bathrooms', 0, 5, 2)
    Fireplaces = st.slider('Number of Fireplaces', 0, 5, 1)

with col2:
    GrLivArea = st.number_input('Ground Living Area', 500, 6000, 1500)
    TotalBsmtSF = st.number_input('Basement Area', 0, 5000, 800)
    YearBuilt = st.slider('Year Built', 1900, 2025, 2000)
    LotArea = st.number_input('Lot Area', 1000, 100000, 8000)

# =========================
# Create Input DataFrame
# =========================

input_dict = {
    'OverallQual': OverallQual,
    'GrLivArea': GrLivArea,
    'GarageCars': GarageCars,
    'TotalBsmtSF': TotalBsmtSF,
    'FullBath': FullBath,
    'YearBuilt': YearBuilt,
    'LotArea': LotArea,
    'Fireplaces': Fireplaces,
    '1stFlrSF': GrLivArea,
    '2ndFlrSF': 0,
    'HalfBath': 0,
    'BsmtFullBath': 1,
    'BsmtHalfBath': 0,
    'YearRemodAdd': YearBuilt,
    'YrSold': 2010,
    'OpenPorchSF': 0,
    'EnclosedPorch': 0,
    '3SsnPorch': 0,
    'ScreenPorch': 0,
    'GarageArea': GarageCars * 200,
}

input_df = pd.DataFrame([input_dict])

# =========================
# Feature Engineering
# =========================

input_df['TotalSF'] = (
    input_df['TotalBsmtSF'] +
    input_df['1stFlrSF'] +
    input_df['2ndFlrSF']
)

input_df['TotalBath'] = (
    input_df['FullBath'] +
    (0.5 * input_df['HalfBath']) +
    input_df['BsmtFullBath'] +
    (0.5 * input_df['BsmtHalfBath'])
)

input_df['HouseAge'] = (
    input_df['YrSold'] -
    input_df['YearBuilt']
)

input_df['RemodelAge'] = (
    input_df['YrSold'] -
    input_df['YearRemodAdd']
)

input_df['TotalPorchSF'] = (
    input_df['OpenPorchSF'] +
    input_df['EnclosedPorch'] +
    input_df['3SsnPorch'] +
    input_df['ScreenPorch']
)

input_df['HasGarage'] = 1
input_df['HasBsmt'] = 1
input_df['HasFireplace'] = 1 if Fireplaces > 0 else 0

# =========================
# Match Training Columns
# =========================

final_df = pd.DataFrame(columns=columns)

for col in final_df.columns:
    if col in input_df.columns:
        final_df.loc[0, col] = input_df.loc[0, col]
    else:
        final_df.loc[0, col] = 0

final_df = final_df.fillna(0)

# =========================
# Prediction
# =========================

st.write("")

if st.button('🚀 Predict House Price'):

    prediction_log = model.predict(final_df)

    prediction = np.expm1(prediction_log)

    st.markdown(f"""
    <div class="result-box">
        <h2>Estimated House Price</h2>
        <div class="result-price">
            ${prediction[0]:,.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)