import streamlit as st
import joblib

import numpy as np
import pandas as pd
import pickle
from sklearn_functions import RandomForestClassifier


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'RandomForestClassifier':
            return RandomForestClassifier
        return super().find_class(module, name)


# Load model
with open("cardio_model.pkl", "rb") as f:
    bundle = CustomUnpickler(f).load()


model = bundle['model']
feature_order = bundle['columns']
scaler_hi = bundle['scaler_hi']
scaler_lo = bundle['scaler_lo']

st.markdown("""
    <style>
    /* Main background - Pure Black */
    .stApp {
        background: #000000;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: -apple-system, sans-serif;
    }
    
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Input labels */
    .stSelectbox label, .stNumberInput label {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    /* Number input styling */
    .stNumberInput input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }
    
    .stNumberInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 1px #667eea !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* Form container */
    .stForm {
        background: rgba(20, 20, 30, 0.5);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Submit Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
    }
    
    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 2rem 0;
    }
    
    .high-risk {
        background: linear-gradient(135deg, #ff0844 0%, #ff5370 100%);
        border: 1px solid rgba(255, 8, 68, 0.5);
    }
    
    .low-risk {
        background: linear-gradient(135deg, #00f260 0%, #0575e6 100%);
        border: 1px solid rgba(0, 242, 96, 0.5);
    }
    
    /* Metric boxes */
    .metric-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .metric-label {
        color: #999;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Section headers */
    .section-header {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
    }
    
    /* Info box */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 3px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        color: #ccc;
    }
    
    /* Radio buttons */
    .stRadio > div {
        flex-direction: row;
        gap: 0.75rem;
        justify-content: center;
    }
    
    .stRadio > div > label {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        color: #888;
        font-weight: 500;
    }
    
    .stRadio > div > label:hover {
        border-color: rgba(255, 255, 255, 0.3);
        background: rgba(255, 255, 255, 0.08);
    }
    
    .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-color: #667eea;
        color: #ffffff;
    }
    
    /* Checkbox styling */
    .stCheckbox {
        padding: 0.75rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stCheckbox > label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    .stCheckbox input[type="checkbox"] {
        width: 20px;
        height: 20px;
        accent-color: #667eea;
    }
    
    /* Subsection headers */
    h4 {
        color: #fff !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin: 1.5rem 0 1rem 0 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Tab label */
    .tab-label {
        color: #ffffff;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.75rem;
        display: block;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config(page_title="Cardio Risk Predictor", page_icon="❤️", layout="centered")

# Header
st.markdown('<h1 class="main-title">Cardiovascular Disease Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered heart health assessment</p>', unsafe_allow_html=True)

# Info box
st.markdown("""
    <div class="info-box">
        <strong>How it works:</strong> Enter your health metrics below to receive a personalized cardiovascular risk assessment 
        powered by machine learning algorithms trained on thousands of patient records.
    </div>
""", unsafe_allow_html=True)

# --- Input form ---
st.markdown('<div class="section-header">Patient Information</div>', unsafe_allow_html=True)

with st.form("input_form"):
    # Personal Information
    st.markdown("#### Demographics")
    col1, col2 = st.columns(2)
    age_years = col1.number_input("Age (years)", 20, 120, 50)
    gender = col2.selectbox("Gender", ["Male", "Female"])
    
    st.markdown("#### Physical Measurements")
    col3, col4 = st.columns(2)
    height = col3.number_input("Height (cm)", 120, 220, 170)
    weight = col4.number_input("Weight (kg)", 30, 200, 70)
    
    st.markdown("#### Vital Signs")
    col5, col6 = st.columns(2)
    ap_hi = col5.number_input("Systolic BP (mmHg)", 90, 250, 120)
    ap_lo = col6.number_input("Diastolic BP (mmHg)", 40, 180, 80)
    
    st.markdown("#### Clinical Markers")
    
    # Cholesterol with radio buttons
    st.markdown('<span class="tab-label">Cholesterol Level</span>', unsafe_allow_html=True)
    cholesterol = st.radio(
        "cholesterol_radio",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "High"}[x],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Glucose with radio buttons
    st.markdown('<span class="tab-label">Glucose Level</span>', unsafe_allow_html=True)
    gluc = st.radio(
        "glucose_radio",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "High"}[x],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("#### Lifestyle Factors")
    
    # Toggle switches for lifestyle factors
    col7, col8, col9 = st.columns(3)
    
    with col7:
        smoke = st.checkbox("Smoking", value=False, key="smoke_toggle")
        smoke = 1 if smoke else 0
    
    with col8:
        alco = st.checkbox("Alcohol", value=False, key="alco_toggle")
        alco = 1 if alco else 0
    
    with col9:
        active = st.checkbox("Physical Activity", value=False, key="active_toggle")
        active = 1 if active else 0
    
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Analyze Risk")

# --- Prediction logic ---
if submitted:
    # Load scalers (now available in bundle)
    scaler_hi = bundle['scaler_hi']
    scaler_lo = bundle['scaler_lo']

    # Fix gender: 1=Female, 2=Male
    gender_val = 2 if gender == "Male" else 1

    # Base input DataFrame
    features_df = pd.DataFrame([{
        'gender': gender_val, 'height': height, 'weight': weight,
        'ap_hi': ap_hi, 'ap_lo': ap_lo, 'cholesterol': cholesterol,
        'gluc': gluc, 'smoke': smoke, 'alco': alco, 'active': active,
        'age_years': age_years
    }])


    # BMI
    features_df['bmi'] = features_df['weight'] / ((features_df['height'] / 100) ** 2)

    # EXACT training binning: pd.cut([0,40,50,60,np.inf], labels=[1,2,3,4])
    features_df['age_group'] = pd.cut(
        features_df['age_years'],
        bins=[0, 40, 50, 60, np.inf],
        labels=[1, 2, 3, 4],
        right=False
    ).astype(int)

    # EXACT z-scores using loaded scalers
    features_df['ap_hi_z'] = scaler_hi.transform(features_df[['ap_hi']].to_numpy()).flatten()
    features_df['ap_lo_z'] = scaler_lo.transform(features_df[['ap_lo']].to_numpy()).flatten()


    # Exact logs (np.log, as in notebook)
    features_df['ap_hi_log'] = np.log(features_df['ap_hi'])
    features_df['ap_lo_log'] = np.log(features_df['ap_lo'])
    features_df['bmi_log'] = np.log(features_df['bmi'])

    # Reindex to training columns
    features_df = features_df.reindex(columns=feature_order, fill_value=0)

    # Cast categoricals to int
    cat_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_group']
    for col in cat_cols:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(int)

    # Predict with RF model
    prediction = model.predict(features_df.to_numpy())[0]
    proba = model.predict_proba(features_df.to_numpy())[0][1] * 100  # P(high risk)

    st.markdown('<div class="section-header">Assessment Results</div>', unsafe_allow_html=True)
    
    if prediction == 1:
        st.markdown(f"""
            <div class="result-card high-risk">
                <strong>HIGH RISK DETECTED</strong><br>
                <span style="font-size: 2rem; font-weight: 700;">{proba:.1f}%</span><br>
                <span style="font-size: 0.9rem; opacity: 0.9;">Probability of Cardiovascular Disease</span>
            </div>
        """, unsafe_allow_html=True)
        st.warning("**Recommendation:** Please consult with a healthcare professional for a comprehensive evaluation.")
    else:
        st.markdown(f"""
            <div class="result-card low-risk">
                <strong>LOW RISK PROFILE</strong><br>
                <span style="font-size: 2rem; font-weight: 700;">{proba:.1f}%</span><br>
                <span style="font-size: 0.9rem; opacity: 0.9;">Probability of Cardiovascular Disease</span>
            </div>
        """, unsafe_allow_html=True)
        st.success("**Great news!** Your risk indicators suggest a healthy cardiovascular profile.")
    
    # Display calculated metrics
    bmi = features_df['bmi'].values[0]
    st.markdown('<div class="section-header">Your Health Metrics</div>', unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">BMI</div>
                <div class="metric-value">{bmi:.1f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Blood Pressure</div>
                <div class="metric-value">{ap_hi}/{ap_lo}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        bmi_category = "Normal" if 18.5 <= bmi < 25 else ("Underweight" if bmi < 18.5 else "Overweight")
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Category</div>
                <div class="metric-value" style="font-size: 1.2rem;">{bmi_category}</div>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #555; padding: 20px;">
        <small>Disclaimer: This tool is for informational purposes only and should not replace professional medical advice.</small>
    </div>
""", unsafe_allow_html=True)





# streamlit run app.py