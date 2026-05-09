import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="CreditIQ | Predictive Risk Analysis",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOAD ASSETS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
IMAGE_PATH = os.path.join(BASE_DIR, 'finance_credit_card_dashboard_hero_1778150356803.png')

@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.error("Model files not found. Please run train_model.py first.")
        return None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_models()

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- CUSTOM CSS ---
def local_css():
    img_base64 = ""
    if os.path.exists(IMAGE_PATH):
        img_base64 = get_base64_of_bin_file(IMAGE_PATH)
    
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        
        * {{
            font-family: 'Inter', sans-serif;
        }}
        
        .main {{
            background: radial-gradient(circle at top right, #1e3a8a, #0f172a);
            color: #f8fafc;
        }}
        
        .stApp {{
            background: transparent;
        }}
        
        .hero-section {{
            height: 300px;
            background-image: linear-gradient(rgba(15, 23, 42, 0.6), rgba(15, 23, 42, 0.9)), url('data:image/png;base64,{img_base64}');
            background-size: cover;
            background-position: center;
            border-radius: 20px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .glass-card {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 2rem;
            margin-bottom: 1rem;
        }}
        
        .stButton>button {{
            width: 100%;
            border-radius: 12px;
            height: 3.5em;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            color: white;
            font-weight: 800;
            font-size: 1.1rem;
            border: none;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        }}
        
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(59, 130, 246, 0.6);
            border: none;
            color: white;
        }}
        
        .prediction-box {{
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin-top: 1rem;
            animation: fadeIn 0.8s ease-in-out;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .risk-high {{
            background: rgba(220, 38, 38, 0.2);
            border: 1px solid #ef4444;
            color: #f87171;
        }}
        
        .risk-low {{
            background: rgba(5, 150, 105, 0.2);
            border: 1px solid #10b981;
            color: #34d399;
        }}
        
        .metric-label {{
            font-size: 0.9rem;
            color: #94a3b8;
            margin-bottom: 0.2rem;
        }}
        
        .metric-value {{
            font-size: 1.5rem;
            font-weight: 600;
        }}
        
        /* Fix tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 20px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px;
            color: #94a3b8;
            font-weight: 600;
        }}
        
        .stTabs [aria-selected="true"] {{
            color: #3b82f6 !important;
            border-bottom-color: #3b82f6 !important;
        }}
        
        /* Number inputs and selects styling */
        .stNumberInput input, .stSelectbox div[data-baseweb="select"] {{
            background-color: rgba(15, 23, 42, 0.5) !important;
            color: white !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
        }}
        </style>
    """, unsafe_allow_html=True)

local_css()

# --- HERO SECTION ---
st.markdown(f"""
    <div class="hero-section">
        <h1 style='font-weight: 800; font-size: 3.5rem; margin-bottom: 0;'>CreditIQ</h1>
        <p style='font-size: 1.2rem; opacity: 0.8; letter-spacing: 2px;'>ADVANCED RISK PREDICTIVE ENGINE</p>
    </div>
""", unsafe_allow_html=True)

# --- APP BODY ---
col_form, col_res = st.columns([1.5, 1])

with col_form:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0'>Input Parameters</h3>", unsafe_allow_html=True)
    
    tabs = st.tabs(["👤 Profile", "⏳ History", "💰 Financials"])
    
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            limit_bal = st.number_input("Credit Limit Balance (NT$)", min_value=1000, value=50000, step=1000)
            sex = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        with c2:
            age = st.slider("Age", 18, 100, 30)
            education = st.selectbox("Education", options=[1, 2, 3, 4], 
                                     format_func=lambda x: ["Graduate School", "University", "High School", "Others"][x-1])
        
        marriage = st.selectbox("Marital Status", options=[1, 2, 3], 
                                format_func=lambda x: ["Married", "Single", "Others"][x-1])

    with tabs[1]:
        st.markdown("<p style='font-size:0.8rem; color:#94a3b8'>Status Key: -1=Duly, 1=1mo Delay, 2=2mo Delay, etc.</p>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            pay_0 = st.number_input("Sept Status", -2, 8, 0)
            pay_2 = st.number_input("Aug Status", -2, 8, 0)
        with c2:
            pay_3 = st.number_input("July Status", -2, 8, 0)
            pay_4 = st.number_input("June Status", -2, 8, 0)
        with c3:
            pay_5 = st.number_input("May Status", -2, 8, 0)
            pay_6 = st.number_input("April Status", -2, 8, 0)

    with tabs[2]:
        sub_tab = st.tabs(["Bills", "Payments"])
        with sub_tab[0]:
            c1, c2 = st.columns(2)
            with c1:
                bill_amt1 = st.number_input("Bill Sept", value=0)
                bill_amt2 = st.number_input("Bill Aug", value=0)
                bill_amt3 = st.number_input("Bill July", value=0)
            with c2:
                bill_amt4 = st.number_input("Bill June", value=0)
                bill_amt5 = st.number_input("Bill May", value=0)
                bill_amt6 = st.number_input("Bill April", value=0)
        with sub_tab[1]:
            c1, c2 = st.columns(2)
            with c1:
                pay_amt1 = st.number_input("Paid Sept", value=0)
                pay_amt2 = st.number_input("Paid Aug", value=0)
                pay_amt3 = st.number_input("Paid July", value=0)
            with c2:
                pay_amt4 = st.number_input("Paid June", value=0)
                pay_amt5 = st.number_input("Paid May", value=0)
                pay_amt6 = st.number_input("Paid April", value=0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("RUN PREDICTION ENGINE")
    st.markdown('</div>', unsafe_allow_html=True)

with col_res:
    st.markdown('<div class="glass-card" style="height: 100%">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0'>Risk Analysis Result</h3>", unsafe_allow_html=True)
    
    if predict_btn:
        if model is not None and scaler is not None:
            features = [
                limit_bal, sex, education, marriage, age,
                pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
                bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
                pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6
            ]
            
            input_df = pd.DataFrame([features], columns=[
                'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
                'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
            ])
            
            # Scale and Predict
            features_scaled = scaler.transform(input_df)
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            if prediction == 1:
                st.markdown(f"""
                    <div class="prediction-box risk-high">
                        <h2 style='margin-bottom:0'>CRITICAL RISK</h2>
                        <p style='font-size:1.2rem'>Potential Default Detected</p>
                        <h1 style='font-size:4rem; margin:1rem 0'>{probability[1]:.1%}</h1>
                        <p>PROBABILITY SCORE</p>
                    </div>
                """, unsafe_allow_html=True)
                st.error("Financial institution advised to review this profile immediately.")
            else:
                st.markdown(f"""
                    <div class="prediction-box risk-low">
                        <h2 style='margin-bottom:0'>LOW RISK</h2>
                        <p style='font-size:1.2rem'>Healthy Credit Profile</p>
                        <h1 style='font-size:4rem; margin:1rem 0'>{probability[0]:.1%}</h1>
                        <p>STABILITY SCORE</p>
                    </div>
                """, unsafe_allow_html=True)
                st.success("This profile shows strong indicators of financial stability.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Model Insights")
            st.write("The Random Forest classifier has analyzed your inputs against historical default patterns. The score above represents the model's confidence in this classification.")
            
        else:
            st.error("System Error: Model components failed to initialize.")
    else:
        st.markdown("""
            <div style='text-align:center; padding: 4rem 2rem; opacity: 0.5'>
                <p style='font-size: 4rem'>⏳</p>
                <p>Awaiting Data Input...<br>Configure parameters and run the engine.</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; margin-top: 2rem; opacity: 0.6; font-size: 0.8rem;'>
        &copy; 2026 CreditIQ Technologies | Powered by Advanced Machine Learning | Unauthorized access prohibited
    </div>
""", unsafe_allow_html=True)
