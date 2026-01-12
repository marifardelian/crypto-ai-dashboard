import streamlit as st
import pandas as pd
import requests
import joblib
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Bitcoin Market Analyst",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Professional Dark Mode) ---
st.markdown("""
<style>
    /* Force Dark Theme Backgrounds */
    .stApp { background-color: #0E1117; }
    
    /* Metrics Cards Styling */
    div[data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 5px;
        color: #fff;
    }
    
    /* Typography */
    h1, h2, h3 { font-family: 'Arial', sans-serif; color: #E0E0E0; }
    p, label { color: #B0B0B0; }
    
    /* Remove default Streamlit padding */
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_all_models():
    models = {}
    try:
        models["Random Forest"] = joblib.load('Random_Forest_model.pkl')
        models["Logistic Regression"] = joblib.load('Logistic_Regression_model.pkl')
        models["SVM"] = joblib.load('SVM_model.pkl')
    except Exception:
        return None
    return models

models = load_all_models()

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.header("Control Panel")
    model_choice = st.selectbox("Select Model", ["Random Forest", "Logistic Regression", "SVM"])
    st.divider()
    st.text("System Status: Online")
    st.text("Data Source: CoinGecko API")

# --- ROBUST DATA ENGINE ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def get_live_data():
    try:
        # Attempt to fetch real data
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {'vs_currency': 'usd', 'days': '90', 'interval': 'daily'} 
        
        response = requests.get(url, params=params, timeout=5)
        if response.status_code != 200:
            raise Exception("API Error")
            
        data = response.json()
        if 'prices' not in data:
            raise Exception("Empty Data")

        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df, False # False = Real Data

    except Exception:
        # FALLBACK: Generate simulation data if API fails
        dates = pd.date_range(end=pd.Timestamp.now(), periods=90)
        base_price = 90000
        prices = []
        for _ in range(90):
            change = np.random.uniform(-0.03, 0.03)
            base_price = base_price * (1 + change)
            prices.append(base_price)
            
        df = pd.DataFrame({'timestamp': dates, 'price': prices})
        return df, True # True = Simulated Data

df, is_simulated = get_live_data()

# --- DISPLAY WARNING IF SIMULATED ---
if is_simulated:
    st.warning("Notice: API connection limit reached. Displaying simulated market data.")

# --- FEATURE ENGINEERING ---
df['MA_7'] = df['price'].rolling(window=7).mean()
df['MA_30'] = df['price'].rolling(window=30).mean()
df['Volatility'] = df['price'].rolling(window=7).std()

latest = df.iloc[-1]
features = [[latest['price'], latest['MA_7'], latest['MA_30'], latest['Volatility']]]

# --- PREDICTION LOGIC ---
if models:
    current_model = models[model_choice]
    prediction = current_model.predict(features)[0]
    probs = current_model.predict_proba(features)[0]
    confidence = max(probs) * 100
    
    # Text Outputs
    direction = "UP" if prediction == 1 else "DOWN"
    color_code = "#00FF00" if prediction == 1 else "#FF0000" # Green vs Red
else:
    direction = "Error"
    confidence = 0.0
    color_code = "#FFFFFF"

# --- DASHBOARD LAYOUT ---
st.title("Bitcoin Market Analyst")
st.markdown("Real-time market analysis powered by Machine Learning.")

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${latest['price']:,.0f}")
col2.metric("Model Prediction", direction)
col3.metric("Confidence Score", f"{confidence:.1f}%")
col4.metric("Market Volatility", f"{latest['Volatility']:.2f}")

st.divider()

# --- PRO CHART WITH FORECAST ---
st.subheader(f"Forecast Analysis ({model_choice})")

fig = go.Figure()

# 1. Historical Price (Solid Line)
fig.add_trace(go.Scatter(
    x=df['timestamp'], 
    y=df['price'], 
    mode='lines', 
    name='Historical Price',
    line=dict(color='#FFFFFF', width=2)
))

# 2. Moving Averages
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['MA_7'], mode='lines', name='7-Day MA',
    line=dict(color='#FFFF00', width=1, dash='dot')
))
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['MA_30'], mode='lines', name='30-Day MA',
    line=dict(color='#0000FF', width=1)
))

# 3. THE MAGIC: Projected Forecast Line
last_date = df['timestamp'].iloc[-1]
next_date = last_date + timedelta(days=1)

# Prediction math: If UP, +2%. If DOWN, -2%
forecast_value = latest['price'] * (1.02 if prediction == 1 else 0.98)

# FIXED: Removed 'dash' property and increased width to 4 to ensure visibility
fig.add_trace(go.Scatter(
    x=[last_date, next_date],
    y=[latest['price'], forecast_value],
    mode='lines+markers',
    name='Projected Path',
    line=dict(color=color_code, width=4), # Solid thick line
    marker=dict(size=10, color=color_code)
))

# 4. Chart Styling
fig.update_layout(
    height=500,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color="white"),
    xaxis=dict(showgrid=False, title="Date"),
    yaxis=dict(showgrid=True, gridcolor='#333', title="Price (USD)"),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# --- TECHNICAL DATA TABLE ---
with st.expander("View Technical Analysis Data"):
    st.write("Model Consensus:")
    
    comp_data = []
    if models:
        for name, model in models.items():
            pred = model.predict(features)[0]
            prob = max(model.predict_proba(features)[0]) * 100
            signal = "BULLISH" if pred == 1 else "BEARISH"
            comp_data.append({"Model": name, "Signal": signal, "Confidence": f"{prob:.1f}%"})
        
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)