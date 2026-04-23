import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tf_keras.models import Sequential
from tf_keras.layers import LSTM, Dense, Dropout
import plotly.express as px
import plotly.graph_objects as go
import warnings
from statsmodels.stats.weightstats import ztest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# GPU Detection Logic
gpus = tf.config.list_physical_devices('GPU')
gpu_status = f"✅ Main Engine : **ONLINE (GPU)**" if gpus else "✅ Main Engine : **ONLINE (CPU)**"
gpu_msg = f"Detecting {len(gpus)} GPU(s)" if gpus else "No GPU detected, falling back to CPU"

# Set page configuration
st.set_page_config(page_title="Pro Stock Engine", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# Currency Conversion Logic (Strictly lock to INR natively)
currency_symbol = "₹"
target_curr = "INR"

@st.cache_data(ttl=86400)
def get_exchange_rates():
    rates = {"USD": 1.0}
    try:
        inr_df = yf.download("USDINR=X", period="1d", progress=False)
        rates["INR"] = float(inr_df['Close'].iloc[-1]) if not inr_df.empty else 83.0
    except: rates["INR"] = 83.0
    try:
        eur_df = yf.download("USDEUR=X", period="1d", progress=False)
        rates["EUR"] = float(eur_df['Close'].iloc[-1]) if not eur_df.empty else 0.92
    except: rates["EUR"] = 0.92
    try:
        gbp_df = yf.download("USDGBP=X", period="1d", progress=False)
        rates["GBP"] = float(gbp_df['Close'].iloc[-1]) if not gbp_df.empty else 0.79
    except: rates["GBP"] = 0.79
    return rates

rates_dict = get_exchange_rates()

def get_origin_currency(ticker):
    ticker = ticker.upper()
    if ticker.endswith(".NS") or ticker.endswith(".BO"): return "INR"
    elif ticker.endswith(".L"): return "GBP"
    elif ticker.endswith(".DE"): return "EUR"
    else: return "USD"

def convert_val(price, origin_curr, target_curr, rates):
    if origin_curr == target_curr: return price
    price_in_usd = price / rates.get(origin_curr, 1.0)
    return price_in_usd * rates.get(target_curr, 1.0)

# Custom CSS for Premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Metric Glassmorphism */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px 25px;
        border-radius: 12px;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.15);
        border-left: 4px solid #00E676;
    }

    /* Radio buttons -> Premium Sidebar Cards */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding: 14px 18px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: scale(1.02);
        border-color: #4CAF50;
    }
    
    /* Clean up title spacing */
    h1, h2, h3 {
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Input Box Tweaks */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# Professional Sidebar UI
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 35px; margin-top: 10px;">
        <h1 style="color: #4CAF50; font-size: 42px; font-weight: 900; margin-bottom: 0px; padding: 0; line-height: 1;">⚡ Fin<span style="color: #ffffff;">AI</span></h1>
        <p style="color: #888888; font-size: 11px; font-weight: 700; letter-spacing: 3px; margin-top: 5px;">ALGORITHMIC TRADING</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size: 11px; font-weight: 700; color: #777; letter-spacing: 1px;'>🧭 SYSTEM MODULES</p>", unsafe_allow_html=True)
    app_mode = st.radio("MENU:", ["📊 Advanced Dashboard", "🔍 User Prediction", "🧠 Model Analytics", "📈 Live Trading Terminal"], label_visibility="collapsed")

    st.markdown("<br><br><p style='font-size: 11px; font-weight: 700; color: #777; letter-spacing: 1px;'>⚙️ SERVER STATUS</p>", unsafe_allow_html=True)
    st.success(f"{gpu_status}\n\n✅ Data Node : **SYNCED**\n\n🔗 Market : **GLOBAL**")

    st.markdown("""
    <div style="margin-top: 30px; text-align: center; color: #888; font-size: 12px; padding: 15px; border-radius: 8px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05);">
        <strong style='color: #ddd;'>Ensemble ML Model v3.0</strong><br>
        <span style='font-size: 11px;'>LSTM × ARIMA × RandForest × XGBoost</span><br><br>
        <div style='display:inline-block; width:8px; height:8px; background-color:#4CAF50; border-radius:50%; margin-right:5px;'></div><i>System Active</i>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def resolve_ticker(ticker):
    ticker = str(ticker).strip().upper()
    if not ticker: return ""
    if ticker.isdigit() and len(ticker) == 6: return f"{ticker}.BO"
    if "." in ticker: return ticker
    df = yf.download(ticker, period="1d", progress=False)
    if not df.empty: return ticker
    df_ns = yf.download(f"{ticker}.NS", period="1d", progress=False)
    if not df_ns.empty: return f"{ticker}.NS"
    df_bo = yf.download(f"{ticker}.BO", period="1d", progress=False)
    if not df_bo.empty: return f"{ticker}.BO"
    return ticker

@st.cache_data(ttl=3600)
def fetch_data(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.levels[1]: df = df.xs(ticker, level=1, axis=1) 
            elif ticker in df.columns.levels[0]: df = df[ticker]
        c_data = df['Close']
        if isinstance(c_data, pd.DataFrame): c_data = c_data.iloc[:, 0]
        clean_df = pd.DataFrame({'Close': c_data})
        clean_df['MA10'] = clean_df['Close'].rolling(window=10).mean()
        clean_df['MA50'] = clean_df['Close'].rolling(window=50).mean()
        clean_df.dropna(inplace=True)
        return clean_df
    except Exception as e:
        return pd.DataFrame()

# ─────────────────────────────────────────────
# HELPER: Build & train LSTM model
# ─────────────────────────────────────────────
LOOKBACK = 60

def _build_lstm_prediction(close_prices):
    """Scales data, trains LSTM, returns (model, scaler, last_sequence)."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices.reshape(-1, 1))

    X_seq, y_seq = [], []
    for i in range(LOOKBACK, len(scaled)):
        X_seq.append(scaled[i - LOOKBACK:i, 0])
        y_seq.append(scaled[i, 0])
    X_seq = np.array(X_seq).reshape(-1, LOOKBACK, 1)
    y_seq = np.array(y_seq)

    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_seq, y_seq, epochs=10, batch_size=32, verbose=0)

    last_seq = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
    return model, scaler, last_seq, scaled

def _arima_forecast(close_prices, steps=1):
    """Fits ARIMA(5,1,0) and returns a 1-d numpy array of forecasted values."""
    try:
        model = ARIMA(close_prices, order=(5, 1, 0))
        result = model.fit()
        return result.forecast(steps=steps)
    except Exception:
        # Fallback: repeat last known price
        return np.array([close_prices[-1]] * steps)

# ─────────────────────────────────────────────
# CORE: Next-day prediction ensemble
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def predict_stock_from_df(df, ticker):
    if df.empty or len(df) < 60:
        return None, None, None, None, "Insufficient Data"

    close_prices = df['Close'].values.astype(float)

    # ── LSTM ──────────────────────────────────
    lstm_model, scaler, last_seq, scaled = _build_lstm_prediction(close_prices)
    pred_lstm_scaled = lstm_model.predict(last_seq, verbose=0)[0][0]
    pred_lstm = float(scaler.inverse_transform([[pred_lstm_scaled]])[0][0])

    # ── ARIMA ─────────────────────────────────
    pred_arima = float(_arima_forecast(close_prices, steps=1)[0])

    # ── Random Forest & XGBoost (tabular) ─────
    df_tab = df.copy()
    df_tab['Target'] = df_tab['Close'].shift(-1)
    train_df = df_tab.dropna().copy()
    if len(train_df) < 10:
        return None, None, None, None, "Insufficient Data"

    X_tab = train_df[['Close', 'MA10', 'MA50']]
    y_tab = train_df['Target']

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # XGBoost GPU acceleration
    device_param = 'cuda' if gpus else 'cpu'
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0, device=device_param)
    rf.fit(X_tab, y_tab)
    xgb.fit(X_tab.values, y_tab.values)

    last_row = df.iloc[[-1]][['Close', 'MA10', 'MA50']]
    pred_rf  = float(rf.predict(last_row)[0])
    pred_xgb = float(xgb.predict(last_row.values)[0])

    # ── Weighted Ensemble ──────────────────────
    current_price = float(close_prices[-1])
    final_pred = (
        pred_lstm * 0.35 +
        pred_arima * 0.25 +
        pred_rf   * 0.25 +
        pred_xgb  * 0.15
    )

    # ── Directional Accuracy (tabular proxy) ──
    pred_y_rf  = rf.predict(X_tab)
    pred_y_xgb = xgb.predict(X_tab.values)
    ensemble_tab = (pred_y_rf * 0.5) + (pred_y_xgb * 0.5)
    actual_dir = (y_tab.values - train_df['Close'].values) > 0
    pred_dir   = (ensemble_tab - train_df['Close'].values) > 0
    accuracy_pct = (actual_dir == pred_dir).mean() * 100

    change_pct = ((final_pred - current_price) / current_price) * 100
    if change_pct > 1.0:   rec = "BUY"
    elif change_pct < -1.0: rec = "SELL"
    else:                   rec = "HOLD"

    origin    = get_origin_currency(ticker)
    curr_conv = convert_val(current_price, origin, target_curr, rates_dict)
    pred_conv = convert_val(final_pred,    origin, target_curr, rates_dict)
    return curr_conv, pred_conv, change_pct, accuracy_pct, rec

# ─────────────────────────────────────────────
# CORE: 7-Day autoregressive projection
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def predict_future_7_days(ticker):
    df = fetch_data(ticker, period="1y")
    if df.empty or len(df) < 60:
        return []

    close_prices = df['Close'].values.astype(float)

    # ── LSTM setup ────────────────────────────
    lstm_model, scaler, _, scaled = _build_lstm_prediction(close_prices)
    lstm_window = list(scaled[-LOOKBACK:, 0])

    # ── ARIMA: forecast all 7 steps at once ───
    arima_7 = _arima_forecast(close_prices, steps=7)

    # ── RF & XGBoost setup ────────────────────
    df_tab = df.copy()
    df_tab['Target'] = df_tab['Close'].shift(-1)
    train_df = df_tab.dropna().copy()
    X_tab = train_df[['Close', 'MA10', 'MA50']]
    y_tab = train_df['Target']

    rf  = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    device_param = 'cuda' if gpus else 'cpu'
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0, device=device_param)
    rf.fit(X_tab, y_tab)
    xgb.fit(X_tab.values, y_tab.values)

    # ── Autoregressive loop ───────────────────
    future_data  = []
    sim_close    = list(close_prices)
    last_row_tab = df.iloc[[-1]][['Close', 'MA10', 'MA50']].copy()

    for i in range(7):
        # LSTM step
        seq_input = np.array(lstm_window[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
        pred_lstm_s  = lstm_model.predict(seq_input, verbose=0)[0][0]
        pred_lstm    = float(scaler.inverse_transform([[pred_lstm_s]])[0][0])

        # ARIMA step (pre-computed)
        pred_arima = float(arima_7[i])

        # RF & XGBoost step
        pred_rf  = float(rf.predict(last_row_tab)[0])
        pred_xgb = float(xgb.predict(last_row_tab.values)[0])

        # Weighted ensemble
        final_pred = (
            pred_lstm  * 0.35 +
            pred_arima * 0.25 +
            pred_rf    * 0.25 +
            pred_xgb   * 0.15
        )
        future_data.append(final_pred)

        # ─ Update rolling state ───────────────
        sim_close.append(final_pred)
        next_ma10 = float(np.mean(sim_close[-10:]))
        next_ma50 = float(np.mean(sim_close[-50:]))

        # LSTM window: append scaled new price
        new_scaled = float(scaler.transform([[final_pred]])[0][0])
        lstm_window.append(new_scaled)

        last_row_tab = pd.DataFrame({'Close': [final_pred], 'MA10': [next_ma10], 'MA50': [next_ma50]})

    return future_data

@st.cache_data(ttl=3600)
def predict_stock(ticker, period="1y"):
    df = fetch_data(ticker, period=period)
    return predict_stock_from_df(df, ticker)

def get_recommendation_color(rec):
    if rec == "BUY":  return "#00E676"  # neon green
    elif rec == "SELL": return "#FF1744"  # neon red
    return "#FFEA00"  # neon yellow

def get_trend_color(trend):
    return "#00E676" if "Up" in trend else "#FF1744"

# PLOTLY THEME
plotly_theme = "plotly_dark"

if app_mode == "🔍 User Prediction":
    st.markdown("<h1 style='color: #eee;'>User Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #aaa; margin-bottom: 20px;'>Enter any stock ticker below. The AI engine will fetch live data, train 4 models, and generate predictions — all in real time.</p>", unsafe_allow_html=True)

    # Initialize state variables
    current_price = None
    predicted_price = None
    change_pct = None
    accuracy = None
    rec = None
    ticker_input = None

    # ── Clean input row ──
    inp_col, period_col, btn_col = st.columns([2.5, 1.5, 1])
    with inp_col:
        ticker_input_raw = st.text_input("Stock Ticker", value="", placeholder="e.g. RELIANCE, AAPL", label_visibility="collapsed")
    with period_col:
        selected_period = st.selectbox("Training Data", ["6mo", "1y", "2y", "5y", "max"], index=1, label_visibility="collapsed", help="Amount of historical data used to train the AI models.")
    with btn_col:
        predict_btn = st.button("⚡ Analyze", type="primary", use_container_width=True)
    # ── Only run when button is clicked AND ticker is not empty ──
    if predict_btn and ticker_input_raw.strip():
        ticker_input = resolve_ticker(ticker_input_raw)
        
        if ticker_input != ticker_input_raw.strip().upper():
            st.info(f"💡 Auto-Routed: **{ticker_input_raw.upper()}** → **{ticker_input}**")
        
        # ── Live Processing Logs ──
        with st.status("🔄 Running AI Analysis...", expanded=True) as status:
            st.write(f"📡 Fetching {selected_period} of market data...")
            df = fetch_data(ticker_input, period=selected_period)
            if df.empty or len(df) < 60:
                status.update(label="❌ Analysis Failed", state="error", expanded=True)
                st.error(f"⚠️ Could not retrieve sufficient data for **{ticker_input}**. Need at least 60 trading days.")
            else:
                close_prices = df['Close'].values.astype(float)
                origin = get_origin_currency(ticker_input)
                current_price_raw = float(close_prices[-1])
                curr_display = convert_val(current_price_raw, origin, target_curr, rates_dict)
                st.write(f"✅ Loaded **{len(df)}** trading days | Current Price: **{currency_symbol}{curr_display:,.2f}**")
                
                # ── LSTM ──
                st.write("🧠 Training LSTM neural network (60-day sequences)...")
                lstm_model, scaler, last_seq, scaled = _build_lstm_prediction(close_prices)
                pred_lstm_scaled = lstm_model.predict(last_seq, verbose=0)[0][0]
                pred_lstm = float(scaler.inverse_transform([[pred_lstm_scaled]])[0][0])
                pred_lstm_conv = convert_val(pred_lstm, origin, target_curr, rates_dict)
                lstm_chg = ((pred_lstm - current_price_raw) / current_price_raw) * 100
                st.write(f"✅ LSTM → **{currency_symbol}{pred_lstm_conv:,.2f}** ({lstm_chg:+.2f}%) | Weight: **35%** | Contribution: **{currency_symbol}{convert_val(pred_lstm * 0.35, origin, target_curr, rates_dict):,.2f}**")
                
                # ── ARIMA ──
                st.write("📊 Running ARIMA(5,1,0) statistical model...")
                pred_arima = float(_arima_forecast(close_prices, steps=1)[0])
                pred_arima_conv = convert_val(pred_arima, origin, target_curr, rates_dict)
                arima_chg = ((pred_arima - current_price_raw) / current_price_raw) * 100
                st.write(f"✅ ARIMA → **{currency_symbol}{pred_arima_conv:,.2f}** ({arima_chg:+.2f}%) | Weight: **25%** | Contribution: **{currency_symbol}{convert_val(pred_arima * 0.25, origin, target_curr, rates_dict):,.2f}**")
                
                # ── Random Forest ──
                st.write("🌲 Fitting Random Forest (100 decision trees)...")
                df_tab = df.copy()
                df_tab['Target'] = df_tab['Close'].shift(-1)
                train_df = df_tab.dropna().copy()
                X_tab = train_df[['Close', 'MA10', 'MA50']]
                y_tab = train_df['Target']
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_tab, y_tab)
                last_row = df.iloc[[-1]][['Close', 'MA10', 'MA50']]
                pred_rf = float(rf.predict(last_row)[0])
                pred_rf_conv = convert_val(pred_rf, origin, target_curr, rates_dict)
                rf_chg = ((pred_rf - current_price_raw) / current_price_raw) * 100
                st.write(f"✅ Random Forest → **{currency_symbol}{pred_rf_conv:,.2f}** ({rf_chg:+.2f}%) | Weight: **25%** | Contribution: **{currency_symbol}{convert_val(pred_rf * 0.25, origin, target_curr, rates_dict):,.2f}**")
                
                # ── XGBoost ──
                st.write("⚡ Training XGBoost (gradient boosting)...")
                device_param = 'cuda' if gpus else 'cpu'
                xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0, device=device_param)
                xgb.fit(X_tab.values, y_tab.values)
                pred_xgb = float(xgb.predict(last_row.values)[0])
                pred_xgb_conv = convert_val(pred_xgb, origin, target_curr, rates_dict)
                xgb_chg = ((pred_xgb - current_price_raw) / current_price_raw) * 100
                st.write(f"✅ XGBoost → **{currency_symbol}{pred_xgb_conv:,.2f}** ({xgb_chg:+.2f}%) | Weight: **15%** | Contribution: **{currency_symbol}{convert_val(pred_xgb * 0.15, origin, target_curr, rates_dict):,.2f}**")
                
                # ── Ensemble Sum ──
                st.write("---")
                st.write("🔗 **Computing Weighted Ensemble Sum...**")
                final_pred = pred_lstm * 0.35 + pred_arima * 0.25 + pred_rf * 0.25 + pred_xgb * 0.15
                predicted_price = convert_val(final_pred, origin, target_curr, rates_dict)
                current_price = convert_val(current_price_raw, origin, target_curr, rates_dict)
                change_pct = ((final_pred - current_price_raw) / current_price_raw) * 100
                
                st.write(f"**Ensemble Target = (LSTM×0.35) + (ARIMA×0.25) + (RF×0.25) + (XGB×0.15)**")
                st.write(f"**Ensemble Target = {currency_symbol}{predicted_price:,.2f}** (Change: **{change_pct:+.2f}%**)")
                
                # ── Threshold Comparison ──
                st.write("---")
                st.write("📏 **Comparing against decision thresholds...**")
                st.write(f"Change = **{change_pct:+.2f}%** vs Thresholds: BUY > +1.0% | SELL < -1.0% | HOLD = between")
                
                if change_pct > 1.0: rec = "BUY"
                elif change_pct < -1.0: rec = "SELL"
                else: rec = "HOLD"
                
                rec_color = "#00E676" if rec == "BUY" else ("#FF1744" if rec == "SELL" else "#FFEA00")
                st.write(f"**→ {change_pct:+.2f}% {'>' if change_pct > 1.0 else ('<' if change_pct < -1.0 else 'is between')} {'1.0%' if change_pct > 1.0 else ('-1.0%' if change_pct < -1.0 else '±1.0%')} → Conclusion: {rec}**")
                
                # ── Directional accuracy ──
                pred_y_rf = rf.predict(X_tab)
                pred_y_xgb = xgb.predict(X_tab.values)
                ensemble_tab = (pred_y_rf * 0.5) + (pred_y_xgb * 0.5)
                actual_dir = (y_tab.values - train_df['Close'].values) > 0
                pred_dir = (ensemble_tab - train_df['Close'].values) > 0
                accuracy = float((actual_dir == pred_dir).mean() * 100)
                st.write(f"📊 Directional Accuracy (Backtest): **{accuracy:.1f}%**")
                
                status.update(label=f"✅ Analysis Complete → {rec}", state="complete", expanded=False)
        
        # ── Results Section (only if analysis succeeded) ──
        if current_price is not None:
            st.markdown("---")
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.02); padding: 15px 20px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 20px;'>
                <p style='font-size: 12px; color: #777; margin: 0; letter-spacing: 1px; font-weight: 700;'>📊 RESULTS FOR <span style='color: #00E676;'>{ticker_input}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Current Price", f"{currency_symbol}{current_price:,.2f}")
            mcol2.metric("AI Target (Next Day)", f"{currency_symbol}{predicted_price:,.2f}", f"{change_pct:+.2f}%")
            
            color = get_recommendation_color(rec)
            mcol3.markdown(f"""
            <div data-testid="metric-container" style="border-left-color: {color};">
                <div style="font-size: 14px; color: #aaa; margin-bottom: 4px;">AI Recommendation</div>
                <div style="font-size: 28px; font-weight: 800; color: {color};">{rec}</div>
            </div>
            """, unsafe_allow_html=True)
            
            mcol4.metric("Directional Accuracy", f"{accuracy:.1f}%", "Historic Backtest", delta_color="off")

            # ── Ensemble Breakdown ──
            with st.expander("🔬 Ensemble Breakdown — How did the AI reach this conclusion?", expanded=False):
                st.markdown("<p style='color: #bbb; font-size: 13px; line-height: 1.7;'>The final price is a <b>weighted average</b> of 4 independent models:</p>", unsafe_allow_html=True)
                w1, w2, w3, w4 = st.columns(4)
                w1.markdown("<div style='text-align:center; padding: 15px; background: rgba(0,176,255,0.06); border-radius: 10px; border: 1px solid rgba(0,176,255,0.15);'><div style='font-size: 28px; font-weight: 800; color: #00B0FF;'>35%</div><div style='font-size: 13px; font-weight: 700; color: #ddd; margin-top: 4px;'>LSTM</div><div style='font-size: 10px; color: #888; margin-top: 4px;'>Deep Learning<br>60-day sequences</div></div>", unsafe_allow_html=True)
                w2.markdown("<div style='text-align:center; padding: 15px; background: rgba(255,234,0,0.06); border-radius: 10px; border: 1px solid rgba(255,234,0,0.15);'><div style='font-size: 28px; font-weight: 800; color: #FFEA00;'>25%</div><div style='font-size: 13px; font-weight: 700; color: #ddd; margin-top: 4px;'>ARIMA</div><div style='font-size: 10px; color: #888; margin-top: 4px;'>Statistical Model<br>Trend & drift</div></div>", unsafe_allow_html=True)
                w3.markdown("<div style='text-align:center; padding: 15px; background: rgba(0,230,118,0.06); border-radius: 10px; border: 1px solid rgba(0,230,118,0.15);'><div style='font-size: 28px; font-weight: 800; color: #00E676;'>25%</div><div style='font-size: 13px; font-weight: 700; color: #ddd; margin-top: 4px;'>Random Forest</div><div style='font-size: 10px; color: #888; margin-top: 4px;'>100 Decision Trees<br>MA10 & MA50 features</div></div>", unsafe_allow_html=True)
                w4.markdown("<div style='text-align:center; padding: 15px; background: rgba(255,23,68,0.06); border-radius: 10px; border: 1px solid rgba(255,23,68,0.15);'><div style='font-size: 28px; font-weight: 800; color: #FF1744;'>15%</div><div style='font-size: 13px; font-weight: 700; color: #ddd; margin-top: 4px;'>XGBoost</div><div style='font-size: 10px; color: #888; margin-top: 4px;'>Gradient Boosting<br>Error correction</div></div>", unsafe_allow_html=True)
                    
            # ── Charts Section ──
            st.markdown("---")

            # ── 180-Day Trajectory ──
            st.markdown("""
            <h3 style='color: #eee;'>📈 180-Day Lookback</h3>
            <p style='color: #777; font-size: 13px; margin-top: -5px;'>Historical closing price over the past 6 months. Line color reflects the current trend.</p>
            """, unsafe_allow_html=True)
            df_chart = fetch_data(ticker_input, period="6mo")
            if not df_chart.empty:
                origin = get_origin_currency(ticker_input)
                df_chart['Close'] = df_chart['Close'].apply(lambda x: convert_val(x, origin, target_curr, rates_dict))
                df_chart['MA50'] = df_chart['MA50'].apply(lambda x: convert_val(x, origin, target_curr, rates_dict))
                
                ma_trend = "Uptrend ↗" if current_price > df_chart['MA50'].iloc[-1] else "Downtrend ↘"
                trend_color = '#00E676' if "Up" in ma_trend else "#FF1744"
                
                st.markdown(f"""
                <div style='display: inline-block; padding: 6px 16px; border-radius: 20px; background: {"rgba(0,230,118,0.1)" if "Up" in ma_trend else "rgba(255,23,68,0.1)"}; border: 1px solid {trend_color}; margin-bottom: 15px;'>
                    <span style='color: {trend_color}; font-weight: 700; font-size: 13px;'>{ma_trend}</span>
                </div>
                """, unsafe_allow_html=True)
                
                fig = px.line(df_chart, y='Close', template=plotly_theme)
                fig.update_traces(line_color=trend_color, line_width=2.5)
                fig.update_layout(
                    yaxis_title=f"Market Price ({currency_symbol})", xaxis_title=None,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=10, b=0), height=350
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")

                # ── Past 7-Day Momentum ──
                st.markdown("""
                <h3 style='color: #eee;'>⚡ Last 7 Trading Days</h3>
                <p style='color: #777; font-size: 13px; margin-top: -5px;'>Recent momentum. Each dot is one trading day's closing price.</p>
                """, unsafe_allow_html=True)
                df_7d = df_chart.tail(7).copy()
                fig_7d = px.line(df_7d, y='Close', template=plotly_theme, markers=True)
                fig_7d.update_traces(line_color='#FFEA00', line_width=2.5, marker=dict(size=8, color='#FFEA00'))
                fig_7d.update_layout(
                    yaxis_title=f"Price ({currency_symbol})", xaxis_title=None,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=10, b=0), height=250
                )
                st.plotly_chart(fig_7d, use_container_width=True)
                
                st.markdown("---")

                # ── 7-Day Future Projection ──
                st.markdown("""
                <h3 style='color: #eee;'>🔮 Next 7-Day Prediction</h3>
                <p style='color: #777; font-size: 13px; margin-top: -5px;'>AI-generated future prices. Each day's prediction feeds into the next (autoregressive). Accuracy decreases with each additional day.</p>
                """, unsafe_allow_html=True)
                
                future_7 = predict_future_7_days(ticker_input)
                if future_7:
                    future_7_conv = [convert_val(v, origin, target_curr, rates_dict) for v in future_7]
                    last_date = df_chart.index[-1]
                    
                    try:
                        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=7)
                    except:
                        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
                    if len(future_dates) < 7:
                        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
                        
                    df_future = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_7_conv})
                    
                    fig_future = px.line(df_future, x='Date', y='Predicted Close', template=plotly_theme, markers=True)
                    fig_future.update_traces(line_color='#00B0FF', line_width=3, marker=dict(size=8, color='#00E676', symbol='circle'))
                    fig_future.update_layout(
                        yaxis_title=f"Projected Price ({currency_symbol})", xaxis_title=None,
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=10, b=0), height=250
                    )
                    st.plotly_chart(fig_future, use_container_width=True)
                    
                    st.markdown("""
                    <div style='background: rgba(255, 23, 68, 0.05); border-left: 4px solid #FF1744; padding: 15px; margin-top: 25px; border-radius: 4px;'>
                        <h4 style='color: #FF1744; margin-top: 0; margin-bottom: 8px; font-weight: 800;'>⚠️ Educational Disclaimer</h4>
                        <p style='color: #ddd; font-size: 13px; line-height: 1.6; margin-bottom: 0;'>
                        This projection is for <b>educational and study purposes only</b>. Stock markets are influenced by unpredictable real-world events that no historical model can anticipate.<br><br>
                        <b>Do NOT make real trades based on this forecast.</b> Always consult a licensed financial advisor.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)


elif app_mode == "📊 Advanced Dashboard":
    st.markdown("<h1 style='color: #eee;'>Market Sector Scanning</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #aaa; margin-bottom: 20px;'>Select stocks from the pool below and run comparative AI analysis across all 4 models.</p>", unsafe_allow_html=True)
    
    # Pool of 20 popular stocks
    stock_pool = [
        "AAPL", "TSLA", "MSFT", "GOOGL", "META", "NVDA", "AMZN", "INTC", "BABA", "NFLX",
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS", "WIPRO.NS", "LT.NS", "BHARTIARTL.NS", "TATAMOTORS.NS"
    ]
    
    stocks = st.multiselect(
        "Pick stocks to analyze (select 2 or more):",
        options=stock_pool,
        default=["AAPL", "RELIANCE.NS", "TCS.NS", "TSLA", "NVDA"],
        help="Choose from 20 global & Indian stocks for comparative analysis"
    )
    
    if len(stocks) < 2:
        st.warning("Please select at least 2 stocks to run comparative analysis.")
    
    colA, colB, colC = st.columns([1.5, 2, 3])
    with colA:
        dash_period = st.selectbox("Training Data", ["6mo", "1y", "2y", "5y", "max"], index=1, label_visibility="collapsed")
    with colB:
        run_dash = st.button("🚀 Run Batch Analysis", type="primary", use_container_width=True)
        
    if run_dash and len(stocks) >= 2:
        st.markdown("---")
        with st.spinner(f"Analyzing all stocks in parallel using {dash_period} of data..."):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            import concurrent.futures
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_ticker = {executor.submit(predict_stock, t, dash_period): t for t in stocks}
                    completed = 0
                    
                    for future in concurrent.futures.as_completed(future_to_ticker):
                        ticker = future_to_ticker[future]
                        completed += 1
                        
                        status_text.markdown(f"<span style='color: #00E676; font-weight: 600;'>Analyzing</span> <span style='color: #ddd;'>{ticker}</span> <span style='color: #666;'>({completed}/{len(stocks)})</span>", unsafe_allow_html=True)
                        progress_bar.progress(completed / len(stocks))
                        
                        curr, pred, chg, acc, rec = future.result()
                        if curr is not None:
                            results.append({
                                "Asset ID": ticker,
                                rf"Price ({currency_symbol})": round(curr, 2),
                                rf"Target ({currency_symbol})": round(pred, 2),
                                "Daily Shift (%)": round(chg, 2),
                                "Precision (%)": round(acc, 2),
                                "Action": rec
                            })
                status_text.markdown("<span style='color: #00E676;'>✅ Analysis complete.</span>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Analysis interrupted: {e}")
            
            if results:
                df_results = pd.DataFrame(results)
                df_results = df_results.sort_values("Daily Shift (%)", ascending=False).reset_index(drop=True)
                
                # ── Top Performers ──
                st.markdown("""
                <h3 style='color: #eee; margin-top: 30px; margin-bottom: 5px;'>🏆 Top 3 Performers</h3>
                <p style='color: #777; font-size: 13px; margin-bottom: 20px;'>The assets with the highest projected daily yield, sorted by predicted price change.</p>
                """, unsafe_allow_html=True)
                top_picks = df_results.head(3)
                if not top_picks.empty:
                    cols = st.columns(len(top_picks))
                    for i, (_, row) in enumerate(top_picks.iterrows()):
                        cols[i].metric(
                            f"{row['Asset ID']} ({row['Action']})", 
                            f"{currency_symbol}{row[rf'Target ({currency_symbol})']:,.2f}", 
                            f"{row['Daily Shift (%)']}% Yield"
                        )
                
                st.markdown("---")
                
                # ── Full Results Table ──
                st.markdown("""
                <h3 style='color: #eee; margin-bottom: 5px;'>📊 Complete Results</h3>
                <p style='color: #777; font-size: 13px; margin-bottom: 20px;'>Full breakdown of all analyzed stocks. <b style='color:#00E676;'>Green</b> = BUY, <b style='color:#FFEA00;'>Yellow</b> = HOLD, <b style='color:#FF1744;'>Red</b> = SELL.</p>
                """, unsafe_allow_html=True)
                
                def style_rec(val):
                    color = get_recommendation_color(val)
                    return f'color: {color}; font-weight: 800;'
                
                st.dataframe(
                    df_results.style.map(style_rec, subset=['Action'])
                                    .format({
                                        rf"Price ({currency_symbol})": f"{currency_symbol}{{:.2f}}", 
                                        rf"Target ({currency_symbol})": f"{currency_symbol}{{:.2f}}", 
                                        "Daily Shift (%)": "{:.2f}%", 
                                        "Precision (%)": "{:.2f}%"}),
                    use_container_width=True,
                    hide_index=True
                )
                
                
                st.markdown("---")
                
                # ── Bar Chart & Pie Chart (merged from Graphs page) ──
                st.markdown("""
                <h3 style='color: #eee; margin-bottom: 5px;'>📊 Yield Comparison & Recommendation Split</h3>
                <p style='color: #777; font-size: 13px; margin-bottom: 20px;'>Bar chart shows each stock's projected daily change (%). Donut chart shows the BUY/HOLD/SELL distribution.</p>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([1.5, 1])
                with col1:
                    fig_bar = px.bar(
                        df_results, x='Asset ID', y='Daily Shift (%)', 
                        color='Action',
                        color_discrete_map={"BUY": "#00E676", "HOLD": "#ffc107", "SELL": "#FF1744"},
                        text_auto='.2f', template=plotly_theme
                    )
                    fig_bar.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
                        margin=dict(l=0, r=0, b=0, t=20),
                        yaxis_title="Daily Shift (%)", xaxis_title=""
                    )
                    fig_bar.update_traces(marker_line_width=0)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                with col2:
                    fig_pie = px.pie(
                        df_results, names='Action', 
                        color='Action',
                        color_discrete_map={"BUY": "#00E676", "HOLD": "#ffc107", "SELL": "#FF1744"},
                        hole=0.55, template=plotly_theme
                    )
                    fig_pie.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
                        margin=dict(l=0, r=0, b=0, t=20)
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)))
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                st.markdown("---")
                
                # ── Moving Average Overlay ──
                st.markdown("""
                <h3 style='color: #eee; margin-bottom: 5px;'>📈 Moving Average Overlay</h3>
                <p style='color: #777; font-size: 13px; margin-bottom: 20px;'>Select a stock to see its 1-year price with <b style='color:#00B0FF;'>10-day</b> and <b style='color:#FF1744;'>50-day</b> moving averages.</p>
                """, unsafe_allow_html=True)
                selected_stock = st.selectbox("Select a stock to inspect:", df_results['Asset ID'].tolist())
                
                if selected_stock:
                    with st.spinner(f"Loading chart for {selected_stock}..."):
                        df_hist = fetch_data(selected_stock, period="1y")
                        if not df_hist.empty:
                            origin = get_origin_currency(selected_stock)
                            df_hist['Close'] = df_hist['Close'].apply(lambda x: convert_val(x, origin, target_curr, rates_dict))
                            df_hist['MA10'] = df_hist['MA10'].apply(lambda x: convert_val(x, origin, target_curr, rates_dict))
                            df_hist['MA50'] = df_hist['MA50'].apply(lambda x: convert_val(x, origin, target_curr, rates_dict))
                            
                            fig_ma = go.Figure()
                            fig_ma.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Close'], mode='lines', name='Closing Price', line=dict(color='#EEEEEE', width=2)))
                            fig_ma.add_trace(go.Scatter(x=df_hist.index, y=df_hist['MA10'], mode='lines', name='10-Day MA', line=dict(color='#00B0FF', width=2, dash='dot')))
                            fig_ma.add_trace(go.Scatter(x=df_hist.index, y=df_hist['MA50'], mode='lines', name='50-Day MA', line=dict(color='#FF1744', width=2.5)))
                            
                            fig_ma.update_layout(
                                xaxis_title="", yaxis_title=f"Price ({currency_symbol})", 
                                template=plotly_theme,
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
                                margin=dict(l=0, r=0, t=10, b=0)
                            )
                            st.plotly_chart(fig_ma, use_container_width=True)

elif app_mode == "🧠 Model Analytics":
    st.markdown("<h1 style='color: #eee;'>🧠 Model Analytics & Diagnostics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #aaa; margin-bottom: 20px;'>Evaluate the mathematical accuracy and reliability of the AI engine using statistical tests and regression metrics.</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style='color: #eee; margin-bottom: 5px;'>Select a Stock to Analyze</h3>
    <p style='color: #777; font-size: 13px; margin-bottom: 20px;'>Enter any ticker symbol. The system will fetch 2 years of data and run diagnostic tests on it.</p>
    """, unsafe_allow_html=True)
    selected_stock = st.text_input("Enter Asset Ticker (e.g. RELIANCE.NS, TSLA)", value="RELIANCE.NS")
    
    if selected_stock:
        selected_stock = resolve_ticker(selected_stock)
        with st.spinner(f"Analyzing {selected_stock}..."):
            df = fetch_data(selected_stock, period="2y")
            
            if not df.empty and len(df) > 60:
                origin = get_origin_currency(selected_stock)
                
                # Convert base prices to targeted equivalent for metric continuity 
                df['Close'] = df['Close'].apply(lambda x: convert_val(x, origin, target_curr, rates_dict))
                df['MA10'] = df['MA10'].apply(lambda x: convert_val(x, origin, target_curr, rates_dict))
                df['MA50'] = df['MA50'].apply(lambda x: convert_val(x, origin, target_curr, rates_dict))
                
                # Calculate Daily Returns natively
                df['Daily_Return'] = df['Close'].pct_change()
                df_clean = df.dropna().copy()
                
                # --- 1. Z-Test ---
                st.markdown("---")
                st.markdown("""
                <h3 style='color: #eee; margin-bottom: 5px;'>1. Statistical Drift Test (Z-Test)</h3>
                <p style='color: #777; font-size: 13px; margin-bottom: 5px;'><b>Purpose:</b> Determines if the stock has a genuine directional trend or if price movements are just random noise.</p>
                <p style='color: #666; font-size: 12px; margin-bottom: 20px;'><b>How it works:</b> Compares the average daily return against zero. If P-Value < 0.05 → the trend is statistically significant (not random). The Z-Statistic sign tells us the direction.</p>
                """, unsafe_allow_html=True)
                z_stat, p_value = ztest(df_clean['Daily_Return'], value=0)
                
                z_col1, z_col2, z_col3 = st.columns(3)
                z_col1.metric("Z-Statistic", f"{z_stat:.4f}")
                z_col2.metric("P-Value", f"{p_value:.6f}")
                
                if p_value < 0.05:
                    if z_stat > 0:
                        msg = "POSITIVE DRIFT (Bullish)"
                        color = "#00E676"
                        interpretation = f"The Z-Test confirms **{selected_stock}** has a **statistically significant upward trend** (P={p_value:.4f} < 0.05). This means the stock's upward movement is **not random** — there is real buying momentum. AI predictions for this stock carry **higher confidence**."
                    else:
                        msg = "NEGATIVE DRIFT (Bearish)"
                        color = "#FF1744"
                        interpretation = f"The Z-Test confirms **{selected_stock}** has a **statistically significant downward trend** (P={p_value:.4f} < 0.05). This means the decline is **not random** — there is real selling pressure. AI predictions should be treated with **caution** as bearish momentum may continue."
                else:
                    msg = "RANDOM WALK (No Bias)"
                    color = "#FFEA00"
                    interpretation = f"The Z-Test shows **{selected_stock}** has **no significant trend** (P={p_value:.4f} > 0.05). Price movements are essentially random noise. This means the AI has **less directional signal** to work with — predictions may be less reliable for this stock."
                    
                z_col3.markdown(f"""
                <div data-testid="metric-container" style="border-left-color: {color};">
                    <div style="font-size: 14px; color: #aaa; margin-bottom: 4px;">Market Bias</div>
                    <div style="font-size: 16px; font-weight: 800; color: {color};">{msg}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style='background: rgba(255,255,255,0.03); border-left: 4px solid {color}; padding: 12px 15px; border-radius: 4px; margin-top: 10px;'>
                    <p style='color: #bbb; font-size: 12px; line-height: 1.6; margin: 0;'>
                    <b>📌 Interpretation:</b> {interpretation}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                col_hm, col_reg = st.columns([1, 1.2])
                
                # --- 2. Heatmap Engine (Plotly Smooth) ---
                with col_hm:
                    st.markdown("""
                    <h3 style='color: #eee; margin-bottom: 5px;'>2. Feature Correlation Heatmap</h3>
                    <p style='color: #777; font-size: 13px; margin-bottom: 5px;'><b>Purpose:</b> Checks if the AI's input features provide unique information or are redundant copies of each other.</p>
                    <p style='color: #666; font-size: 12px; margin-bottom: 20px;'><b>How it works:</b> Calculates Pearson correlation between all feature pairs. Values near <b>1.0</b> = redundant, near <b>0</b> = independent signal.</p>
                    """, unsafe_allow_html=True)
                    corr_matrix = df_clean[['Close', 'MA10', 'MA50', 'Daily_Return']].corr()
                    
                    fig_hm = px.imshow(
                        corr_matrix, 
                        text_auto=True, 
                        color_continuous_scale=['#FF1744', '#1E1E1E', '#00E676'], 
                        aspect="auto",
                        template=plotly_theme
                    )
                    fig_hm.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
                        margin=dict(l=0, r=0, t=10, b=0), height=350
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)
                    
                    # Dynamic interpretation
                    close_ma10_corr = corr_matrix.loc['Close', 'MA10']
                    return_corr = corr_matrix.loc['Close', 'Daily_Return']
                    st.markdown(f"""
                    <div style='background: rgba(255,255,255,0.03); border-left: 4px solid #00B0FF; padding: 12px 15px; border-radius: 4px; margin-top: 10px;'>
                        <p style='color: #bbb; font-size: 12px; line-height: 1.6; margin: 0;'>
                        <b>📌 Interpretation:</b> Close↔MA10 correlation = <b>{close_ma10_corr:.3f}</b> ({"high redundancy — MA10 mirrors price closely" if close_ma10_corr > 0.95 else "moderate — MA10 adds smoothing value"}). 
                        Daily Return correlation with Close = <b>{return_corr:.3f}</b> ({"weak — good, it provides independent signal" if abs(return_corr) < 0.3 else "moderate — some overlap"}).
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # --- 3. Regression Error ---
                with col_reg:
                    st.markdown("""
                    <h3 style='color: #eee; margin-bottom: 5px;'>3. Prediction Accuracy (XGBoost)</h3>
                    <p style='color: #777; font-size: 13px; margin-bottom: 5px;'><b>Purpose:</b> Measures how close the AI's predicted prices are to actual historical prices using a held-out test set.</p>
                    <p style='color: #666; font-size: 12px; margin-bottom: 20px;'><b>How it works:</b> Trains XGBoost on 80% of data, tests on the remaining 20%. Each dot = one prediction. Closer to the <b style='color:#FF1744;'>red line</b> = more accurate.</p>
                    """, unsafe_allow_html=True)
                    df_tab = df_clean.copy()
                    df_tab['Target'] = df_tab['Close'].shift(-1)
                    df_tab = df_tab.dropna()
                    
                    X_class = df_tab[['Close', 'MA10', 'MA50']]
                    y_class = df_tab['Target']
                    
                    X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, shuffle=False)
                    device_param = 'cuda' if gpus else 'cpu'
                    xgb_eval = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0, device=device_param)
                    xgb_eval.fit(X_train.values, y_train.values)
                    y_pred_prices = xgb_eval.predict(X_test.values)
                    
                    mae = mean_absolute_error(y_test, y_pred_prices)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred_prices))
                    
                    fig_scat = px.scatter(
                        x=y_test, y=y_pred_prices, 
                        opacity=0.6, 
                        color_discrete_sequence=['#00B0FF'],
                        template=plotly_theme,
                        labels={'x': f'Actual Price ({currency_symbol})', 'y': f'Predicted Price ({currency_symbol})'}
                    )
                    
                    min_val = min(y_test.min(), y_pred_prices.min())
                    max_val = max(y_test.max(), y_pred_prices.max())
                    fig_scat.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="#FF1744", width=2, dash="dash"))
                    
                    fig_scat.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
                        margin=dict(l=0, r=0, t=10, b=0), height=300
                    )
                    st.plotly_chart(fig_scat, use_container_width=True)
                    
                    e_col1, e_col2 = st.columns(2)
                    e_col1.metric("Mean Absolute Error", f"{currency_symbol}{mae:.2f}")
                    e_col2.metric("Root Mean Sq. Error", f"{currency_symbol}{rmse:.2f}")
                    
                    # Dynamic interpretation based on price range
                    avg_price = float(y_test.mean())
                    mae_pct = (mae / avg_price) * 100 if avg_price > 0 else 0
                    if mae_pct < 1:
                        quality = "Excellent"
                        quality_color = "#00E676"
                        quality_msg = "The model is highly accurate — predictions deviate by less than 1% from actual prices."
                    elif mae_pct < 3:
                        quality = "Good"
                        quality_color = "#FFEA00"
                        quality_msg = "The model performs well with predictions within 3% of actual prices."
                    else:
                        quality = "Fair"
                        quality_color = "#FF1744"
                        quality_msg = "The model has moderate accuracy. Consider this stock's high volatility when interpreting predictions."
                    
                    st.markdown(f"""
                    <div style='background: rgba(255,255,255,0.03); border-left: 4px solid {quality_color}; padding: 12px 15px; border-radius: 4px; margin-top: 10px;'>
                        <p style='color: #bbb; font-size: 12px; line-height: 1.6; margin: 0;'>
                        <b>📌 Interpretation:</b> MAE = <b>{currency_symbol}{mae:.2f}</b> on avg price of <b>{currency_symbol}{avg_price:,.2f}</b> → error rate of <b>{mae_pct:.2f}%</b>. 
                        Model quality: <b style='color:{quality_color};'>{quality}</b>. {quality_msg}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.error("Insufficient market data (need at least 60 trading days). Try a different ticker.")

elif app_mode == "📈 Live Trading Terminal":
    st.markdown("<h1 style='color: #eee;'>Live Trading Terminal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #aaa; margin-bottom: 20px;'>Real-time charting powered by TradingView. Search any stock or crypto, add indicators (RSI, MACD, Bollinger), and switch timeframes.</p>", unsafe_allow_html=True)
    
    tradingview_html = """
    <style>
      .tv-container { width: 100%; height: 80vh; min-height: 500px; max-height: 900px; }
      .tv-container .tradingview-widget-container { width: 100% !important; height: 100% !important; }
    </style>
    <div class="tv-container">
      <div class="tradingview-widget-container" style="width:100%;height:100%;">
        <div class="tradingview-widget-container__widget" style="width:100%;height:100%;"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
        {
          "autosize": true,
          "symbol": "NASDAQ:AAPL",
          "interval": "D",
          "timezone": "Asia/Kolkata",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "allow_symbol_change": true,
          "save_image": true,
          "calendar": false,
          "hide_volume": false,
          "support_host": "https://www.tradingview.com"
        }
        </script>
      </div>
    </div>
    """
    
    components.html(tradingview_html, height=700, scrolling=False)

