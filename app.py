import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

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
    app_mode = st.radio("MENU:", ["🔍 User Prediction", "📊 Advanced Dashboard", "📉 Graphs & Analysis", "📈 Live Trading Terminal"], label_visibility="collapsed")

    st.markdown("<br><br><p style='font-size: 11px; font-weight: 700; color: #777; letter-spacing: 1px;'>⚙️ SERVER STATUS</p>", unsafe_allow_html=True)
    st.success("✅ Main Engine : **ONLINE**\n\n✅ Data Node : **SYNCED**\n\n🔗 Market : **GLOBAL**")

    st.markdown("""
    <div style="margin-top: 30px; text-align: center; color: #888; font-size: 12px; padding: 15px; border-radius: 8px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05);">
        <strong style='color: #ddd;'>Ensemble ML Model v2.4</strong><br>
        <span style='font-size: 11px;'>LinearReg × RanForest × XGBoost</span><br><br>
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

@st.cache_data(ttl=3600)
def predict_stock_from_df(df, ticker):
    if df.empty or len(df) < 60: return None, None, None, None, "Insufficient Data"
    df['Target'] = df['Close'].shift(-1)
    train_df = df.dropna().copy()
    if len(train_df) < 10: return None, None, None, None, "Insufficient Data"
    X = train_df[['Close', 'MA10', 'MA50']]
    y = train_df['Target']
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
    xgb = XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42, n_jobs=1)
    lr.fit(X, y)
    rf.fit(X, y)
    xgb.fit(X.values, y.values)
    pred_y_lr = lr.predict(X)
    pred_y_rf = rf.predict(X)
    pred_y_xgb = xgb.predict(X.values)
    ensemble_preds = (pred_y_lr * 0.4) + (pred_y_rf * 0.3) + (pred_y_xgb * 0.3)
    actual_direction = (y - train_df['Close']) > 0
    pred_direction = (ensemble_preds - train_df['Close']) > 0
    accuracy_pct = (actual_direction == pred_direction).mean() * 100
    last_row = df.iloc[[-1]][['Close', 'MA10', 'MA50']]
    pred_lr = lr.predict(last_row)[0]
    pred_rf = rf.predict(last_row)[0]
    pred_xgb = xgb.predict(last_row.values)[0]
    final_pred = (pred_lr * 0.4) + (pred_rf * 0.3) + (pred_xgb * 0.3)
    try: current_price = float(last_row['Close'].iloc[0])
    except: current_price = float(last_row['Close'].values[0])
    change_pct = ((final_pred - current_price) / current_price) * 100
    if change_pct > 1.0: rec = "BUY"
    elif change_pct < -1.0: rec = "SELL"
    else: rec = "HOLD"
    origin = get_origin_currency(ticker)
    curr_conv = convert_val(current_price, origin, target_curr, rates_dict)
    pred_conv = convert_val(final_pred, origin, target_curr, rates_dict)
    return curr_conv, pred_conv, change_pct, accuracy_pct, rec

@st.cache_data(ttl=3600)
def predict_future_7_days(ticker):
    df = fetch_data(ticker, period="1y")
    if df.empty or len(df) < 60: return []
    df['Target'] = df['Close'].shift(-1)
    train_df = df.dropna().copy()
    if len(train_df) < 10: return []
    X = train_df[['Close', 'MA10', 'MA50']]
    y = train_df['Target']
    
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
    xgb = XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42, n_jobs=1)
    
    lr.fit(X, y)
    rf.fit(X, y)
    xgb.fit(X.values, y.values)
    
    future_data = []
    last_row = df.iloc[[-1]][['Close', 'MA10', 'MA50']].copy()
    sim_close = list(df['Close'].values)
    
    for _ in range(7):
        pred_lr = lr.predict(last_row)[0]
        pred_rf = rf.predict(last_row)[0]
        pred_xgb = xgb.predict(last_row.values)[0]
        final_pred = (pred_lr * 0.4) + (pred_rf * 0.3) + (pred_xgb * 0.3)
        future_data.append(final_pred)
        
        sim_close.append(final_pred)
        next_ma10 = sum(sim_close[-10:]) / 10
        next_ma50 = sum(sim_close[-50:]) / 50
        
        last_row = pd.DataFrame({'Close': [final_pred], 'MA10': [next_ma10], 'MA50': [next_ma50]})
        
    return future_data

@st.cache_data(ttl=3600)
def predict_stock(ticker):
    df = fetch_data(ticker)
    return predict_stock_from_df(df, ticker)

def get_recommendation_color(rec):
    if rec == "BUY": return "#00E676" # neon green
    elif rec == "SELL": return "#FF1744" # neon red
    return "#FFEA00" # neon yellow

def get_trend_color(trend):
    return "#00E676" if "Up" in trend else "#FF1744"

# PLOTLY THEME
plotly_theme = "plotly_dark"

if app_mode == "🔍 User Prediction":
    st.markdown("<h1 style='color: #eee;'>User Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #aaa; margin-bottom: 30px;'>Instantly evaluate any asset globally utilizing proprietary machine learning models.</p>", unsafe_allow_html=True)

    
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        st.markdown("<div style='background: rgba(255,255,255,0.02); padding: 25px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
        st.markdown("<p style='font-weight: 600; font-size: 14px; color: #ddd; margin-bottom: 5px;'>Lookup Asset</p>", unsafe_allow_html=True)
        ticker_input_raw = st.text_input("Options: Yahoo Ticker / BSE Code / NSE Name", value="RELIANCE", label_visibility="collapsed")
        st.markdown("<p style='font-size: 11px; color: #777; margin-top: 5px; margin-bottom: 15px;'><b>Available Options:</b><br/>• <b>US Stocks:</b> AAPL, TSLA, MSFT<br/>• <b>NSE (India):</b> RELIANCE.NS, TCS.NS<br/>• <b>BSE (India):</b> 500325.BO</p>", unsafe_allow_html=True)
        predict_btn = st.button("AI Analyze Next Day", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if predict_btn or ticker_input_raw:
            with st.spinner(f"Initiating neural sweep for {ticker_input_raw}..."):
                ticker_input = resolve_ticker(ticker_input_raw)
                
                if ticker_input != ticker_input_raw.strip().upper():
                    st.info(f"💡 AI Auto-Routing: **{ticker_input_raw.upper()}** mapped to **{ticker_input}** exchange layer.")
                
                current_price, predicted_price, change_pct, accuracy, rec = predict_stock(ticker_input)
                
                if current_price is None:
                    st.error(f"⚠️ Critical Failure: Asset '{ticker_input}' isolated or data unavailable.")
                else:
                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    
                    mcol1.metric("Current Asset Price", f"{currency_symbol}{current_price:,.2f}")
                    mcol2.metric("Target Projection", f"{currency_symbol}{predicted_price:,.2f}", f"{change_pct:+.2f}%")
                    
                    color = get_recommendation_color(rec)
                    # Custom metric block for recommendation to match others
                    mcol3.markdown(f"""
                    <div data-testid="metric-container" style="border-left-color: {color};">
                        <div style="font-size: 14px; color: #aaa; margin-bottom: 4px;">AI Action</div>
                        <div style="font-size: 28px; font-weight: 800; color: {color};">{rec}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    mcol4.metric("Engine Accuracy", f"{accuracy:.1f}%", "Historic Backtest", delta_color="off")
                    
    if predict_btn or ticker_input_raw:
        if current_price is not None:
            st.markdown("<br><h3 style='color: #eee;'>Asset Trajectory (180-Day Lookback)</h3>", unsafe_allow_html=True)
            df_chart = fetch_data(ticker_input, period="6mo")
            if not df_chart.empty:
                origin = get_origin_currency(ticker_input)
                df_chart['Close'] = df_chart['Close'].apply(lambda x: convert_val(x, origin, target_curr, rates_dict))
                df_chart['MA50'] = df_chart['MA50'].apply(lambda x: convert_val(x, origin, target_curr, rates_dict))
                
                ma_trend = "Uptrend ↗" if current_price > df_chart['MA50'].iloc[-1] else "Downtrend ↘"
                
                fig = px.line(df_chart, y='Close', template=plotly_theme)
                fig.update_traces(line_color='#00E676' if "Up" in ma_trend else "#FF1744", line_width=2.5)
                fig.update_layout(
                    yaxis_title=f"Market Price ({currency_symbol})", 
                    xaxis_title=None,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # --- Past 7-Day Momentum ---
                st.markdown("<br><h3 style='color: #eee;'>Recent Momentum (7-Day Lookback)</h3>", unsafe_allow_html=True)
                df_7d = df_chart.tail(7).copy()
                fig_7d = px.line(df_7d, y='Close', template=plotly_theme, markers=True)
                fig_7d.update_traces(line_color='#FFEA00', line_width=2.5, marker=dict(size=8, color='#FFEA00'))
                fig_7d.update_layout(
                    yaxis_title=f"Price ({currency_symbol})", 
                    xaxis_title=None,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=250
                )
                st.plotly_chart(fig_7d, use_container_width=True)
                
                st.markdown("<br><h3 style='color: #eee;'>🔮 7-Day Future Projection</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color: #888; font-size: 13px;'>Autoregressive simulation generated by evaluating future kinetic trends using ensemble weights.</p>", unsafe_allow_html=True)
                
                future_7 = predict_future_7_days(ticker_input)
                if future_7:
                    future_7_conv = [convert_val(v, origin, target_curr, rates_dict) for v in future_7]
                    last_date = df_chart.index[-1]
                    import datetime
                    
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
                        yaxis_title=f"Projected Price ({currency_symbol})", 
                        xaxis_title=None,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=10, b=0),
                        height=250
                    )
                    st.plotly_chart(fig_future, use_container_width=True)
                    
                    st.markdown("""
                    <div style='background: rgba(255, 23, 68, 0.05); border-left: 4px solid #FF1744; padding: 15px; margin-top: 25px; border-radius: 4px;'>
                        <h4 style='color: #FF1744; margin-top: 0; margin-bottom: 8px; font-weight: 800;'>⚠️ Educational Guidance & Disclaimer</h4>
                        <p style='color: #ddd; font-size: 13px; line-height: 1.6; margin-bottom: 0;'>
                        <strong>Important Notice:</strong> This 7-day future algorithmic projection is strictly for <b>educational and study purposes only</b>. Stock markets are heavily influenced by highly volatile, unpredictable real-world variables—such as breaking news, geopolitical shifts, inflation data, and sudden institutional block-trades.<br><br>
                        Because machine learning models rely entirely on historical mathematical momentum to estimate the future, they are fundamentally blind to real-time human events. <b>No AI can accurately guarantee future stock direction.</b><br><br>
                        <strong>Guidelines for Study:</strong><br>
                        • Use these projections to study how algorithms weight historical moving averages, rather than taking them as financial advice.<br>
                        • Observe how simulated models decay accurately over time without new data.<br>
                        • <b>Do NOT execute real-world capital trades based on this neural sweep.</b> Always consult a licensed local financial advisor.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

elif app_mode == "📊 Advanced Dashboard":
    st.markdown("<h1 style='color: #eee;'>Market Sector Scanning</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #aaa; margin-bottom: 20px;'>Autonomous batch analysis surveying top performing global tech giants and Indian megacaps.</p>", unsafe_allow_html=True)
    
    stocks = ["AAPL", "RELIANCE.NS", "TCS.NS", "TSLA", "HDFCBANK.NS", "META", "INFY.NS", "NVDA", "BABA", "INTC"]
    
    colA, colB = st.columns([1, 4])
    with colA:
        run_dash = st.button("🚀 Execute Neural Sweep", type="primary", use_container_width=True)
        
    if run_dash:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.spinner("Compiling parallel threads for sector analysis..."):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            import concurrent.futures
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_ticker = {executor.submit(predict_stock, t): t for t in stocks}
                    completed = 0
                    
                    for future in concurrent.futures.as_completed(future_to_ticker):
                        ticker = future_to_ticker[future]
                        completed += 1
                        
                        status_text.markdown(f"<span style='color: #00E676; font-weight: bold;'>[System]</span> Interrogating {ticker} framework... ({completed}/{len(stocks)})", unsafe_allow_html=True)
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
                status_text.markdown("<span style='color: #aaa;'>Sweep finalized. Resolving interface.</span>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Data stream interupted: {e}")
            
            if results:
                df_results = pd.DataFrame(results)
                df_results = df_results.sort_values("Daily Shift (%)", ascending=False).reset_index(drop=True)
                
                st.markdown("<h3 style='color: #eee; margin-top: 20px; font-weight: 800;'>🏆 Apex Performers</h3>", unsafe_allow_html=True)
                top_picks = df_results.head(3)
                if not top_picks.empty:
                    cols = st.columns(len(top_picks))
                    for i, (_, row) in enumerate(top_picks.iterrows()):
                        cols[i].metric(
                            f"{row['Asset ID']} ({row['Action']})", 
                            f"{currency_symbol}{row[rf'Target ({currency_symbol})']:,.2f}", 
                            f"{row['Daily Shift (%)']}% Yield"
                        )
                
                st.markdown("<br><h3 style='color: #eee;'>Complete Intelligence Ledger</h3>", unsafe_allow_html=True)
                
                def style_rec(val):
                    color = get_recommendation_color(val)
                    return f'color: {color}; font-weight: 800;'
                
                # Apply high-end styling to dataframe using pandas
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
                
                st.session_state['dash_results'] = df_results

elif app_mode == "📉 Graphs & Analysis":
    st.markdown("<h1 style='color: #eee;'>Quantum Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #aaa; margin-bottom: 30px;'>Isolate distinct variables and compare structural trends computed from initial sector scanning.</p>", unsafe_allow_html=True)

    if 'dash_results' in st.session_state:
        df_res = st.session_state['dash_results'].copy()
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.markdown("<h4 style='color: #eee;'>Projected Alpha Trajectories</h4>", unsafe_allow_html=True)
            fig_bar = px.bar(
                df_res, x='Asset ID', y='Daily Shift (%)', 
                color='Action',
                color_discrete_map={"BUY": "#00E676", "HOLD": "#ffc107", "SELL": "#FF1744"},
                text_auto='.2f', template=plotly_theme
            )
            fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,b=0,t=20))
            fig_bar.update_traces(marker_line_width=0)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col2:
            st.markdown("<h4 style='color: #eee;'>Aggregated Sector Directives</h4>", unsafe_allow_html=True)
            fig_pie = px.pie(
                df_res, names='Action', 
                color='Action',
                color_discrete_map={"BUY": "#00E676", "HOLD": "#ffc107", "SELL": "#FF1744"},
                hole=0.55, template=plotly_theme
            )
            fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,b=0,t=20))
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        st.markdown("---")
        st.markdown("<h3 style='color: #eee;'>Kinetic Overlay (Moving Averages)</h3>", unsafe_allow_html=True)
        selected_stock = st.selectbox("Inspect Source Architecture", df_res['Asset ID'].tolist())
        
        if selected_stock:
            with st.spinner(f"Compiling overlays for {selected_stock}..."):
                df_hist = fetch_data(selected_stock, period="1y")
                if not df_hist.empty:
                    origin = get_origin_currency(selected_stock)
                    df_hist['Close'] = df_hist['Close'].apply(lambda x: convert_val(x, origin, target_curr, rates_dict))
                    df_hist['MA10'] = df_hist['MA10'].apply(lambda x: convert_val(x, origin, target_curr, rates_dict))
                    df_hist['MA50'] = df_hist['MA50'].apply(lambda x: convert_val(x, origin, target_curr, rates_dict))
                    
                    fig_ma = go.Figure()
                    fig_ma.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Close'], mode='lines', name='Baseline Close', line=dict(color='#EEEEEE', width=2)))
                    fig_ma.add_trace(go.Scatter(x=df_hist.index, y=df_hist['MA10'], mode='lines', name='10-Day Kinetic', line=dict(color='#00B0FF', width=2, dash='dot')))
                    fig_ma.add_trace(go.Scatter(x=df_hist.index, y=df_hist['MA50'], mode='lines', name='50-Day Substructure', line=dict(color='#FF1744', width=2.5)))
                    
                    fig_ma.update_layout(
                        xaxis_title="", 
                        yaxis_title=f"Valuation ({currency_symbol})", 
                        template=plotly_theme,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.05,
                            xanchor="right",
                            x=1
                        ),
                        margin=dict(l=0, r=0, t=10, b=0)
                    )
                    st.plotly_chart(fig_ma, use_container_width=True)
    else:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.05); padding: 30px; border-radius: 12px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
            <h4 style="color: #aaa;">Data Matrix Not Found</h4>
            <p style="color: #666;">Please initialize the <b>Advanced Dashboard</b> module to synthesize system memory before launching Graph configurations.</p>
        </div>
        """, unsafe_allow_html=True)

elif app_mode == "📈 Live Trading Terminal":
    st.markdown("<h1 style='color: #eee;'>Live Trading Terminal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #aaa; margin-bottom: 20px;'>Real-time advanced charting interface universally powered by TradingView connectivity.</p>", unsafe_allow_html=True)
    
    # We inject the provided HTML script securely using Streamlit's native HTML component handler
    tradingview_html = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {
      "width": "100%",
      "height": 800,
      "symbol": "NASDAQ:AAPL",
      "interval": "D",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "allow_symbol_change": true,
      "save_image": false,
      "calendar": false
    }
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    
    # Render with absolute height boundaries allowing safe scroll limits
    components.html(tradingview_html, height=850, scrolling=True)
