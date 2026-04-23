import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells.extend([
    nbf.v4.new_markdown_cell('# Pro Stock Engine - Step-by-Step Training\nThis notebook breaks down the algorithmic stock prediction pipeline from the main app into executable steps for experimentation in Google Colab.'),
    
    nbf.v4.new_markdown_cell('## 1. Imports and Setup\nFirst, we load the required libraries and suppress unnecessary warnings from TensorFlow.'),
    nbf.v4.new_code_cell('''import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')'''),
    
    nbf.v4.new_markdown_cell('## 2. Configuration & Asset Selection\nSet the target stock ticker symbol. You can test with symbols like `AAPL`, `TSLA`, `RELIANCE.NS`, etc.'),
    nbf.v4.new_code_cell('''TARGET_TICKER = "RELIANCE.NS"
# You can change PERIOD to "6mo", "1y", "2y", "5y", or "max" just like the Dashboard
PERIOD = "1y"'''),

    nbf.v4.new_markdown_cell('## 3. Data Fetching & Preprocessing\nDownload historical stock data from Yahoo Finance and compute the 10-day and 50-day moving averages, simulating market momentum.'),
    nbf.v4.new_code_cell('''def fetch_data(ticker, period="1y"):
    print(f"Fetching {period} data for {ticker}...")
    df = yf.download(ticker, period=period, progress=False)
    
    if df.empty: 
        raise ValueError("No data returned!")
        
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

df = fetch_data(TARGET_TICKER, PERIOD)
display(df.tail())'''),

    nbf.v4.new_markdown_cell('## 4. LSTM Neural Network (Deep Learning Pipeline)\nThe Long Short-Term Memory network expects scaled sequence data consisting of the past 60 days of closing prices.'),
    nbf.v4.new_code_cell('''LOOKBACK = 60
close_prices = df['Close'].values.astype(float)

# 1. Scale Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(close_prices.reshape(-1, 1))

# 2. Build Sequences
X_seq, y_seq = [], []
for i in range(LOOKBACK, len(scaled)):
    X_seq.append(scaled[i - LOOKBACK:i, 0])
    y_seq.append(scaled[i, 0])
X_seq = np.array(X_seq).reshape(-1, LOOKBACK, 1)
y_seq = np.array(y_seq)

# 3. Model Architecture
lstm_model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')

# 4. Training
print("Training LSTM network...")
lstm_model.fit(X_seq, y_seq, epochs=10, batch_size=32, verbose=1)

# 5. Predict the next day
last_seq = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
pred_lstm_scaled = lstm_model.predict(last_seq, verbose=0)[0][0]
pred_lstm = float(scaler.inverse_transform([[pred_lstm_scaled]])[0][0])
print(f"\\nLSTM Extrapolation: {pred_lstm:.2f}")'''),

    nbf.v4.new_markdown_cell('## 5. ARIMA (Statistical Modeling)\nAutoregressive Integrated Moving Average (5, 1, 0) captures inherent trend logic without requiring structural sequences.'),
    nbf.v4.new_code_cell('''print("Fitting ARIMA (5,1,0) model...")
try:
    model = ARIMA(close_prices, order=(5, 1, 0))
    result = model.fit()
    pred_arima = float(result.forecast(steps=1)[0])
except Exception as e:
    print(f"ARIMA Failed: {e}")
    pred_arima = float(close_prices[-1])

print(f"ARIMA Projection: {pred_arima:.2f}")'''),

    nbf.v4.new_markdown_cell('## 6. Random Forest & XGBoost (Decision Trees)\nPreparing tabular features using our earlier Moving Averages, where the `Target` is simply the closing price of the next trading day.'),
    nbf.v4.new_code_cell('''# Prepare Tabular Data
df_tab = df.copy()
df_tab['Target'] = df_tab['Close'].shift(-1)
train_df = df_tab.dropna().copy()

X_tab = train_df[['Close', 'MA10', 'MA50']]
y_tab = train_df['Target']

# Initialize Models
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0)

# Train Models
print("Training Forest & Gradient Boosting layers...")
rf.fit(X_tab, y_tab)
xgb.fit(X_tab.values, y_tab.values)

# Get today's values to predict tomorrow
last_row = df.iloc[[-1]][['Close', 'MA10', 'MA50']]

# Predict
pred_rf  = float(rf.predict(last_row)[0])
pred_xgb = float(xgb.predict(last_row.values)[0])

print(f"Random Forest Expectation: {pred_rf:.2f}")
print(f"XGBoost Expectation: {pred_xgb:.2f}")'''),

    nbf.v4.new_markdown_cell('## 7. Ensemble Assembly & Final Projection\nThe final result merges the models using a weighted logic focusing on deep patterns while anchoring with baseline stats.'),
    nbf.v4.new_code_cell('''current_price = float(close_prices[-1])

# Weighted Ensemble
final_pred = (
    pred_lstm * 0.35 +
    pred_arima * 0.25 +
    pred_rf   * 0.25 +
    pred_xgb  * 0.15
)

# Market Shift
change_pct = ((final_pred - current_price) / current_price) * 100

if change_pct > 1.0:   rec = "BUY"
elif change_pct < -1.0: rec = "SELL"
else:                   rec = "HOLD"

print(f"--- Analysis for {TARGET_TICKER} ---")
print(f"Current Asset Price: {current_price:.2f}")
print(f"Next Day Target:     {final_pred:.2f} ({change_pct:+.2f}%)")
print(f"Algorithm Action:    {rec}")'''),
    
    nbf.v4.new_markdown_cell('## 8. Visualize Component Alignment\nAn optional block to view previous momentum (the past 180 days) matching the UI logic.'),
    nbf.v4.new_code_cell('''df_chart = df.tail(180)
fig = px.line(df_chart, y=['Close', 'MA10', 'MA50'], 
              title=f"{TARGET_TICKER} 180-Day Momentum Overlays",
              template="plotly_dark")
fig.show()'''),
    
    nbf.v4.new_markdown_cell('## 9. Z-Test on Daily Returns\nA statistical test to evaluate if the stock has a statistically significant core drift vs just random noise.'),
    nbf.v4.new_code_cell('''from statsmodels.stats.weightstats import ztest

# 1. Calculate the daily percentage returns of the stock
df['Daily_Return'] = df['Close'].pct_change().dropna()

# 2. Perform a One-Sample Z-Test
z_stat, p_value = ztest(df['Daily_Return'], value=0)

print("--- Z-Test for Daily Returns ---")
print(f"Z-Statistic: {z_stat:.4f}")
print(f"P-Value:     {p_value:.6f}")
print("-" * 32)

# 3. Interpret the Results at a 95% Confidence Level
if p_value < 0.05:
    if z_stat > 0:
        print("💡 Conclusion: POSITIVE DRIFT (Bullish)\\nReject the Null Hypothesis. The stock has a statistically significant upward trend. This means the stock's upward movement is not random — there is real buying momentum. AI predictions for this stock carry higher confidence.")
    else:
        print("💡 Conclusion: NEGATIVE DRIFT (Bearish)\\nReject the Null Hypothesis. The stock has a statistically significant downward trend. This means the decline is not random — there is real selling pressure. AI predictions should be treated with caution.")
else:
    print("📉 Conclusion: RANDOM WALK (No Bias)\\nFail to reject the Null Hypothesis. Price movements are essentially random noise. This means the AI has less directional signal to work with — predictions may be less reliable for this stock.")'''),
    
    nbf.v4.new_markdown_cell('## 10. Education: Correlation Heatmap\nHere we use Seaborn to see how "redundant" our data is. If two variables are highly correlated (dark red), feeding them both to an AI might be useless since they provide the exact same information!'),
    nbf.v4.new_code_cell('''import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
# We calculate correlation across numerical data
corr_matrix = df[['Close', 'MA10', 'MA50', 'Daily_Return']].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap")
plt.show()'''),

    nbf.v4.new_markdown_cell('## 11. Regression Metrics (Evaluating Exact Prices)\nSince our models predict the EXACT dollar amount of the stock (Regression) rather than categories, we evaluate them using Mean Absolute Error (MAE) and Scatter Plots instead of Confusion Matrices.'),
    nbf.v4.new_code_cell('''from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Train/Test Split based on Tabular Data we already have
X_train, X_test, y_train, y_test = train_test_split(X_tab, y_tab, test_size=0.2, shuffle=False)

# Re-train XGBoost strictly on the Training set so we can test it blindly
xgb_eval = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
xgb_eval.fit(X_train.values, y_train.values)

# Predict the exact prices on the blind Test set
y_pred_prices = xgb_eval.predict(X_test.values)

# 1. Error Metrics
mae = mean_absolute_error(y_test, y_pred_prices)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_prices))

print("--- Regression Error Metrics ---")
print(f"Mean Absolute Error (MAE): ₹{mae:.2f}")
print(f"Root Mean Squared Error:   ₹{rmse:.2f}")
avg_price = y_test.mean()
mae_pct = (mae / avg_price) * 100
print(f"\\nThis means our AI is on average off by {mae:.2f} per prediction.")
if mae_pct < 2.0:
    print(f"Error is only {mae_pct:.1f}% of the average stock price -> Excellent model quality.")
else:
    print(f"Error is {mae_pct:.1f}% of the average stock price -> Moderate/Poor model quality.")

# 2. Scatter Plot: Predicted vs Actual Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_prices, alpha=0.5, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Perfect prediction line
plt.xlabel("Actual Stock Closing Prices")
plt.ylabel("AI Predicted Closing Prices")
plt.title("XGBoost Regression Accuracy (Predicted vs Actual)")
plt.show()''')
])

output_file = 'Pro_Stock_Engine.ipynb'
with open(output_file, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Successfully generated {output_file}")
