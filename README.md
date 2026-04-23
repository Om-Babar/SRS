# ⚡ FinAI: Algorithmic Trading AI (Viva & Defense Guide)

A high-performance algorithmic trading dashboard built using Python and Streamlit. This project aggregates daily market data, sequences historic momentums, and deploys a quad-engine ensemble Machine Learning model to forecast market trajectories.

---

## 🎓 Viva & Defense Preparation Guide

This section contains all the mathematical and logical justifications you will need to defend your project in a viva or presentation.

### 1. What does the application do?
It is an **Algorithmic Trading Dashboard** that uses an **Ensemble Machine Learning** approach to predict the next day's stock price and provide a clear `BUY`, `HOLD`, or `SELL` recommendation based on mathematical confidence.

### 2. How much data is the AI trained on?
* **Live Prediction (User Prediction / Dashboard):** The user can dynamically select the training data period (6 Months, 1 Year, 2 Years, 5 Years, or Max). By default, it uses the **past 1 year** of data. It trains on **100% of the downloaded data** to make the most accurate prediction for tomorrow.
* **Model Analytics (Diagnostics):** It fetches the past **2 years** of data to ensure a large enough sample size. It uses an **80% Training / 20% Testing split** to evaluate the model's past accuracy.

### 3. What are the 4 Machine Learning Models used?
Instead of relying on one model, the system uses a weighted **Ensemble Method**:
1. **LSTM (Deep Learning - 35% Weight):** Uses a 60-day lookback window. LSTMs are excellent at remembering sequential patterns and human emotional trading cycles over time.
2. **ARIMA (Statistical - 25% Weight):** Uses an order of (5,1,0). It grounds the AI in classical statistics, looking at the momentum of the last 5 days to stabilize volatile deep learning predictions.
3. **Random Forest (Machine Learning - 25% Weight):** Builds 100 decision trees to evaluate tabular features — specifically how the Closing price interacts with the 10-day and 50-day Moving Averages.
4. **XGBoost (Machine Learning - 15% Weight):** A highly volatile gradient booster that iteratively corrects errors made by previous decision trees.

### 4. How does the Ensemble Logic work?
The final predicted price is calculated by multiplying each model's prediction by its confidence weight and summing them up:
`Ensemble Target = (LSTM × 0.35) + (ARIMA × 0.25) + (RF × 0.25) + (XGBoost × 0.15)`

### 5. How does the system decide to BUY, HOLD, or SELL?
Once the Ensemble Target price is calculated, the system calculates the **Expected Percentage Change** from today's actual price.
* **BUY:** If the predicted change is **> +1.0%**
* **SELL:** If the predicted change is **< -1.0%**
* **HOLD:** If the predicted change is between **-1.0% and +1.0%** (The AI is not confident enough to recommend a trade).

### 6. What are the Model Diagnostics? (How do you prove it works?)
The `Model Analytics` tab runs three mathematical tests to prove the AI isn't just guessing:
1. **Z-Test (Statistical Drift):** Tests if the stock is actually trending or just moving randomly. A **P-Value < 0.05** proves the trend is statistically significant and the AI predictions can be trusted.
2. **Feature Correlation Heatmap:** Proves whether the AI's inputs (`Close`, `MA10`, `MA50`) are useful. Values near 1.0 mean redundancy; values near 0 mean independent, useful signals.
3. **Regression Matrices (MAE & RMSE):** Calculates the **Mean Absolute Error (MAE)** by splitting data 80/20. It shows exactly how many Rupees/Dollars the AI is off by on average.

### 7. What is the Moving Average Overlay?
A technical analysis chart showing the actual price overlaid with the **10-day (Short-term)** and **50-day (Long-term)** moving averages. 
* **Golden Cross (Bullish):** When the 10-day line crosses ABOVE the 50-day line.
* **Death Cross (Bearish):** When the 10-day line crosses BELOW the 50-day line.

---

## 🛠️ Technology Stack
* **Frontend UI**: Streamlit 
* **Data Pipelines**: `yfinance`, Pandas, NumPy
* **Deep Learning**: TensorFlow (GPU Accelerated), TF-Keras (LSTM)
* **Statistical Logic**: Statsmodels (ARIMA)
* **Tabular Machine Learning**: Scikit-Learn (Random Forest), XGBoost (GPU Accelerated)
* **Visualization**: Plotly

---

## 🚀 How to Run

### Method 1: Natively (Local)
Ensure you have Python 3.11 installed. Install the exact requirements natively to prevent dependency mismatches.
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

### Method 2: Docker Container (Recommended for GPU)
Isolate the entire dependency environment using the internal `Dockerfile`. Note: Requires **NVIDIA Container Toolkit**.
```bash
# Build the Environment
docker build -t pro-stock-engine .

# Ignite the Application with GPU Support
docker run --gpus all -p 8501:8501 pro-stock-engine
```
