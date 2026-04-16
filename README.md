# ⚡ Pro Stock Engine: Algorithmic Trading AI

A high-performance stock tracking and algorithmic analysis dashboard built using Python and Streamlit. The application aggregates daily market data, sequences historic momentums, and deploys a quad-engine ensemble Machine Learning model to forecast market trajectories with precision.

---

## 🛠️ Technology Stack
* **Frontend UI**: Streamlit 
* **Data Pipelines**: `yfinance`, Pandas, NumPy
* **Deep Learning (Sequential)**: TensorFlow (CPU), TF-Keras (LSTM Networks)
* **Statistical Logic**: Statsmodels (ARIMA)
* **Tabular Machine Learning**: Scikit-Learn (Random Forest), XGBoost (Gradient Boosting Trees)
* **Visualization**: Plotly, Seaborn, Matplotlib
* **Containerization**: Docker (`python:3.11-slim`)

---

## 🧠 Machine Learning Architecture
Instead of relying on a single structural entity, the Pro Stock Engine uses a weighted **Ensemble Method**:
1. **LSTM (Long Short-Term Memory)**: Evaluates the raw sequential pricing over a rolling 60-day window to map human emotional trading dependencies. [Weight: 35%]
2. **ARIMA (5,1,0)**: Grounds the AI in classical statistics, calculating stationary shifts and drift to stabilize volatile predictions. [Weight: 25%]
3. **Random Forest**: Builds 100 decision trees to evaluate the stock's closing price cross-referencing its 10-day and 50-day moving averages (MA10, MA50). [Weight: 25%]
4. **XGBoost**: A highly volatile gradient booster that corrects statistical errors iteratively using tabular momentum data. [Weight: 15%]

---

## 🎛️ Application Modules

* **🔍 User Prediction**: Look up any Yahoo Finance ticker (e.g. `AAPL`, `RELIANCE.NS`) to view a simulated 7-day algorithmic trajectory, historic line charts, and the AI's conclusive `BUY`/`SELL` directive.
* **📊 Advanced Dashboard**: An autonomous sweeping tool that parallel-processes ten prominent global stocks concurrently to extract top-performing sector alpha yields.
* **📉 Graphs & Analysis**: View quantum analytics displaying sector directives via interactive Plotly Dashboards and Pie Charts.
* **🧠 Model Analytics**: A dedicated mathematical integration pane calculating live analytics on your models (See "Model Analytics" below).
* **📈 Live Trading Terminal**: Implements a native embedded Javascript TradingView widget for limitless technical analysis.

---

## 🔬 Deep Model Analytics (Educational Integration)

To prove mathematical efficacy, the system allows data-scientists/beginners to evaluate the structural integrity of the underlying models via the Analytics tab:

1. **Statistical Drift (Z-Test)**
   Calculates the daily percentage returns of a selected stock and runs a One-Sample Z-Test to determine if the stock is merely existing in a *Random Walk*, or if it has a reliable, mathematically significant Bullish/Bearish bias.
2. **Feature Correlation Heatmap**
   Generates a live Plotly Heatmap comparing `Close`, `MA10`, `MA50`, and `Daily Returns`. Identical, highly-correlated data is highlighted to reveal potential feature redundancies.
3. **Regression Matrices (MAE & RMSE)**
   Splits historic sets into rigid `Train` and `Test` states, trains independent trees on the isolated data, and identifies the absolute average error bounds of the AI natively. Results are displayed via a beautiful scatter-plot visualizing `Predicted` vs `Actual` values across a strict diagonal baseline.

---

## 📚 Jupyter / Google Colab Sandbox
Every mathematical model functioning beneath the hood natively in `app.py` has been systematically decoupled and formatted natively into a step-by-step Jupyter Notebook.
Upload **`Pro_Stock_Engine.ipynb`** natively to **Google Colab** to systematically tinker, adjust hyperparameters, or run educational components sequentially.

---

## 🚀 How to Run

### Method 1: Natively (Local)
Ensure you have Python 3.11 installed. Install the exact requirements natively to prevent dependency mismatches with `tensorflow-cpu`.
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

### Method 2: Docker Container (Recommended)
Isolate the entire dependency environment using the internal `Dockerfile`.
```bash
# Build the Environment
docker build -t pro-stock-engine .

# Ignite the Application
docker run -p 8501:8501 pro-stock-engine
```

docker run -p 8888:8888 -v "d:\SRS:/app" pro-stock-engine python -m notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
