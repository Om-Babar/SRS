import yfinance as yf
import pandas as pd

print("Fetching Reliance Data via yfinance...")
df = yf.download("RELIANCE.NS", period="6mo", progress=False)

# MultiIndex handling
if isinstance(df.columns, pd.MultiIndex):
    if "RELIANCE.NS" in df.columns.levels[1]: df = df.xs("RELIANCE.NS", level=1, axis=1) 
    elif "RELIANCE.NS" in df.columns.levels[0]: df = df["RELIANCE.NS"]

c_data = df['Close']
if isinstance(c_data, pd.DataFrame): c_data = c_data.iloc[:, 0]
clean_df = pd.DataFrame({'Close': c_data})

# Adding engineering features just like app.py
clean_df['MA10'] = clean_df['Close'].rolling(window=10).mean()
clean_df['MA50'] = clean_df['Close'].rolling(window=50).mean()

clean_df.dropna(inplace=True)
clean_df['Target'] = clean_df['Close'].shift(-1)

# Rounding for cleanliness and stripping down to the last 15 rows for an easy viewing experience
clean_df = round(clean_df, 2)
demo_df = clean_df.tail(15)

# Save overriding the fake mock file
demo_df.to_csv("c:/Users/om/OneDrive/Attachments/Desktop/SPS/sample_stock_dataset.csv")
print("Successfully generated sample_stock_dataset.csv with REAL yfinance data.")
