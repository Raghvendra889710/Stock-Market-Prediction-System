import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# UI Setup
st.set_page_config(page_title="Stock Market Predictor", layout="wide")
st.title("📈 Stock Market Prediction System")
st.write("Analyze historical data and predict the next day's closing price using Machine Learning.")

# Sidebar for User Input
st.sidebar.header("User Settings")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, GOOGL)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Function to load data (cached so it doesn't redownload on every click)
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

# Fetch Data
with st.spinner("Fetching data from Yahoo Finance..."):
    df = load_data(ticker_symbol, start_date, end_date)

if df.empty:
    st.error("No data found for this ticker. Please check the symbol and try again.")
else:
    st.subheader(f"Raw Data Extract for {ticker_symbol.upper()}")
    st.write(df.tail())

    # Preprocessing & Feature Engineering
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    df['Target_Next_Close'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    # 1. Plot Historical Trends
    st.subheader("Historical Stock Price & Moving Averages")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax1.plot(df.index, df['MA_50'], label='50-Day MA', color='orange', linestyle='--')
    ax1.plot(df.index, df['MA_200'], label='200-Day MA', color='red', linestyle='--')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    st.pyplot(fig1)

    # Model Training
    st.subheader("Machine Learning Prediction (Random Forest)")
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_50', 'MA_200']
    X = df[features]
    y = df['Target_Next_Close']

    # Chronological Train-Test Split (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluation Metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
    col2.metric("Root Mean Squared Error (RMSE)", f"${rmse:.2f}")
    col3.metric("R² Score", f"{r2:.4f}")

    # 2. Plot Actual vs Predicted
    st.subheader("Model Evaluation: Actual vs. Predicted (Test Data)")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(y_test.index, y_test.values, label="Actual Price", color='blue')
    ax2.plot(y_test.index, predictions, label="Predicted Price", color='red', alpha=0.7)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()
    st.pyplot(fig2)

    # Final Prediction
    st.success(f"Based on the latest data, the model predicts the next closing price to be: **${predictions[-1]:.2f}**")
