# Stock Market Prediction System 📈

## Overview
A machine learning project that fetches real-time historical stock data and predicts future closing prices. Built to demonstrate time-series data handling, exploratory data analysis (EDA), and predictive modeling.

## Features
* **Live Data Extraction:** Uses `yfinance` to pull historical data for any stock ticker (e.g., AAPL, TSLA).
* **Feature Engineering:** Calculates 50-day and 200-day Moving Averages to analyze market trends.
* **Predictive Modeling:** Uses a Random Forest Regressor to predict the next day's closing price.
* **Performance Evaluation:** Evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), RMSE, and R² Score.

## Technologies Used
* **Languages & Libraries:** Python, Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn
* **Data Sourcing:** yfinance API
