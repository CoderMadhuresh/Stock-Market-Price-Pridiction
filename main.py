import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# CONFIGURATION

stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
start_date = "2015-01-01"
end_date = "2024-01-01"

print("Fetching data...")

# 1. DATA FETCHING & CLEANING

df = yf.download(stocks, start=start_date, end=end_date)
df = df.stack(level=1).reset_index()
df.columns = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
df = df.sort_values(["Symbol", "Date"])

print("Data fetched successfully!\n")

print("Cleaned dataset preview:")
print(df.head(), "\n")

# 2. EXPLORATORY DATA ANALYSIS

plt.figure(figsize=(12, 6))
for stock in stocks:
    sub = df[df["Symbol"] == stock]
    plt.plot(sub["Date"], sub["Close"], label=stock)

plt.title("Stock Close Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for stock in stocks:
    sub = df[df["Symbol"] == stock]
    plt.plot(sub["Date"], sub["Volume"], label=stock)

plt.title("Stock Trading Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.legend()
plt.tight_layout()
plt.show()


# 3. FEATURE ENGINEERING

def create_features(stock_df):
    stock_df["Return"] = stock_df["Close"].pct_change()
    stock_df["MA7"] = stock_df["Close"].rolling(window=7).mean()
    stock_df["MA21"] = stock_df["Close"].rolling(window=21).mean()
    stock_df["Vol_Change"] = stock_df["Volume"].pct_change()
    stock_df.dropna(inplace=True)
    return stock_df


df_features = df.groupby("Symbol").apply(create_features).reset_index(drop=True)

print("Feature-engineered dataset preview:")
print(df_features.head(), "\n")


# 4. ARIMA MODELING FOR AAPL

symbol = "AAPL"
print(f"Training ARIMA model for: {symbol}")

df_arima = df_features[df_features["Symbol"] == symbol][["Date", "Close"]].copy()

# Set Date as index WITHOUT forcing frequency
df_arima["Date"] = pd.to_datetime(df_arima["Date"])
df_arima.set_index("Date", inplace=True)

# Train-test split
train = df_arima.iloc[:-30]
test = df_arima.iloc[-30:]

# Fit ARIMA(5,1,2)
model = ARIMA(train["Close"], order=(5, 1, 2))
model_fit = model.fit()

forecast = model_fit.forecast(steps=30)
forecast.index = test.index

# Evaluation
arima_rmse = sqrt(mean_squared_error(test["Close"], forecast))
arima_mae = mean_absolute_error(test["Close"], forecast)

print(f"\nARIMA Evaluation for {symbol}:")
print(f"RMSE: {arima_rmse:.4f}")
print(f"MAE : {arima_mae:.4f}")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(train["Close"], label="Train")
plt.plot(test["Close"], label="Test")
plt.plot(forecast, label="Forecast")
plt.title(f"{symbol} ARIMA Forecast")
plt.legend()
plt.tight_layout()
plt.show()


# 5. GRADIENT BOOSTING MODEL

print("\nTraining Gradient Boosting Model...")

gb_results = []

for stock in stocks:
    stock_df = df_features[df_features["Symbol"] == stock].copy()

    # Input features
    X = stock_df[["Return", "MA7", "MA21", "Vol_Change"]]
    y = stock_df["Close"]

    # Train-test split
    split = int(0.8 * len(stock_df))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    gb = GradientBoostingRegressor()
    gb.fit(X_train, y_train)

    preds = gb.predict(X_test)

    rmse = sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    gb_results.append([stock, rmse, mae])
    print(f"{stock} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# Output summary
gb_results_df = pd.DataFrame(gb_results, columns=["Stock", "RMSE", "MAE"])
print("\nGradient Boosting Performance:")
print(gb_results_df)


# 6. MODEL COMPARISON

print("\n--- MODEL COMPARISON SUMMARY ---")
print(f"ARIMA on {symbol} -> RMSE={arima_rmse:.4f}, MAE={arima_mae:.4f}")
print(gb_results_df)
