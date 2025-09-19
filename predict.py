import os
import subprocess
from datetime import datetime, timedelta
import numpy as np
import pickle
import sqlite3
from database_utils import get_latest_available_date, get_data_for_prediction

# Disable oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def get_latest_historical_date():
    """Get the latest date from the historical_prices table"""
    conn = sqlite3.connect("stock_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(Date) FROM historical_prices")
    latest_date = cursor.fetchone()[0]
    conn.close()
    return latest_date

def is_trading_day(date):
    """Check if the given date is a trading day (not a weekend)"""
    # 5 = Saturday, 6 = Sunday
    return date.weekday() < 5

def get_next_trading_day(date):
    """Get the next trading day after the given date"""
    next_day = date + timedelta(days=1)
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    return next_day

# Step 1: Update the stock prices
print("\n📊 Updating stock prices...")
subprocess.run(["python", "update_price.py"], check=True)

# Get the latest date from historical_prices
latest_historical_date = get_latest_historical_date()
print(f"✅ Latest historical date: {latest_historical_date}")

# Step 2: Update news and sentiment
print("\n📰 Fetching and analyzing news...")
try:
    subprocess.run(["python", "fetch_news.py"], check=True)
    subprocess.run(["python", "sentiment_analysis.py"], check=True)
except Exception as e:
    print(f"⚠️ Warning: Error in news processing: {e}")
    print("⚠️ Continuing with prediction without latest news...")

# Step 3: Process the data
print("\n🔄 Processing data...")
try:
    subprocess.run(["python", "getdataready.py"], check=True)
except Exception as e:
    print(f"❌ Error processing data: {e}")
    exit(1)

# Step 4: Get the latest date from processed_data
processed_date = get_latest_available_date()
if processed_date is None:
    print("❌ No available data in the database. Cannot proceed with prediction.")
    exit(1)

# Check if processed data is up to date
if processed_date != latest_historical_date:
    print(f"⚠️ Warning: Processed data ({processed_date}) is not up to date with latest stock data ({latest_historical_date})")
    print("⚠️ This may affect prediction accuracy")

# Step 5: Import prediction modules after data is ready
from lstm_model import load_model, predict_next_day_price

# Step 6: Get data for prediction
latest_date = datetime.strptime(processed_date, '%Y-%m-%d')
print(f"\n📈 Using data up to {latest_date.strftime('%Y-%m-%d')} for prediction")

# Load model and scalers
model = load_model("stock_price_model.h5")

# Load the price scaler
with open("price_scaler.pkl", "rb") as price_file:
    price_scaler = pickle.load(price_file)

# Fetch relevant data for prediction
data = get_data_for_prediction(processed_date)
if data is None:
    print("❌ Insufficient data for prediction")
    exit(1)

# Make prediction
predicted_scaled_price = predict_next_day_price(model, data)

# Inverse transform the predicted scaled price
# Create a dummy array for the other features
dummy_features = np.zeros((1, 8))  # Assuming 8 price features in the scaler
dummy_features[0, 7] = predicted_scaled_price  # Set the prediction as the last value (Next_Day_Close)

# Inverse transform to get actual price
predicted_price = price_scaler.inverse_transform(dummy_features)[0, 7]

# Calculate next trading day (skip weekends)
next_trading_day = get_next_trading_day(latest_date)

# Debug information
print(f"Debug info:")
print(f"- Latest processed date: {latest_date.strftime('%Y-%m-%d')} (weekday: {latest_date.weekday()})")
print(f"- Next day would be: {(latest_date + timedelta(days=1)).strftime('%Y-%m-%d')} (weekday: {(latest_date + timedelta(days=1)).weekday()})")
print(f"- Next trading day: {next_trading_day.strftime('%Y-%m-%d')} (weekday: {next_trading_day.weekday()})")

# Print the prediction result
print(f"\n🔮 PREDICTION RESULT:")
print(f"📅 Date: {next_trading_day.strftime('%Y-%m-%d')} ({['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][next_trading_day.weekday()]})")
print(f"💰 Predicted Closing Price: ₹{predicted_price:.2f}")