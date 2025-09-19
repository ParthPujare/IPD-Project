import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime

DB_PATH = "stock_data.db"
N_PAST = 10  # Number of past days to include as features

# Connect to the database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print(" Starting data processing...")

# Create the 'processed_data' table if it does not exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS processed_data (
        Date TEXT PRIMARY KEY,
        Open REAL,
        High REAL,
        Low REAL,
        Close REAL,
        Volume INTEGER,
        Sentiment REAL,
        Previous_Close REAL,
        Moving_Avg_3d REAL,
        Moving_Avg_7d REAL,
        Next_Day_Close REAL
    )
""")
print(" Table 'processed_data' created successfully!")

# Check the latest available dates in both tables
cursor.execute("SELECT MAX(Date) FROM historical_prices")
latest_price_date = cursor.fetchone()[0]

cursor.execute("SELECT MAX(Date) FROM processed_data")
latest_processed_date = cursor.fetchone()[0]

print(f" Latest date in historical_prices: {latest_price_date}")
print(f" Latest date in processed_data: {latest_processed_date if latest_processed_date else 'No data'}")

# Fetch historical stock prices - ensure we get all available data
df_prices = pd.read_sql("SELECT * FROM historical_prices ORDER BY Date ASC", conn)
print(f" Fetched {len(df_prices)} historical price records")

# Fetch sentiment scores from news articles
try:
    df_news = pd.read_sql("""
        SELECT Date, AVG(Sentiment) as Sentiment 
        FROM news_articles 
        GROUP BY Date
    """, conn)
    print(f" Fetched {len(df_news)} sentiment records")
except:
    print(" No sentiment data available. Using neutral sentiment.")
    # Create empty sentiment DataFrame with neutral sentiment (0.5)
    df_news = pd.DataFrame(columns=["Date", "Sentiment"])

# Merge stock prices and sentiment scores
df_merged = pd.merge(df_prices, df_news, on="Date", how="left")

# Fill missing sentiment scores with neutral value (0.5)
df_merged["Sentiment"].fillna(0.5, inplace=True)

# Convert 'Date' column to datetime format
df_merged["Date"] = pd.to_datetime(df_merged["Date"])

# Sort data by date
df_merged = df_merged.sort_values(by="Date")

# Create new columns for additional features
df_merged["Previous_Close"] = df_merged["Close"].shift(1)
df_merged["Moving_Avg_3d"] = df_merged["Close"].rolling(window=3).mean()
df_merged["Moving_Avg_7d"] = df_merged["Close"].rolling(window=7).mean()
df_merged["Next_Day_Close"] = df_merged["Close"].shift(-1)

# For the most recent day, estimate Next_Day_Close as 2*Today's Close - Yesterday's Close
if not df_merged.iloc[-1]["Next_Day_Close"]:
    last_idx = df_merged.index[-1]
    if last_idx > 0:
        today_close = df_merged.loc[last_idx, "Close"]
        yesterday_close = df_merged.loc[last_idx-1, "Close"] if last_idx > 0 else today_close
        estimated_next_close = 2 * today_close - yesterday_close
        df_merged.loc[last_idx, "Next_Day_Close"] = estimated_next_close
        print(f" Estimated next day close for {df_merged.iloc[-1]['Date'].strftime('%Y-%m-%d')}")

# Remove rows with missing values, except for the last row's Next_Day_Close
df_merged = df_merged.dropna(subset=["Previous_Close", "Moving_Avg_3d", "Moving_Avg_7d"])

# Normalize price-related and sentiment-related features separately
price_features = ['Open', 'High', 'Low', 'Close', 'Previous_Close', 'Moving_Avg_3d', 'Moving_Avg_7d', 'Next_Day_Close']
sentiment_features = ['Sentiment', 'Volume']

# Save original values before scaling
original_df = df_merged.copy()

price_scaler = MinMaxScaler()
sentiment_scaler = MinMaxScaler()

df_merged[price_features] = price_scaler.fit_transform(df_merged[price_features])
df_merged[sentiment_features] = sentiment_scaler.fit_transform(df_merged[sentiment_features])

# Save scalers for future use
with open("price_scaler.pkl", "wb") as price_file:
    pickle.dump(price_scaler, price_file)

with open("sentiment_scaler.pkl", "wb") as sentiment_file:
    pickle.dump(sentiment_scaler, sentiment_file)

# Convert 'Date' back to string before storing in SQLite
df_merged["Date"] = df_merged["Date"].dt.strftime("%Y-%m-%d")

# Store the processed data into the 'processed_data' table
original_df["Date"] = original_df["Date"].dt.strftime("%Y-%m-%d")
original_df.to_sql("processed_data", conn, if_exists="replace", index=False)

print(f" 'processed_data' table updated successfully with {len(original_df)} records!")
print(f" Latest date processed: {original_df['Date'].max()}")

conn.close()
