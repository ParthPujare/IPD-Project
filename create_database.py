import time
import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

STOCK_SYMBOL = "ADANIGREEN.NS"

# Get current date and set end date to tomorrow to ensure we get latest data
current_date = datetime.now()
end_date = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")

# Retry loop to avoid rate limits
for attempt in range(5):  # Retry up to 5 times
    try:
        stock_data = yf.download(STOCK_SYMBOL, start="2020-01-01", end=end_date)
        break  # If download is successful, break out of the loop
    except yf.YFRateLimitError:
        print(f"Rate limit hit. Retrying in {2**attempt} seconds...")
        time.sleep(2**attempt)  # Exponential backoff

# Check if data was retrieved successfully
if stock_data.empty:
    raise Exception("Failed to retrieve stock data. Please try again later.")

stock_data.reset_index(inplace=True)

# Fix column name mismatch by dropping 'Adj Close'
stock_data.drop(columns=['Adj Close'], inplace=True)  # Drop extra column
stock_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
stock_data['Date'] = stock_data['Date'].astype(str)

# Database setup
conn = sqlite3.connect("stock_data.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS historical_prices")
cursor.execute("DROP TABLE IF EXISTS news_articles")

cursor.execute("""
    CREATE TABLE historical_prices (
        Date TEXT PRIMARY KEY,
        Open REAL,
        High REAL,
        Low REAL,
        Close REAL,
        Volume INTEGER
    )
""")

cursor.execute("""
    CREATE TABLE news_articles (
        Date TEXT,
        Title TEXT,
        Source TEXT,
        Sentiment REAL,
        Link TEXT
    )
""")

stock_data.to_sql("historical_prices", conn, if_exists="append", index=False)

conn.close()
print("Historical stock data and news database initialized successfully!")
