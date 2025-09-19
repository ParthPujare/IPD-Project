import yfinance as yf
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import time
import random

# Constants
DB_PATH = "stock_data.db"
SYMBOL = "ADANIGREEN.NS"  # .NS suffix for NSE stocks

def fetch_stock_data(start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            # Create a Ticker object
            stock = yf.Ticker(SYMBOL)
            
            # Add a small delay to avoid rate limiting
            time.sleep(random.uniform(1, 3))
            
            # Fetch historical data
            df = stock.history(start=start_date, end=end_date)
            
            if not df.empty:
                return df
            return None
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} - Error fetching data: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # Progressive backoff
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Could not fetch data.")
                return None
    return None

# Connect to SQLite database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS historical_prices (
    Date TEXT PRIMARY KEY,
    Open REAL,
    High REAL,
    Low REAL,
    Close REAL,
    Volume INTEGER
)
""")

# Get current date
today = datetime.now()
current_date_str = today.strftime("%Y-%m-%d")
print(f"Current date: {current_date_str}")

# Clean up any future dates or dates beyond current date
cursor.execute("DELETE FROM historical_prices WHERE Date > ?", (current_date_str,))
deleted_rows = cursor.rowcount
if deleted_rows > 0:
    print(f"Cleaned up {deleted_rows} future date entries from the database")

# Get the latest valid date from database (excluding future dates)
cursor.execute("SELECT MAX(Date) FROM historical_prices WHERE Date <= ?", (current_date_str,))
latest_date = cursor.fetchone()[0]

# Set date range
if latest_date:
    print(f"Latest valid date in database: {latest_date}")
    start_date = datetime.strptime(latest_date, "%Y-%m-%d") + timedelta(days=1)
    
    # If start_date is in the future or today, fetch last 30 days to ensure we have recent data
    if start_date >= today:
        print("Latest date is current or future. Fetching last 30 days of data...")
        start_date = today - timedelta(days=30)
else:
    print("No existing data found. Fetching from 2 years ago...")
    start_date = today - timedelta(days=730)  # 2 years back

end_date = today
print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Fetch and store data
df = fetch_stock_data(start_date, end_date)

if df is not None and not df.empty:
    records_added = 0
    for index, row in df.iterrows():
        # Skip future dates
        if index.date() > today.date():
            continue
            
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO historical_prices (Date, Open, High, Low, Close, Volume)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                index.strftime("%Y-%m-%d"),
                row['Open'],
                row['High'],
                row['Low'],
                row['Close'],
                row['Volume']
            ))
            records_added += 1
        except Exception as e:
            print(f"Error inserting record for {index.strftime('%Y-%m-%d')}: {e}")
    
    if records_added > 0:
        print(f"Successfully added {records_added} new records to the database")
        
        # Print the latest prices
        cursor.execute("""
            SELECT Date, Open, High, Low, Close, Volume 
            FROM historical_prices 
            WHERE Date <= ?
            ORDER BY Date DESC 
            LIMIT 1
        """, (current_date_str,))
        latest_record = cursor.fetchone()
        
        if latest_record:
            print("\nLatest stock data:")
            print(f"Date: {latest_record[0]}")
            print(f"Open: {latest_record[1]:.2f}")
            print(f"High: {latest_record[2]:.2f}")
            print(f"Low: {latest_record[3]:.2f}")
            print(f"Close: {latest_record[4]:.2f}")
            print(f"Volume: {latest_record[5]:,}")
    else:
        print("No new data was added (possibly due to market closure or all data being up to date)")
else:
    print("No new data available for the specified period")

# Commit and close connection
conn.commit()
conn.close()

print("\nNote: Data has been updated successfully.")
print("The database now contains historical prices for ADANI Green Energy.")
print("You can use this data for your LSTM model training.")
