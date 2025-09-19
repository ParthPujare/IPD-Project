import sqlite3
from datetime import datetime, timedelta

# Connect to the database
DB_PATH = "stock_data.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get current date and set cutoff date to tomorrow to ensure we don't accidentally delete current data
current_date = datetime.now()
cutoff_date = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")

# Delete entries from the cutoff date onwards (future dates)
cursor.execute("DELETE FROM historical_prices WHERE Date >= ?", (cutoff_date,))

# Commit changes and close connection
conn.commit()
conn.close()

print(f"Deleted stock data from {cutoff_date} onwards (future dates).")
