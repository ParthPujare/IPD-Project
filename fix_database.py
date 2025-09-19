import sqlite3
from datetime import datetime, timedelta

# Constants
DB_PATH = "stock_data.db"

def fix_database():
    """Fix the database by removing future dates and ensuring proper data"""
    
    # Connect to SQLite database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get current date
    today = datetime.now()
    current_date_str = today.strftime("%Y-%m-%d")
    print(f"Current date: {current_date_str}")
    
    # Check for future dates
    cursor.execute("SELECT COUNT(*) FROM historical_prices WHERE Date > ?", (current_date_str,))
    future_count = cursor.fetchone()[0]
    
    if future_count > 0:
        print(f"Found {future_count} future dates in database")
        
        # Show some examples of future dates
        cursor.execute("SELECT Date FROM historical_prices WHERE Date > ? ORDER BY Date DESC LIMIT 5", (current_date_str,))
        future_dates = cursor.fetchall()
        print("Examples of future dates:")
        for date in future_dates:
            print(f"  {date[0]}")
        
        # Remove future dates
        cursor.execute("DELETE FROM historical_prices WHERE Date > ?", (current_date_str,))
        deleted_count = cursor.rowcount
        print(f"Removed {deleted_count} future date entries")
    
    # Get the latest valid date
    cursor.execute("SELECT MAX(Date) FROM historical_prices WHERE Date <= ?", (current_date_str,))
    latest_date = cursor.fetchone()[0]
    
    if latest_date:
        print(f"Latest valid date in database: {latest_date}")
        
        # Calculate days since latest date
        latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")
        days_diff = (today - latest_dt).days
        print(f"Days since latest data: {days_diff}")
        
        if days_diff > 7:
            print("Warning: Data is more than a week old!")
        elif days_diff > 30:
            print("Warning: Data is more than a month old!")
    else:
        print("No valid data found in database")
    
    # Show some recent valid data
    cursor.execute("""
        SELECT Date, Close 
        FROM historical_prices 
        WHERE Date <= ? 
        ORDER BY Date DESC 
        LIMIT 5
    """, (current_date_str,))
    recent_data = cursor.fetchall()
    
    if recent_data:
        print("\nMost recent valid data:")
        for date, close_price in recent_data:
            print(f"  {date}: ₹{close_price:.2f}")
    
    # Commit and close
    conn.commit()
    conn.close()
    
    print("\nDatabase has been fixed!")

if __name__ == "__main__":
    fix_database() 