import sqlite3
import numpy as np

DB_PATH = "stock_data.db"

def get_latest_available_date():
    """Fetch the most recent available date from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM processed_data")
    latest_date = cursor.fetchone()[0]
    conn.close()
    return latest_date

def get_historical_data():
    """Fetch historical stock prices and sentiment scores from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT date, Close, Sentiment FROM processed_data ORDER BY date ASC")
    data = [{"date": row[0], "price": row[1], "sentiment_score": row[2]} for row in cursor.fetchall()]
    conn.close()
    return data

def get_raw_historical_prices():
    """Fetch raw historical prices from the historical_prices table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if historical_prices table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='historical_prices'")
    if not cursor.fetchone():
        # Fallback to processed_data if historical_prices doesn't exist
        cursor.execute("SELECT date, Open, High, Low, Close, Volume FROM processed_data ORDER BY date ASC")
    else:
        cursor.execute("SELECT Date, Open, High, Low, Close, Volume FROM historical_prices ORDER BY Date ASC")
    
    columns = [desc[0] for desc in cursor.description]
    
    # Convert to dictionary with proper column names
    rows = cursor.fetchall()
    data = []
    for row in rows:
        data_dict = {}
        for i, col_name in enumerate(columns):
            # Standardize column names (lowercase)
            std_col_name = col_name.lower()
            data_dict[std_col_name] = row[i]
        data.append(data_dict)
        
    conn.close()
    return data

N_PAST = 10

def get_data_for_prediction(latest_date):
    """Fetch the last N_PAST days of feature sets for prediction."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT Open, High, Low, Close, Volume, Sentiment, Previous_Close, Moving_Avg_3d, Moving_Avg_7d 
        FROM processed_data 
        WHERE date <= ? 
        ORDER BY date DESC 
        LIMIT ?
    """, (latest_date, N_PAST))
    
    rows = cursor.fetchall()
    conn.close()
    
    if len(rows) == N_PAST:
        # Reshape the data to match the expected input shape (1, N_PAST, number_of_features)
        return np.array(rows).reshape(1, N_PAST, 9)  # Reshape to (1, N_PAST, 9)
    return None