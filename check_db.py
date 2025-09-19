import sqlite3
import pandas as pd

# Check database structure
conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()

# Get table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print("Tables in database:", tables)

# Check each table structure
for table in tables:
    print(f"\nTable: {table}")
    cursor.execute(f"PRAGMA table_info({table})")
    columns = cursor.fetchall()
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    
    # Show sample data
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"  Rows: {count}")
    
    if count > 0:
        cursor.execute(f"SELECT * FROM {table} LIMIT 3")
        sample = cursor.fetchall()
        print(f"  Sample data: {sample[:2]}")

conn.close() 