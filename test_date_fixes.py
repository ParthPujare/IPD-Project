#!/usr/bin/env python3
"""
Test script to verify that the date fixes are working correctly.
This script will test the dynamic date logic in the update scripts.
"""

import sqlite3
from datetime import datetime, timedelta
import subprocess
import sys

def test_dynamic_dates():
    """Test that the dynamic date logic is working correctly"""
    
    print("Testing dynamic date fixes...")
    
    # Test 1: Check current date logic
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    two_years_ago = today - timedelta(days=730)
    
    print(f"Current date: {today.strftime('%Y-%m-%d')}")
    print(f"Tomorrow: {tomorrow.strftime('%Y-%m-%d')}")
    print(f"Two years ago: {two_years_ago.strftime('%Y-%m-%d')}")
    
    # Test 2: Check database for any hardcoded date issues
    try:
        conn = sqlite3.connect("stock_data.db")
        cursor = conn.cursor()
        
        # Check if database exists and has data
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='historical_prices'")
        if cursor.fetchone():
            print("✓ Database exists with historical_prices table")
            
            # Check latest date in database
            cursor.execute("SELECT MAX(Date) FROM historical_prices")
            latest_date = cursor.fetchone()[0]
            
            if latest_date:
                print(f"✓ Latest date in database: {latest_date}")
                
                # Check if latest date is reasonable (not too old)
                latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")
                days_diff = (today - latest_dt).days
                
                if days_diff <= 30:
                    print(f"✓ Data is recent (within {days_diff} days)")
                else:
                    print(f"⚠ Data is {days_diff} days old - may need updating")
                
                # Check for any future dates
                cursor.execute("SELECT COUNT(*) FROM historical_prices WHERE Date > ?", (today.strftime("%Y-%m-%d"),))
                future_count = cursor.fetchone()[0]
                
                if future_count == 0:
                    print("✓ No future dates found in database")
                else:
                    print(f"⚠ Found {future_count} future dates in database")
            else:
                print("⚠ No data found in database")
        else:
            print("⚠ Database exists but no historical_prices table found")
        
        conn.close()
        
    except Exception as e:
        print(f"⚠ Error checking database: {e}")
    
    # Test 3: Test the update scripts
    print("\nTesting update scripts...")
    
    try:
        # Test update_price.py
        print("Testing update_price.py...")
        result = subprocess.run([sys.executable, "update_price.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ update_price.py ran successfully")
            if "Successfully added" in result.stdout or "No new data was added" in result.stdout:
                print("✓ update_price.py logic working correctly")
            else:
                print("⚠ update_price.py output unclear")
        else:
            print(f"⚠ update_price.py failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⚠ update_price.py timed out")
    except Exception as e:
        print(f"⚠ Error testing update_price.py: {e}")
    
    print("\nDate fix verification complete!")
    print("If all tests pass, your data should now update beyond April 2024.")

if __name__ == "__main__":
    test_dynamic_dates() 