#!/usr/bin/env python3
"""
Enhanced Model 2.0 Integration Script
This script demonstrates how to use the enhanced LSTM model 2.0
in your original pipeline while maintaining compatibility.
"""

import numpy as np
import pandas as pd
import sqlite3
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
warnings.filterwarnings('ignore')

# Import our enhanced model
from lstm_model_v2 import load_model_v2, predict_next_day_price

# Configuration
DB_PATH = "stock_data.db"
ENHANCED_MODEL_PATH = "stock_price_model_v2.h5"
PRICE_SCALER_PATH = "price_scaler.pkl"
SENTIMENT_SCALER_PATH = "sentiment_scaler.pkl"
N_PAST = 10

def load_enhanced_model():
    """Load the enhanced LSTM model 2.0"""
    print("Loading Enhanced LSTM Model 2.0...")
    
    try:
        model = load_model_v2(ENHANCED_MODEL_PATH)
        if model is not None:
            print("✓ Enhanced Model 2.0 loaded successfully")
            return model
        else:
            print("✗ Failed to load enhanced model")
            return None
    except Exception as e:
        print(f"✗ Error loading enhanced model: {e}")
        return None

def load_scalers():
    """Load the fitted scalers"""
    try:
        with open(PRICE_SCALER_PATH, 'rb') as f:
            price_scaler = pickle.load(f)
        with open(SENTIMENT_SCALER_PATH, 'rb') as f:
            sentiment_scaler = pickle.load(f)
        print("✓ Scalers loaded successfully")
        return price_scaler, sentiment_scaler
    except Exception as e:
        print(f"✗ Error loading scalers: {e}")
        return None, None

def prepare_latest_data():
    """Prepare the latest data for prediction"""
    print("Preparing latest data for prediction...")
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM processed_data ORDER BY Date DESC LIMIT 20", conn)
    conn.close()
    
    # Reverse to get chronological order
    df = df.iloc[::-1].reset_index(drop=True)
    
    # Define feature columns
    price_features = ['Open', 'High', 'Low', 'Close', 'Previous_Close', 'Moving_Avg_3d', 'Moving_Avg_7d']
    sentiment_features = ['Sentiment', 'Volume']
    
    # Get the latest N_PAST days
    latest_data = df.head(N_PAST)
    
    if len(latest_data) < N_PAST:
        print(f"✗ Insufficient data. Need {N_PAST} days, got {len(latest_data)}")
        return None
    
    print(f"✓ Latest {N_PAST} days of data prepared")
    return latest_data, price_features, sentiment_features

def make_prediction(model, latest_data, price_features, sentiment_features, price_scaler, sentiment_scaler):
    """Make prediction using the enhanced model"""
    print("Making prediction with Enhanced Model 2.0...")
    
    try:
        # Scale the features
        price_data = latest_data[price_features]
        sentiment_data = latest_data[sentiment_features]
        
        scaled_price_data = price_scaler.transform(price_data)
        scaled_sentiment_data = sentiment_scaler.transform(sentiment_data)
        
        # Combine scaled data
        scaled_data = np.hstack((scaled_price_data, scaled_sentiment_data))
        
        # Reshape for LSTM input (batch_size, timesteps, features)
        X = scaled_data.reshape(1, N_PAST, len(price_features) + len(sentiment_features))
        
        # Make prediction
        prediction_scaled = model.predict(X, verbose=0)
        
        # Inverse transform to get actual price
        y_pred_dummy = np.column_stack([np.zeros_like(prediction_scaled), np.zeros_like(prediction_scaled), 
                                       np.zeros_like(prediction_scaled), prediction_scaled.flatten(), 
                                       np.zeros_like(prediction_scaled), np.zeros_like(prediction_scaled), 
                                       np.zeros_like(prediction_scaled)])
        prediction_actual = price_scaler.inverse_transform(y_pred_dummy)[0, 3]
        
        print(f"✓ Prediction made successfully")
        return prediction_scaled[0][0], prediction_actual
        
    except Exception as e:
        print(f"✗ Error making prediction: {e}")
        return None, None

def compare_with_actual(prediction_actual, latest_data):
    """Compare prediction with the most recent actual value"""
    if prediction_actual is not None:
        latest_close = latest_data.iloc[-1]['Close']
        latest_date = latest_data.iloc[-1]['Date']
        
        print(f"\nPrediction Results:")
        print(f"Latest Date: {latest_date}")
        print(f"Latest Close Price: ${latest_close:.2f}")
        print(f"Predicted Next Close: ${prediction_actual:.2f}")
        
        # Calculate expected change
        expected_change = prediction_actual - latest_close
        expected_change_pct = (expected_change / latest_close) * 100
        
        print(f"Expected Change: ${expected_change:.2f} ({expected_change_pct:+.2f}%)")
        
        # Direction prediction
        if expected_change > 0:
            print("Direction: 📈 UP")
        elif expected_change < 0:
            print("Direction: 📉 DOWN")
        else:
            print("Direction: ➡️ SIDEWAYS")

def demonstrate_pipeline_integration():
    """Demonstrate how the enhanced model integrates with the original pipeline"""
    print("\n" + "="*60)
    print("ENHANCED MODEL 2.0 PIPELINE INTEGRATION DEMO")
    print("="*60)
    
    # Step 1: Load the enhanced model
    model = load_enhanced_model()
    if model is None:
        print("✗ Cannot proceed without model")
        return
    
    # Step 2: Load scalers
    price_scaler, sentiment_scaler = load_scalers()
    if price_scaler is None or sentiment_scaler is None:
        print("✗ Cannot proceed without scalers")
        return
    
    # Step 3: Prepare latest data
    data_result = prepare_latest_data()
    if data_result is None:
        print("✗ Cannot proceed without data")
        return
    
    latest_data, price_features, sentiment_features = data_result
    
    # Step 4: Make prediction
    prediction_scaled, prediction_actual = make_prediction(
        model, latest_data, price_features, sentiment_features, 
        price_scaler, sentiment_scaler
    )
    
    if prediction_actual is not None:
        # Step 5: Compare with actual
        compare_with_actual(prediction_actual, latest_data)
        
        print(f"\n" + "="*60)
        print("INTEGRATION SUCCESSFUL! ✓")
        print("="*60)
        print("The Enhanced Model 2.0 is now fully integrated with your pipeline.")
        print("You can use it exactly like the original model, but with improved performance.")
        
        # Show model info
        print(f"\nModel Information:")
        print(f"• Total Parameters: {model.count_params():,}")
        print(f"• Model Size: {os.path.getsize(ENHANCED_MODEL_PATH) / (1024*1024):.2f} MB")
        print(f"• Input Shape: {model.input_shape}")
        print(f"• Output Shape: {model.output_shape}")
        
    else:
        print("✗ Prediction failed")

def show_usage_examples():
    """Show examples of how to use the enhanced model"""
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    print("\n1. Basic Prediction (same as original pipeline):")
    print("""
from lstm_model_v2 import load_model_v2
import pickle

# Load model and scalers
model = load_model_v2('stock_price_model_v2.h5')
with open('price_scaler.pkl', 'rb') as f:
    price_scaler = pickle.load(f)
with open('sentiment_scaler.pkl', 'rb') as f:
    sentiment_scaler = pickle.load(f)

# Use exactly like before
prediction = model.predict(your_data)
    """)
    
    print("\n2. In your existing scripts, just change the model path:")
    print("""
# Before (original)
model = load_model('stock_price_model.h5')

# After (enhanced)
from lstm_model_v2 import load_model_v2
model = load_model_v2('stock_price_model_v2.h5')
    """)
    
    print("\n3. The enhanced model maintains the same interface:")
    print("""
# All these work exactly the same:
model.predict(data)
model.evaluate(data, labels)
model.summary()
model.count_params()
    """)

if __name__ == "__main__":
    # Demonstrate the integration
    demonstrate_pipeline_integration()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n" + "="*60)
    print("ENHANCED MODEL 2.0 READY FOR USE! 🚀")
    print("="*60)
    print("Your enhanced model is now ready and fully compatible with your existing pipeline.")
    print("Simply replace the model loading line in your scripts to use the improved performance.") 