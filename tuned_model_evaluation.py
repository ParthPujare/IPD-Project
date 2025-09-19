#!/usr/bin/env python3
"""
Tuned Model 2.0 Performance Evaluation
This script evaluates the performance of the hyperparameter-tuned Model 2.0
and provides only the necessary performance metrics.
"""

import sqlite3
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import os

# Import our tuned model
from lstm_model_v2 import build_lstm_model_v2, load_model_v2

# Configuration
DB_PATH = "stock_data.db"
TUNED_MODEL_PATH = "stock_price_model_v2.h5"
PRICE_SCALER_PATH = "price_scaler.pkl"
SENTIMENT_SCALER_PATH = "sentiment_scaler.pkl"
N_PAST = 10
RANDOM_STATE = 42

def load_data_and_scalers():
    """Load data and scalers"""
    print("Loading data and scalers...")
    
    # Load scalers
    try:
        with open(PRICE_SCALER_PATH, "rb") as f:
            price_scaler = pickle.load(f)
        with open(SENTIMENT_SCALER_PATH, "rb") as f:
            sentiment_scaler = pickle.load(f)
        print("✓ Scalers loaded successfully")
    except FileNotFoundError:
        print("✗ Scaler files not found. Please run training first.")
        return None, None, None
    
    # Load data from database
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM processed_data ORDER BY Date ASC", conn)
    conn.close()
    
    print(f"✓ Loaded {len(df)} records")
    return df, price_scaler, sentiment_scaler

def prepare_test_data(df, price_scaler, sentiment_scaler):
    """Prepare test data for evaluation"""
    print("Preparing test data...")
    
    # Define feature columns
    price_features = ['Open', 'High', 'Low', 'Close', 'Previous_Close', 'Moving_Avg_3d', 'Moving_Avg_7d']
    sentiment_features = ['Sentiment', 'Volume']
    
    # Separate features for scaling
    price_data = df[price_features]
    sentiment_data = df[sentiment_features]
    
    # Scale the data
    scaled_price_data = price_scaler.transform(price_data)
    scaled_sentiment_data = sentiment_scaler.transform(sentiment_data)
    
    # Combine scaled data
    scaled_data = np.hstack((scaled_price_data, scaled_sentiment_data))
    
    # Prepare sequences
    X, y = [], []
    for i in range(N_PAST, len(scaled_data)):
        X.append(scaled_data[i - N_PAST : i])
        y.append(scaled_price_data[i, 3])  # Target: Scaled 'Close' price
    
    X, y = np.array(X), np.array(y)
    
    # Split into train/test (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"✓ Test set prepared: {X_test.shape[0]} samples")
    return X_test, y_test, price_scaler

def load_or_build_tuned_model(input_shape):
    """Load existing tuned model or build new one"""
    print("Loading tuned Model 2.0...")
    
    # Try to load existing model
    model = load_model_v2(TUNED_MODEL_PATH)
    
    if model is None:
        print("Building new tuned model...")
        model = build_lstm_model_v2(input_shape)
        print("✓ New tuned model built")
    else:
        print("✓ Existing tuned model loaded")
    
    return model

def calculate_performance_metrics(y_true, y_pred, price_scaler):
    """Calculate all necessary performance metrics"""
    print("Calculating performance metrics...")
    
    # Inverse transform predictions and actual values to original scale
    # Create dummy arrays with correct shape for inverse transform
    y_pred_reshaped = np.zeros((len(y_pred), 7))  # 7 price features
    y_pred_reshaped[:, 3] = y_pred.flatten()  # Set Close price column
    
    y_true_reshaped = np.zeros((len(y_true), 7))
    y_true_reshaped[:, 3] = y_true
    
    # Inverse transform
    y_pred_actual = price_scaler.inverse_transform(y_pred_reshaped)[:, 3]
    y_true_actual = price_scaler.inverse_transform(y_true_reshaped)[:, 3]
    
    # Calculate metrics
    mae = mean_absolute_error(y_true_actual, y_pred_actual)
    mse = mean_squared_error(y_true_actual, y_pred_actual)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE (with protection against division by zero)
    mape = np.mean(np.abs((y_true_actual - y_pred_actual) / np.where(y_true_actual != 0, y_true_actual, 1)))
    
    # Calculate R²
    r2 = r2_score(y_true_actual, y_pred_actual)
    
    # Calculate directional accuracy
    y_true_diff = np.diff(y_true_actual)
    y_pred_diff = np.diff(y_pred_actual)
    directional_accuracy = np.mean((y_true_diff > 0) == (y_pred_diff > 0)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy,
        'y_true_actual': y_true_actual,
        'y_pred_actual': y_pred_actual
    }

def calculate_efficiency_metrics(model, X_test, training_time=None):
    """Calculate efficiency metrics"""
    print("Calculating efficiency metrics...")
    
    # Model size
    model_size_mb = os.path.getsize(TUNED_MODEL_PATH) / (1024 * 1024)
    
    # Number of trainable parameters
    trainable_params = model.count_params()
    
    # Inference time (average over 1000 predictions)
    start_time = time.time()
    for _ in range(1000):
        _ = model.predict(X_test[:1], verbose=0)
    inference_time = (time.time() - start_time) / 1000 * 1000  # Convert to milliseconds
    
    return {
        'Model_Size_MB': model_size_mb,
        'Trainable_Parameters': trainable_params,
        'Inference_Time_ms': inference_time,
        'Training_Time_s': training_time if training_time else "N/A"
    }

def main():
    """Main evaluation function"""
    print("="*60)
    print("TUNED MODEL 2.0 PERFORMANCE EVALUATION")
    print("="*60)
    
    try:
        # Load data and scalers
        df, price_scaler, sentiment_scaler = load_data_and_scalers()
        if df is None:
            return
        
        # Prepare test data
        X_test, y_test, price_scaler = prepare_test_data(df, price_scaler, sentiment_scaler)
        
        # Load or build tuned model
        model = load_or_build_tuned_model((X_test.shape[1], X_test.shape[2]))
        
        # Make predictions
        print("Making predictions...")
        y_pred = model.predict(X_test, verbose=0)
        
        # Calculate performance metrics
        performance_metrics = calculate_performance_metrics(y_test, y_pred, price_scaler)
        
        # Calculate efficiency metrics
        efficiency_metrics = calculate_efficiency_metrics(model, X_test)
        
        # Display results
        print("\n" + "="*60)
        print("TUNED MODEL 2.0 PERFORMANCE METRICS")
        print("="*60)
        
        print(f"\nPREDICTIVE PERFORMANCE:")
        print(f"• MAE: {performance_metrics['MAE']:.4f}")
        print(f"• RMSE: {performance_metrics['RMSE']:.4f}")
        print(f"• MAPE: {performance_metrics['MAPE']:.4f}")
        print(f"• R²: {performance_metrics['R2']:.4f}")
        print(f"• Directional Accuracy: {performance_metrics['Directional_Accuracy']:.2f}%")
        
        print(f"\nEFFICIENCY METRICS:")
        print(f"• Model Size: {efficiency_metrics['Model_Size_MB']:.2f} MB")
        print(f"• Trainable Parameters: {efficiency_metrics['Trainable_Parameters']:,}")
        print(f"• Inference Time: {efficiency_metrics['Inference_Time_ms']:.2f} ms/sample")
        if efficiency_metrics['Training_Time_s'] != "N/A":
            print(f"• Training Time: {efficiency_metrics['Training_Time_s']:.2f} seconds")
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 