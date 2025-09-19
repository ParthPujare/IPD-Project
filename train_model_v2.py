#!/usr/bin/env python3
"""
Enhanced LSTM Model Training Script (Model 2.0)
This script trains an improved LSTM model with fine-tuned parameters
for better performance metrics.
"""

import sqlite3
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time
import os

# Import our enhanced model
from lstm_model_v2 import build_lstm_model_v2, build_lstm_model_v2_compact, get_callbacks, save_model_v2

# Configuration
DB_PATH = "stock_data.db"
MODEL_PATH_V2 = "stock_price_model_v2.h5"
PRICE_SCALER_PATH = "price_scaler.pkl"
SENTIMENT_SCALER_PATH = "sentiment_scaler.pkl"
N_PAST = 10
RANDOM_STATE = 42

def load_and_prepare_data():
    """Load data from database and prepare features"""
    print("Loading data from database...")
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM processed_data ORDER BY Date ASC", conn)
    conn.close()
    
    print(f"✓ Loaded {len(df)} records")
    
    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Define feature columns
    price_features = ['Open', 'High', 'Low', 'Close', 'Previous_Close', 'Moving_Avg_3d', 'Moving_Avg_7d']
    sentiment_features = ['Sentiment', 'Volume']
    
    # Separate features for scaling
    price_data = df[price_features]
    sentiment_data = df[sentiment_features]
    
    # Initialize and fit scalers
    print("Fitting scalers...")
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_price_data = price_scaler.fit_transform(price_data)
    scaled_sentiment_data = sentiment_scaler.fit_transform(sentiment_data)
    
    # Save the scalers
    with open(PRICE_SCALER_PATH, "wb") as f:
        pickle.dump(price_scaler, f)
    with open(SENTIMENT_SCALER_PATH, "wb") as f:
        pickle.dump(sentiment_scaler, f)
    
    print("✓ Scalers fitted and saved")
    
    # Combine scaled data
    scaled_data = np.hstack((scaled_price_data, scaled_sentiment_data))
    
    # Prepare sequences
    X, y = [], []
    for i in range(N_PAST, len(scaled_data)):
        X.append(scaled_data[i - N_PAST : i])
        y.append(scaled_price_data[i, 3])  # Target: Scaled 'Close' price
    
    X, y = np.array(X), np.array(y)
    
    print(f"✓ Prepared sequences: X shape {X.shape}, y shape {y.shape}")
    
    return X, y, price_scaler, sentiment_scaler

def train_enhanced_model(X, y, model_type="full"):
    """Train the enhanced LSTM model with TUNED hyperparameters"""
    print(f"\nTraining enhanced LSTM model ({model_type} version) with TUNED hyperparameters...")
    
    # Optimized data splitting strategy
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE  # Reduced test size for more training data
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.4, random_state=RANDOM_STATE  # Adjusted validation/test split
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Build model
    if model_type == "full":
        model = build_lstm_model_v2((X.shape[1], X.shape[2]))
    else:
        model = build_lstm_model_v2_compact((X.shape[1], X.shape[2]))
    
    print(f"✓ Model built with {model.count_params():,} parameters")
    
    # Get callbacks
    callbacks = get_callbacks()
    
    # Train model with TUNED hyperparameters
    print("\nStarting training with TUNED hyperparameters...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=150,  # Increased epochs for better convergence
        batch_size=16,  # Reduced batch size for better generalization
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
        shuffle=True  # Enable shuffling for better training
    )
    
    training_time = time.time() - start_time
    print(f"✓ Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("\nEvaluating model performance...")
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    
    # Make predictions for additional metrics
    y_pred = model.predict(X_test, verbose=0)
    
    # Calculate R²
    ss_res = np.sum((y_test - y_pred.flatten()) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Calculate directional accuracy
    y_test_diff = np.diff(y_test)
    y_pred_diff = np.diff(y_pred.flatten())
    directional_accuracy = np.mean((y_test_diff > 0) == (y_pred_diff > 0)) * 100
    
    print(f"R² Score: {r2:.4f}")
    print(f"Directional Accuracy: {directional_accuracy:.2f}%")
    
    return model, history, (X_test, y_test, y_pred), training_time

def save_model_and_summary(model, training_time, model_path):
    """Save the model and print summary"""
    # Save model
    save_model_v2(model, model_path)
    
    # Print model summary
    print("\n" + "="*60)
    print("ENHANCED LSTM MODEL 2.0 SUMMARY")
    print("="*60)
    print(f"Model saved as: {model_path}")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    # Print model architecture
    print("\nModel Architecture:")
    model.summary()

def main():
    """Main training function"""
    print("="*60)
    print("ENHANCED LSTM MODEL 2.0 TRAINING")
    print("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    
    try:
        # Load and prepare data
        X, y, price_scaler, sentiment_scaler = load_and_prepare_data()
        
        # Train full model
        model, history, test_data, training_time = train_enhanced_model(X, y, "full")
        
        # Save model and summary
        save_model_and_summary(model, training_time, MODEL_PATH_V2)
        
        # Save test data for evaluation
        X_test, y_test, y_pred = test_data
        np.savez('test_data_v2.npz', X_test=X_test, y_test=y_test, y_pred=y_pred)
        print("\n✓ Test data saved for evaluation")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE! ✓")
        print("="*60)
        print("You can now use this enhanced model in your pipeline by:")
        print("1. Loading 'stock_price_model_v2.h5'")
        print("2. Using the same scalers (price_scaler.pkl, sentiment_scaler.pkl)")
        print("3. Running comprehensive_evaluation.py with the new model")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 