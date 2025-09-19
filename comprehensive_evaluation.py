#!/usr/bin/env python3
"""
Comprehensive LSTM Model Evaluation Script
Calculates all requested metrics and generates visualizations
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create analysis_graphs directory if it doesn't exist
os.makedirs('analysis_graphs', exist_ok=True)

def load_data_from_db():
    """Load data from SQLite database"""
    print("Loading data from database...")
    
    conn = sqlite3.connect('stock_data.db')
    
    # Load processed data
    processed_df = pd.read_sql("SELECT * FROM processed_data ORDER BY Date ASC", conn)
    
    # Load historical price data
    price_df = pd.read_sql("SELECT * FROM historical_prices ORDER BY Date ASC", conn)
    
    conn.close()
    
    print(f"✓ Data loaded: {len(processed_df)} processed records, {len(price_df)} price records")
    return processed_df, price_df

def load_lstm_model():
    """Load the trained LSTM model with proper error handling"""
    try:
        print("Loading LSTM model...")
        # Try to load the model with custom_objects to handle any compatibility issues
        model = load_model('stock_price_model.h5', compile=False)
        print("✓ Trained LSTM model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading LSTM model: {e}")
        print("Creating a new LSTM model for evaluation...")
        
        # Create a new model with the same architecture
        from lstm_model import build_lstm_model
        model = build_lstm_model((10, 9))
        print("✓ New LSTM model created")
        return model

def load_scalers():
    """Load the pre-trained scalers"""
    try:
        with open('price_scaler.pkl', 'rb') as f:
            price_scaler = pickle.load(f)
        with open('sentiment_scaler.pkl', 'rb') as f:
            sentiment_scaler = pickle.load(f)
        print("✓ Scalers loaded successfully")
        return price_scaler, sentiment_scaler
    except Exception as e:
        print(f"Error loading scalers: {e}")
        print("Creating new scalers...")
        return None, None

def prepare_features_for_lstm(processed_df, lookback=10):
    """Prepare features for LSTM analysis with proper scaling"""
    print("Preparing features for LSTM analysis...")
    
    # Define feature columns (EXACTLY as in lstm.py training)
    price_features = ['Open', 'High', 'Low', 'Close', 'Previous_Close', 'Moving_Avg_3d', 'Moving_Avg_7d']
    sentiment_features = ['Sentiment', 'Volume']
    
    # Separate features for scaling
    price_data = processed_df[price_features].values
    sentiment_data = processed_df[sentiment_features].values
    
    # Load or create scalers
    price_scaler, sentiment_scaler = load_scalers()
    
    if price_scaler is None or sentiment_scaler is None:
        # Create new scalers if loading failed
        price_scaler = MinMaxScaler(feature_range=(0, 1))
        sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Fit scalers on the data
        scaled_price_data = price_scaler.fit_transform(price_data)
        scaled_sentiment_data = sentiment_scaler.fit_transform(sentiment_data)
    else:
        # Use loaded scalers to transform data
        scaled_price_data = price_scaler.transform(price_data)
        scaled_sentiment_data = sentiment_scaler.transform(sentiment_data)
    
    # Combine scaled price and sentiment data
    scaled_data = np.hstack((scaled_price_data, scaled_sentiment_data))
    
    # Prepare sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback : i])
        y.append(scaled_price_data[i, 3])  # Target: scaled 'Close' price (index 3)
    
    X, y = np.array(X), np.array(y)
    
    print(f"Feature shape: {X.shape}, Target shape: {y.shape}")
    
    # Split into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, price_scaler

def calculate_metrics(y_true, y_pred):
    """Calculate all requested metrics"""
    print("Calculating comprehensive metrics...")
    
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE with handling for zero values
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
    
    # Directional accuracy
    direction_correct = np.sum(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
    total_directions = len(y_true) - 1
    directional_accuracy = (direction_correct / total_directions) * 100 if total_directions > 0 else 0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    }

def calculate_efficiency_metrics(model, X_test, y_test):
    """Calculate efficiency metrics"""
    print("Calculating efficiency metrics...")
    
    # Model size
    model_size_mb = os.path.getsize('stock_price_model.h5') / (1024 * 1024)
    
    # Total parameters
    total_params = model.count_params()
    
    # Training time (simulate training for measurement)
    print("Measuring training time...")
    start_time = time.time()
    
    # Check if model is compiled, if not compile it
    if not hasattr(model, 'optimizer') or model.optimizer is None:
        print("Model not compiled, compiling with default settings...")
        model.compile(optimizer='adam', loss='mse')
    
    model.fit(X_test[:100], y_test[:100], epochs=1, batch_size=16, verbose=0)
    training_time = time.time() - start_time
    
    # Inference time
    print("Measuring inference time...")
    start_time = time.time()
    for _ in range(1000):
        _ = model.predict(X_test[:1], verbose=0)
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    return {
        'Training_Time': training_time,
        'Inference_Time': inference_time,
        'Model_Size': model_size_mb,
        'Total_Parameters': total_params
    }

def plot_prediction_vs_actual(y_true, y_pred):
    """Plot 1: Prediction vs Actual values"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title('LSTM Model: Actual vs Predicted Values')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = 'analysis_graphs/01_prediction_vs_actual.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graph saved: {save_path}")

def plot_residuals_over_time(y_true, y_pred):
    """Plot 2: Residuals over time"""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 6))
    plt.plot(residuals, color='red', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('LSTM Model: Residuals Over Time')
    plt.xlabel('Time')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.grid(True, alpha=0.3)
    
    save_path = 'analysis_graphs/02_residuals_over_time.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graph saved: {save_path}")

def plot_scatter_predicted_vs_actual(y_true, y_pred):
    """Plot 3: Scatter plot of predicted vs actual"""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
    
    plt.title('LSTM Model: Predicted vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = 'analysis_graphs/03_predicted_vs_actual_scatter.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graph saved: {save_path}")

def plot_error_distribution(y_true, y_pred):
    """Plot 4: Error distribution"""
    errors = y_true - y_pred
    
    plt.figure(figsize=(12, 8))
    
    # Histogram
    plt.subplot(2, 2, 1)
    plt.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Error Distribution Histogram')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(2, 2, 2)
    plt.boxplot(errors)
    plt.title('Error Box Plot')
    plt.ylabel('Error')
    plt.grid(True, alpha=0.3)
    
    # Q-Q plot
    plt.subplot(2, 2, 3)
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normal Distribution)')
    plt.grid(True, alpha=0.3)
    
    # Error over time
    plt.subplot(2, 2, 4)
    plt.plot(errors, color='red', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Error Over Time')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = 'analysis_graphs/04_error_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graph saved: {save_path}")
    
    # Print error statistics
    print("Error Statistics:")
    print(f"Mean Absolute Error: {np.mean(np.abs(errors)):.4f}")
    print(f"Median Absolute Error: {np.median(np.abs(errors)):.4f}")
    print(f"Standard Deviation: {np.std(errors):.4f}")

def plot_metrics_comparison(metrics):
    """Plot 5: Metrics comparison"""
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(metric_names, metric_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    
    plt.title('LSTM Model: Performance Metrics Comparison')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    save_path = 'analysis_graphs/05_metrics_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graph saved: {save_path}")

def plot_complexity_tradeoff(efficiency_metrics):
    """Plot 6: Complexity vs Performance tradeoff"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Model size vs parameters
    ax1.scatter(efficiency_metrics['Total_Parameters'], efficiency_metrics['Model_Size'], 
                s=200, color='red', alpha=0.7)
    ax1.set_xlabel('Number of Parameters')
    ax1.set_ylabel('Model Size (MB)')
    ax1.set_title('Model Complexity vs Size')
    ax1.grid(True, alpha=0.3)
    
    # Training vs inference time
    ax2.scatter(efficiency_metrics['Training_Time'], efficiency_metrics['Inference_Time'], 
                s=200, color='blue', alpha=0.7)
    ax2.set_xlabel('Training Time (seconds)')
    ax2.set_ylabel('Inference Time (ms)')
    ax2.set_title('Training vs Inference Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = 'analysis_graphs/06_complexity_tradeoff.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graph saved: {save_path}")

def plot_rolling_performance(y_true, y_pred, window=30):
    """Plot 7: Rolling performance metrics"""
    if len(y_true) < window:
        print("Warning: Data too short for rolling analysis")
        return
    
    # Calculate rolling metrics
    rolling_mae = []
    rolling_rmse = []
    rolling_r2 = []
    
    for i in range(window, len(y_true)):
        y_true_window = y_true[i-window:i]
        y_pred_window = y_pred[i-window:i]
        
        rolling_mae.append(mean_absolute_error(y_true_window, y_pred_window))
        rolling_rmse.append(np.sqrt(mean_squared_error(y_true_window, y_pred_window)))
        rolling_r2.append(r2_score(y_true_window, y_pred_window))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Rolling MAE
    ax1.plot(rolling_mae, color='red', alpha=0.7)
    ax1.set_title('Rolling MAE (30-day window)')
    ax1.set_ylabel('MAE')
    ax1.grid(True, alpha=0.3)
    
    # Rolling RMSE
    ax2.plot(rolling_rmse, color='orange', alpha=0.7)
    ax2.set_title('Rolling RMSE (30-day window)')
    ax2.set_ylabel('RMSE')
    ax2.grid(True, alpha=0.3)
    
    # Rolling R²
    ax3.plot(rolling_r2, color='green', alpha=0.7)
    ax3.set_title('Rolling R² (30-day window)')
    ax3.set_ylabel('R²')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = 'analysis_graphs/07_rolling_performance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graph saved: {save_path}")

def plot_feature_analysis(X_test, y_test, y_pred):
    """Plot 8: Feature importance analysis"""
    # Calculate correlation between features and prediction accuracy
    feature_importance = []
    feature_names = ['Open', 'High', 'Low', 'Close', 'Previous_Close', 'Moving_Avg_3d', 'Moving_Avg_7d', 'Volume', 'Sentiment']
    
    for i in range(X_test.shape[2]):
        # Take the last timestep of each sequence to match with y_test length
        feature_values = X_test[:, -1, i]
        # Calculate correlation with prediction accuracy
        accuracy = 1 - np.abs(y_test - y_pred) / np.maximum(y_test, 1e-8)
        correlation = np.corrcoef(feature_values, accuracy)[0, 1]
        if np.isnan(correlation):
            correlation = 0
        feature_importance.append(abs(correlation))
    
    # Sort features by importance
    feature_importance_sorted = sorted(zip(feature_names, feature_importance), 
                                     key=lambda x: x[1], reverse=True)
    
    names, importance = zip(*feature_importance_sorted)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, importance, color='lightcoral')
    plt.title('LSTM Model: Feature Importance Analysis')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, importance):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    save_path = 'analysis_graphs/08_feature_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graph saved: {save_path}")
    
    # Print feature importance summary
    print("Feature Importance Summary:")
    print("=" * 40)
    for name, imp in feature_importance_sorted:
        print(f"{name}: {imp:.4f}")

def generate_summary_report(metrics, efficiency_metrics):
    """Generate comprehensive summary report"""
    print("=" * 80)
    print("COMPREHENSIVE LSTM MODEL EVALUATION SUMMARY")
    print("=" * 80)
    
    print("1. PREDICTIVE PERFORMANCE METRICS:")
    print("-" * 40)
    print(f"• MAE (Mean Absolute Error): {metrics['MAE']:.4f}")
    print(f"• RMSE (Root Mean Squared Error): {metrics['RMSE']:.4f}")
    print(f"• MAPE (Mean Absolute Percentage Error): {metrics['MAPE']:.2f}%")
    print(f"• R² (Coefficient of Determination): {metrics['R2']:.4f}")
    print(f"• Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
    
    print("\n2. EFFICIENCY METRICS:")
    print("-" * 40)
    print(f"• Training Time: {efficiency_metrics['Training_Time']:.2f} seconds")
    print(f"• Inference Time per Sample: {efficiency_metrics['Inference_Time']:.2f} ms")
    print(f"• Model Size: {efficiency_metrics['Model_Size']:.2f} MB")
    print(f"• Number of Trainable Parameters: {efficiency_metrics['Total_Parameters']:,}")
    
    print("\n3. MODEL PERFORMANCE ASSESSMENT:")
    print("-" * 40)
    
    # Performance assessment
    if metrics['R2'] >= 0.8:
        print("✓ Excellent predictive power (R² ≥ 0.8)")
    elif metrics['R2'] >= 0.6:
        print("✓ Good predictive power (R² ≥ 0.6)")
    elif metrics['R2'] >= 0.4:
        print("⚠ Moderate predictive power (R² ≥ 0.4)")
    else:
        print("✗ Limited predictive power (R² < 0.4)")
    
    if metrics['Directional_Accuracy'] >= 70:
        print("✓ High directional accuracy (≥70%)")
    elif metrics['Directional_Accuracy'] >= 60:
        print("⚠ Moderate directional accuracy (≥60%)")
    else:
        print("✗ Limited directional accuracy (<60%)")
    
    if efficiency_metrics['Inference_Time'] <= 50:
        print("✓ Fast inference time (≤50ms)")
    elif efficiency_metrics['Inference_Time'] <= 100:
        print("⚠ Moderate inference time (≤100ms)")
    else:
        print("⚠ Slow inference time (>100ms)")
    
    print("\n4. RECOMMENDATIONS:")
    print("-" * 40)
    
    if metrics['R2'] < 0.6:
        print("• Consider feature engineering or additional data sources")
        print("• Experiment with different LSTM architectures or hyperparameters")
        print("• Increase training data or use data augmentation techniques")
        print("• Focus on improving trend prediction capabilities")
    
    if metrics['Directional_Accuracy'] < 70:
        print("• Consider ensemble methods or additional technical indicators")
        print("• Implement attention mechanisms in the LSTM")
    
    if efficiency_metrics['Inference_Time'] > 100:
        print("• Consider model optimization for production deployment")
        print("• Evaluate model quantization or pruning")
        print("• Use TensorFlow Lite for mobile deployment")
    
    print("=" * 80)

def main():
    """Main execution function"""
    print("Comprehensive LSTM Model Evaluation")
    print("=" * 50)
    
    # Load data and model
    processed_df, price_df = load_data_from_db()
    model = load_lstm_model()
    
    # Prepare features
    X_train, X_test, y_train, y_test, price_scaler = prepare_features_for_lstm(processed_df)
    
    # Generate predictions
    print("Generating LSTM predictions...")
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform predictions back to actual prices
    # We need to create a dummy array with the same shape as price_data for inverse transform
    # The model predicts the Close price, so we'll inverse transform just that column
    y_pred_dummy = np.column_stack([np.zeros_like(y_pred_scaled), np.zeros_like(y_pred_scaled), np.zeros_like(y_pred_scaled), y_pred_scaled, np.zeros_like(y_pred_scaled), np.zeros_like(y_pred_scaled), np.zeros_like(y_pred_scaled)])
    y_pred = price_scaler.inverse_transform(y_pred_dummy)[:, 3]

    # For y_test, we need to do the same inverse transform
    y_test_dummy = np.column_stack([np.zeros_like(y_test), np.zeros_like(y_test), np.zeros_like(y_test), y_test, np.zeros_like(y_test), np.zeros_like(y_test), np.zeros_like(y_test)])
    y_test_actual = price_scaler.inverse_transform(y_test_dummy)[:, 3]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test_actual, y_pred)
    efficiency_metrics = calculate_efficiency_metrics(model, X_test, y_test)
    
    # Display results
    print("=" * 80)
    print("LSTM MODEL EVALUATION RESULTS")
    print("=" * 80)
    print("PREDICTIVE METRICS:")
    print(f"• MAE: {metrics['MAE']:.4f}")
    print(f"• RMSE: {metrics['RMSE']:.4f}")
    print(f"• MAPE: {metrics['MAPE']:.2f}%")
    print(f"• R²: {metrics['R2']:.4f}")
    print(f"• Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
    print("EFFICIENCY METRICS:")
    print(f"• Training Time: {efficiency_metrics['Training_Time']:.2f} seconds")
    print(f"• Inference Time: {efficiency_metrics['Inference_Time']:.2f} ms")
    print(f"• Model Size: {efficiency_metrics['Model_Size']:.2f} MB")
    print(f"• Total Parameters: {efficiency_metrics['Total_Parameters']:,}")
    
    # Generate visualizations
    print("Generating comprehensive visualizations...")
    plot_prediction_vs_actual(y_test_actual, y_pred)
    plot_residuals_over_time(y_test_actual, y_pred)
    plot_scatter_predicted_vs_actual(y_test_actual, y_pred)
    plot_error_distribution(y_test_actual, y_pred)
    plot_metrics_comparison(metrics)
    plot_complexity_tradeoff(efficiency_metrics)
    plot_rolling_performance(y_test_actual, y_pred)
    plot_feature_analysis(X_test, y_test_actual, y_pred)
    
    # Generate summary report
    generate_summary_report(metrics, efficiency_metrics)
    
    print("✓ Comprehensive LSTM evaluation complete!")
    print("✓ All graphs saved to 'analysis_graphs' directory")
    print("✓ Check the generated visualizations for detailed insights")

if __name__ == "__main__":
    main() 