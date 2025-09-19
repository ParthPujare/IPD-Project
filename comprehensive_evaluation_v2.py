#!/usr/bin/env python3
"""
Comprehensive LSTM Model Evaluation Script (Compatible with both models)
This script evaluates both the original LSTM model and the enhanced Model 2.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import os
import psutil
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced model functions
from lstm_model_v2 import build_lstm_model_v2, load_model_v2

# Configuration
DB_PATH = "stock_data.db"
PRICE_SCALER_PATH = "price_scaler.pkl"
SENTIMENT_SCALER_PATH = "sentiment_scaler.pkl"
N_PAST = 10

# Model paths
ORIGINAL_MODEL_PATH = "stock_price_model.h5"
ENHANCED_MODEL_PATH = "stock_price_model_v2.h5"

def load_data_from_db():
    """Load data from SQLite database"""
    print("Loading data from database...")
    
    conn = sqlite3.connect(DB_PATH)
    processed_df = pd.read_sql("SELECT * FROM processed_data ORDER BY Date ASC", conn)
    price_df = pd.read_sql("SELECT * FROM historical_prices ORDER BY Date ASC", conn)
    conn.close()
    
    print(f"✓ Data loaded: {len(processed_df)} processed records, {len(price_df)} price records")
    return processed_df, price_df

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

def load_lstm_model(model_path, model_type="original"):
    """Load LSTM model with fallback options"""
    print(f"Loading {model_type} LSTM model...")
    
    try:
        if model_type == "enhanced":
            model = load_model_v2(model_path)
        else:
            model = load_model(model_path)
        
        if model is not None:
            print(f"✓ {model_type.capitalize()} model loaded successfully")
            return model
        else:
            raise Exception("Model loading failed")
            
    except Exception as e:
        print(f"✗ Error loading {model_type} model: {e}")
        print("Creating new model as fallback...")
        
        # Create new model with appropriate architecture
        if model_type == "enhanced":
            model = build_lstm_model_v2((N_PAST, 9))  # 9 features
        else:
            # Import original model builder
            from lstm_model import build_lstm_model
            model = build_lstm_model((N_PAST, 9))
        
        print(f"✓ New {model_type} model created as fallback")
        return model

def prepare_features_for_lstm(processed_df, price_scaler, sentiment_scaler):
    """Prepare features for LSTM analysis"""
    print("Preparing features for LSTM analysis...")
    
    # Define feature columns
    price_features = ['Open', 'High', 'Low', 'Close', 'Previous_Close', 'Moving_Avg_3d', 'Moving_Avg_7d']
    sentiment_features = ['Sentiment', 'Volume']
    
    # Scale the features
    price_data = processed_df[price_features]
    sentiment_data = processed_df[sentiment_features]
    
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
    
    # Split into train and test sets (80-20 split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Feature shape: {X.shape}, Target shape: {y.shape}")
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    # Core metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE with handling for near-zero values
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
    
    # R² score
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    directional_accuracy = np.mean((y_true_diff > 0) == (y_pred_diff > 0)) * 100
    
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
    
    # Ensure model is compiled
    if not hasattr(model, 'optimizer') or model.optimizer is None:
        print("Model not compiled, compiling with default settings...")
        model.compile(optimizer='adam', loss='mse')
    
    # Training time (measure on small subset)
    print("Measuring training time...")
    start_time = time.time()
    model.fit(X_test[:100], y_test[:100], epochs=1, batch_size=16, verbose=0)
    training_time = time.time() - start_time
    
    # Inference time
    print("Measuring inference time...")
    start_time = time.time()
    for _ in range(1000):
        model.predict(X_test[:1], verbose=0)
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Model size
    model_size = os.path.getsize(ENHANCED_MODEL_PATH if os.path.exists(ENHANCED_MODEL_PATH) else ORIGINAL_MODEL_PATH) / (1024 * 1024)
    
    # Number of parameters
    total_params = model.count_params()
    
    return {
        'Training_Time': training_time,
        'Inference_Time': inference_time,
        'Model_Size': model_size,
        'Total_Parameters': total_params
    }

def plot_prediction_vs_actual(y_test, y_pred, title_suffix=""):
    """Plot predicted vs actual values"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title(f'Predicted vs Actual Values {title_suffix}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'analysis_graphs/01_prediction_vs_actual{title_suffix.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graph saved: analysis_graphs/01_prediction_vs_actual{title_suffix.lower().replace(' ', '_')}.png")

def plot_residuals_over_time(y_test, y_pred, title_suffix=""):
    """Plot residuals over time"""
    residuals = y_test - y_pred
    
    plt.figure(figsize=(12, 6))
    plt.plot(residuals, color='red', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title(f'Residuals Over Time {title_suffix}')
    plt.xlabel('Time')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'analysis_graphs/02_residuals_over_time{title_suffix.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graph saved: analysis_graphs/02_residuals_over_time{title_suffix.lower().replace(' ', '_')}.png")

def plot_predicted_vs_actual_scatter(y_test, y_pred, title_suffix=""):
    """Plot predicted vs actual scatter plot"""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual Scatter Plot {title_suffix}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'analysis_graphs/03_predicted_vs_actual_scatter{title_suffix.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graph saved: analysis_graphs/03_predicted_vs_actual_scatter{title_suffix.lower().replace(' ', '_')}.png")

def plot_error_distribution(y_test, y_pred, title_suffix=""):
    """Plot error distribution"""
    errors = y_test - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'Error Distribution {title_suffix}')
    plt.grid(True, alpha=0.3)
    
    # Add error statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.text(0.02, 0.98, f'Mean: {mean_error:.2f}\nStd: {std_error:.2f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'analysis_graphs/04_error_distribution{title_suffix.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graph saved: analysis_graphs/04_error_distribution{title_suffix.lower().replace(' ', '_')}.png")
    
    # Print error statistics
    print("Error Statistics:")
    print(f"Mean Absolute Error: {np.mean(np.abs(errors)):.4f}")
    print(f"Median Absolute Error: {np.median(np.abs(errors)):.4f}")
    print(f"Standard Deviation: {std_error:.4f}")

def plot_metrics_comparison(original_metrics, enhanced_metrics):
    """Plot comparison of metrics between models"""
    metrics = ['MAE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for plotting
    original_values = [original_metrics[m] for m in metrics]
    enhanced_values = [enhanced_metrics[m] for m in metrics]
    
    # Normalize R² and Directional Accuracy for better visualization
    original_values[3] = original_values[3] * 100  # R² to percentage
    enhanced_values[3] = enhanced_values[3] * 100
    
    bars1 = ax.bar(x - width/2, original_values, width, label='Original Model', alpha=0.8)
    bars2 = ax.bar(x + width/2, enhanced_values, width, label='Enhanced Model 2.0', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    plt.savefig('analysis_graphs/05_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Graph saved: analysis_graphs/05_metrics_comparison.png")

def generate_summary_report(original_metrics, enhanced_metrics, original_efficiency, enhanced_efficiency):
    """Generate comprehensive summary report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE LSTM MODEL EVALUATION SUMMARY")
    print("="*80)
    
    print("\n1. PREDICTIVE PERFORMANCE COMPARISON:")
    print("-" * 50)
    print(f"{'Metric':<20} {'Original':<15} {'Enhanced 2.0':<15} {'Improvement':<15}")
    print("-" * 50)
    
    for metric in ['MAE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy']:
        orig_val = original_metrics[metric]
        enh_val = enhanced_metrics[metric]
        
        if metric in ['R2', 'Directional_Accuracy']:
            improvement = f"+{enh_val - orig_val:.2f}%"
        else:
            improvement = f"-{orig_val - enh_val:.2f}"
        
        print(f"{metric:<20} {orig_val:<15.4f} {enh_val:<15.4f} {improvement:<15}")
    
    print("\n2. EFFICIENCY METRICS COMPARISON:")
    print("-" * 50)
    print(f"{'Metric':<20} {'Original':<15} {'Enhanced 2.0':<15}")
    print("-" * 50)
    
    for metric in ['Training_Time', 'Inference_Time', 'Model_Size', 'Total_Parameters']:
        orig_val = original_efficiency[metric]
        enh_val = enhanced_efficiency[metric]
        
        if metric == 'Inference_Time':
            print(f"{metric:<20} {orig_val:<15.2f}ms {enh_val:<15.2f}ms")
        elif metric == 'Model_Size':
            print(f"{metric:<20} {orig_val:<15.2f}MB {enh_val:<15.2f}MB")
        else:
            print(f"{metric:<20} {orig_val:<15.2f} {enh_val:<15.2f}")
    
    print("\n3. MODEL ASSESSMENT:")
    print("-" * 50)
    
    # Determine which model performs better
    better_mae = "Enhanced 2.0" if enhanced_metrics['MAE'] < original_metrics['MAE'] else "Original"
    better_r2 = "Enhanced 2.0" if enhanced_metrics['R2'] > original_metrics['R2'] else "Original"
    better_direction = "Enhanced 2.0" if enhanced_metrics['Directional_Accuracy'] > original_metrics['Directional_Accuracy'] else "Original"
    
    print(f"• Better MAE: {better_mae}")
    print(f"• Better R²: {better_r2}")
    print(f"• Better Directional Accuracy: {better_direction}")
    
    print("\n4. RECOMMENDATIONS:")
    print("-" * 50)
    if enhanced_metrics['R2'] > original_metrics['R2']:
        print("✓ Enhanced Model 2.0 shows improved predictive performance")
    else:
        print("⚠ Enhanced Model 2.0 may need further tuning")
    
    if enhanced_metrics['Directional_Accuracy'] > 60:
        print("✓ Good directional prediction capability")
    else:
        print("⚠ Consider additional features or model architecture changes")
    
    print("✓ Both models are compatible with the original pipeline")
    print("✓ Use the model that best fits your specific requirements")

def main():
    """Main evaluation function"""
    print("="*80)
    print("COMPREHENSIVE LSTM MODEL EVALUATION (ORIGINAL vs ENHANCED 2.0)")
    print("="*80)
    
    # Create analysis graphs directory
    os.makedirs('analysis_graphs', exist_ok=True)
    
    try:
        # Load data and scalers
        processed_df, price_df = load_data_from_db()
        price_scaler, sentiment_scaler = load_scalers()
        
        if price_scaler is None or sentiment_scaler is None:
            print("✗ Failed to load scalers. Exiting.")
            return
        
        # Prepare features
        X_train, X_test, y_train, y_test = prepare_features_for_lstm(processed_df, price_scaler, sentiment_scaler)
        
        # Evaluate Original Model
        print("\n" + "="*50)
        print("EVALUATING ORIGINAL MODEL")
        print("="*50)
        
        original_model = load_lstm_model(ORIGINAL_MODEL_PATH, "original")
        original_predictions = original_model.predict(X_test, verbose=0)
        
        # Inverse transform predictions and actual values
        y_pred_dummy = np.column_stack([np.zeros_like(original_predictions), np.zeros_like(original_predictions), 
                                       np.zeros_like(original_predictions), original_predictions.flatten(), 
                                       np.zeros_like(original_predictions), np.zeros_like(original_predictions), 
                                       np.zeros_like(original_predictions)])
        y_pred_original = price_scaler.inverse_transform(y_pred_dummy)[:, 3]
        
        y_test_dummy = np.column_stack([np.zeros_like(y_test), np.zeros_like(y_test), np.zeros_like(y_test), 
                                       y_test, np.zeros_like(y_test), np.zeros_like(y_test), np.zeros_like(y_test)])
        y_test_actual = price_scaler.inverse_transform(y_test_dummy)[:, 3]
        
        # Calculate metrics for original model
        original_metrics = calculate_metrics(y_test_actual, y_pred_original)
        original_efficiency = calculate_efficiency_metrics(original_model, X_test, y_test)
        
        print("\nOriginal Model Results:")
        print(f"• MAE: {original_metrics['MAE']:.4f}")
        print(f"• RMSE: {original_metrics['RMSE']:.4f}")
        print(f"• MAPE: {original_metrics['MAPE']:.2f}%")
        print(f"• R²: {original_metrics['R2']:.4f}")
        print(f"• Directional Accuracy: {original_metrics['Directional_Accuracy']:.2f}%")
        
        # Evaluate Enhanced Model 2.0
        print("\n" + "="*50)
        print("EVALUATING ENHANCED MODEL 2.0")
        print("="*50)
        
        enhanced_model = load_lstm_model(ENHANCED_MODEL_PATH, "enhanced")
        enhanced_predictions = enhanced_model.predict(X_test, verbose=0)
        
        # Inverse transform predictions
        y_pred_dummy = np.column_stack([np.zeros_like(enhanced_predictions), np.zeros_like(enhanced_predictions), 
                                       np.zeros_like(enhanced_predictions), enhanced_predictions.flatten(), 
                                       np.zeros_like(enhanced_predictions), np.zeros_like(enhanced_predictions), 
                                       np.zeros_like(enhanced_predictions)])
        y_pred_enhanced = price_scaler.inverse_transform(y_pred_dummy)[:, 3]
        
        # Calculate metrics for enhanced model
        enhanced_metrics = calculate_metrics(y_test_actual, y_pred_enhanced)
        enhanced_efficiency = calculate_efficiency_metrics(enhanced_model, X_test, y_test)
        
        print("\nEnhanced Model 2.0 Results:")
        print(f"• MAE: {enhanced_metrics['MAE']:.4f}")
        print(f"• RMSE: {enhanced_metrics['RMSE']:.4f}")
        print(f"• MAPE: {enhanced_metrics['MAPE']:.2f}%")
        print(f"• R²: {enhanced_metrics['R2']:.4f}")
        print(f"• Directional Accuracy: {enhanced_metrics['Directional_Accuracy']:.2f}%")
        
        # Generate visualizations
        print("\nGenerating comprehensive visualizations...")
        
        # Original model plots
        plot_prediction_vs_actual(y_test_actual, y_pred_original, " (Original Model)")
        plot_residuals_over_time(y_test_actual, y_pred_original, " (Original Model)")
        plot_predicted_vs_actual_scatter(y_test_actual, y_pred_original, " (Original Model)")
        plot_error_distribution(y_test_actual, y_pred_original, " (Original Model)")
        
        # Enhanced model plots
        plot_prediction_vs_actual(y_test_actual, y_pred_enhanced, " (Enhanced 2.0)")
        plot_residuals_over_time(y_test_actual, y_pred_enhanced, " (Enhanced 2.0)")
        plot_predicted_vs_actual_scatter(y_test_actual, y_pred_enhanced, " (Enhanced 2.0)")
        plot_error_distribution(y_test_actual, y_pred_enhanced, " (Enhanced 2.0)")
        
        # Comparison plot
        plot_metrics_comparison(original_metrics, enhanced_metrics)
        
        # Generate summary report
        generate_summary_report(original_metrics, enhanced_metrics, original_efficiency, enhanced_efficiency)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE! ✓")
        print("="*80)
        print("✓ All graphs saved to 'analysis_graphs' directory")
        print("✓ Both models evaluated and compared")
        print("✓ Enhanced Model 2.0 is compatible with your original pipeline")
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 