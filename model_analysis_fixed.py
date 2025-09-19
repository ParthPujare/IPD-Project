#!/usr/bin/env python3
"""
LSTM Model Analysis Script
Generates comprehensive metrics and visualizations for the LSTM stock prediction model
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import os
import pickle
import warnings

warnings.filterwarnings('ignore')

# Create output directory
os.makedirs('analysis_graphs', exist_ok=True)

def load_model_and_data():
    """Load the trained LSTM model and data"""
    print("Loading model and data...")
    
    # Load model
    model_path = "stock_price_model.h5"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return None, None, None
    
    try:
        model = tf.keras.models.load_model(model_path)
        print("✓ Model loaded successfully")
    except:
        print("Failed to load model")
        return None, None, None
    
    # Load data
    try:
        conn = sqlite3.connect('stock_data.db')
        df = pd.read_sql("SELECT * FROM processed_data ORDER BY Date ASC", conn)
        df['Date'] = pd.to_datetime(df['Date'])
        conn.close()
        print(f"✓ Data loaded: {len(df)} records")
        return model, df, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def prepare_features(df, lookback=10):
    """Prepare features for LSTM model"""
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment', 
                       'Previous_Close', 'Moving_Avg_3d', 'Moving_Avg_7d']
    
    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(df[feature_columns].iloc[i-lookback:i].values)
        y.append(df['Next_Day_Close'].iloc[i])
    
    return np.array(X), np.array(y)

def calculate_metrics(y_true, y_pred):
    """Calculate all core predictive metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
    
    # Directional accuracy
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    directional_accuracy = np.mean(direction_true == direction_pred) * 100
    
    return {
        'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 
        'R2': r2, 'Directional_Accuracy': directional_accuracy
    }

def calculate_efficiency_metrics(model, X_test, y_test):
    """Calculate efficiency and resource metrics"""
    # Model size
    model_size_mb = os.path.getsize('stock_price_model.h5') / (1024 * 1024)
    
    # Parameters
    total_params = model.count_params()
    
    # Inference time
    start_time = time.time()
    for _ in range(1000):
        _ = model.predict(X_test[:1], verbose=0)
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / 1000 * 1000
    
    return {
        'Model_Size_MB': model_size_mb,
        'Total_Parameters': total_params,
        'Avg_Inference_Time_ms': avg_inference_time
    }

def create_metrics_table(metrics, efficiency_metrics):
    """Create comprehensive metrics table"""
    print("\n" + "="*80)
    print("COMPREHENSIVE METRICS SUMMARY TABLE")
    print("="*80)
    
    print("\n1. CORE PREDICTIVE METRICS:")
    print("-" * 50)
    print(f"{'Metric':<25} {'Value':<15} {'Description'}")
    print("-" * 50)
    print(f"{'MAE':<25} {metrics['MAE']:<15.4f} {'Mean Absolute Error'}")
    print(f"{'RMSE':<25} {metrics['RMSE']:<15.4f} {'Root Mean Squared Error'}")
    print(f"{'MAPE':<25} {metrics['MAPE']:<15.2f}% {'Mean Absolute Percentage Error'}")
    print(f"{'R²':<25} {metrics['R2']:<15.4f} {'Coefficient of Determination'}")
    print(f"{'Directional Accuracy':<25} {metrics['Directional_Accuracy']:<15.2f}% {'Price Direction Prediction'}")
    
    print("\n2. EFFICIENCY & RESOURCE METRICS:")
    print("-" * 50)
    print(f"{'Metric':<25} {'Value':<15} {'Description'}")
    print("-" * 50)
    print(f"{'Model Size':<25} {efficiency_metrics['Model_Size_MB']:<15.2f} MB {'Disk size of saved model'}")
    print(f"{'Total Parameters':<25} {efficiency_metrics['Total_Parameters']:<15,} {'Total model parameters'}")
    print(f"{'Inference Time':<25} {efficiency_metrics['Avg_Inference_Time_ms']:<15.2f} ms {'Average inference time'}")
    
    print("\n" + "="*80)

def plot_prediction_vs_actual(y_test, y_pred, dates, save_path='analysis_graphs/01_prediction_vs_actual.png'):
    """Plot actual vs predicted values over time"""
    plt.figure(figsize=(15, 8))
    plt.plot(dates, y_test, label='Actual', linewidth=2, color='blue')
    plt.plot(dates, y_pred, label='Predicted', linewidth=2, linestyle='--', color='red')
    plt.title('Actual vs Predicted Stock Prices Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (₹)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Graph saved: {save_path}")

def plot_residuals(y_test, y_pred, dates, save_path='analysis_graphs/02_residuals_over_time.png'):
    """Plot prediction residuals over time"""
    residuals = y_pred - y_test
    plt.figure(figsize=(15, 8))
    plt.plot(dates, residuals, color='purple', linewidth=1, alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.5, label='Zero Line')
    plt.title('Prediction Residuals Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Residual (Predicted - Actual)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Graph saved: {save_path}")

def plot_scatter_predicted_vs_actual(y_test, y_pred, save_path='analysis_graphs/03_predicted_vs_actual_scatter.png'):
    """Scatter plot of predicted vs actual values"""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue', s=50)
    
    # Identity line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    r2 = r2_score(y_test, y_pred)
    plt.title(f'Predicted vs Actual Stock Prices (R² = {r2:.4f})', fontsize=16, fontweight='bold')
    plt.xlabel('Actual Price (₹)', fontsize=12)
    plt.ylabel('Predicted Price (₹)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
             fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Graph saved: {save_path}")

def plot_error_distribution(y_test, y_pred, save_path='analysis_graphs/04_error_distribution.png'):
    """Plot error distribution histogram and boxplot"""
    absolute_errors = np.abs(y_pred - y_test)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    ax1.hist(absolute_errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Absolute Errors', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Absolute Error (₹)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    mean_error = np.mean(absolute_errors)
    median_error = np.median(absolute_errors)
    ax1.axvline(mean_error, color='red', linestyle='--', label=f'Mean: {mean_error:.2f}')
    ax1.axvline(median_error, color='green', linestyle='--', label=f'Median: {median_error:.2f}')
    ax1.legend()
    
    # Boxplot
    ax2.boxplot(absolute_errors, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax2.set_title('Boxplot of Absolute Errors', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Absolute Error (₹)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Graph saved: {save_path}")
    
    print(f"\nError Statistics:")
    print(f"Mean Absolute Error: {mean_error:.4f}")
    print(f"Median Absolute Error: {median_error:.4f}")
    print(f"Standard Deviation: {np.std(absolute_errors):.4f}")

def plot_metrics_comparison(metrics, efficiency_metrics, save_path='analysis_graphs/05_metrics_comparison.png'):
    """Bar chart comparing core metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Error metrics
    error_metrics = ['MAE', 'RMSE']
    error_values = [metrics[m] for m in error_metrics]
    bars1 = ax1.bar(error_metrics, error_values, color=['skyblue', 'lightcoral'])
    ax1.set_title('Error Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Error Value', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    for bar, value in zip(bars1, error_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Accuracy metrics
    acc_metrics = ['R2', 'Directional_Accuracy']
    acc_values = [metrics[m] for m in acc_metrics]
    bars2 = ax2.bar(acc_metrics, acc_values, color=['lightgreen', 'gold'])
    ax2.set_title('Accuracy Metrics', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy Value', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for bar, value in zip(bars2, acc_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # MAPE
    ax3.bar(['MAPE'], [metrics['MAPE']], color='orange')
    ax3.set_title('MAPE', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Percentage (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.text(0, metrics['MAPE'] + 0.1, f'{metrics["MAPE"]:.2f}%', 
              ha='center', va='bottom', fontweight='bold')
    
    # Efficiency metrics
    eff_metrics = ['Model_Size_MB', 'Total_Parameters']
    eff_values = [efficiency_metrics[m] for m in eff_metrics]
    bars4 = ax4.bar(eff_metrics, eff_values, color=['plum', 'lightsteelblue'])
    ax4.set_title('Model Complexity', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Value', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    for bar, value in zip(bars4, eff_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Graph saved: {save_path}")

def plot_complexity_tradeoff(metrics, efficiency_metrics, save_path='analysis_graphs/06_complexity_tradeoff.png'):
    """Scatter plot showing trade-off between complexity and performance"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Parameters vs RMSE
    ax1.scatter(efficiency_metrics['Total_Parameters'], metrics['RMSE'], 
                s=200, alpha=0.7, color='blue', edgecolors='black')
    ax1.set_xlabel('Number of Parameters', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('Model Complexity vs Predictive Accuracy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax1.annotate(f'Current Model\nRMSE: {metrics["RMSE"]:.4f}\nParams: {efficiency_metrics["Total_Parameters"]:,}',
                 xy=(efficiency_metrics['Total_Parameters'], metrics['RMSE']),
                 xytext=(efficiency_metrics['Total_Parameters']*0.8, metrics['RMSE']*1.2),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                 fontsize=10)
    
    # Model Size vs MAE
    ax2.scatter(efficiency_metrics['Model_Size_MB'], metrics['MAE'], 
                s=200, alpha=0.7, color='red', edgecolors='black')
    ax2.set_xlabel('Model Size (MB)', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title('Model Size vs Mean Absolute Error', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax2.annotate(f'Current Model\nMAE: {metrics["MAE"]:.4f}\nSize: {efficiency_metrics["Model_Size_MB"]:.2f} MB',
                 xy=(efficiency_metrics['Model_Size_MB'], metrics['MAE']),
                 xytext=(efficiency_metrics['Model_Size_MB']*0.8, metrics['MAE']*1.2),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                 fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Graph saved: {save_path}")

def plot_ablation_study(df_processed, save_path='analysis_graphs/07_ablation_study.png'):
    """Show the effect of including sentiment features"""
    print("\nAblation Study: Effect of Sentiment Features")
    print("=" * 50)
    
    sentiment_corr = df_processed['Sentiment'].corr(df_processed['Next_Day_Close'])
    print(f"Sentiment-Target Correlation: {sentiment_corr:.4f}")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Sentiment vs target scatter
    ax1.scatter(df_processed['Sentiment'], df_processed['Next_Day_Close'], alpha=0.6, color='green')
    ax1.set_xlabel('Sentiment Score')
    ax1.set_ylabel('Next Day Close Price')
    ax1.set_title('Sentiment vs Target Correlation')
    ax1.grid(True, alpha=0.3)
    
    # Sentiment distribution
    ax2.hist(df_processed['Sentiment'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Sentiment Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Sentiment Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Sentiment over time
    ax3.plot(df_processed['Date'], df_processed['Sentiment'], color='purple', alpha=0.7)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Sentiment Score')
    ax3.set_title('Sentiment Over Time')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Sentiment vs volume
    ax4.scatter(df_processed['Volume'], df_processed['Sentiment'], alpha=0.6, color='orange')
    ax4.set_xlabel('Volume')
    ax4.set_ylabel('Sentiment Score')
    ax4.set_title('Sentiment vs Volume')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Graph saved: {save_path}")
    
    if abs(sentiment_corr) > 0.1:
        print(f"✓ Sentiment shows moderate correlation ({sentiment_corr:.4f}) with target")
    else:
        print(f"✗ Sentiment shows weak correlation ({sentiment_corr:.4f}) with target")

def plot_rolling_performance(y_test, y_pred, window=30, save_path='analysis_graphs/08_rolling_performance.png'):
    """Plot rolling RMSE over time"""
    residuals = y_pred - y_test
    rolling_rmse = pd.Series(residuals).rolling(window=window).apply(
        lambda x: np.sqrt(np.mean(x**2)), raw=True
    )
    
    plt.figure(figsize=(15, 8))
    plt.plot(rolling_rmse.index, rolling_rmse.values, linewidth=2, color='darkblue', 
             label=f'{window}-Day Rolling RMSE')
    
    overall_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    plt.axhline(y=overall_rmse, color='red', linestyle='--', alpha=0.7, 
                label=f'Overall RMSE: {overall_rmse:.4f}')
    
    plt.title(f'{window}-Day Rolling RMSE Performance', fontsize=16, fontweight='bold')
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Rolling RMSE', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Graph saved: {save_path}")

def plot_feature_analysis(df_processed, save_path='analysis_graphs/09_feature_analysis.png'):
    """Analyze feature importance and correlations"""
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment', 
                       'Previous_Close', 'Moving_Avg_3d', 'Moving_Avg_7d']
    
    correlation_matrix = df_processed[feature_columns + ['Next_Day_Close']].corr()
    target_correlations = correlation_matrix['Next_Day_Close'].drop('Next_Day_Close').abs().sort_values(ascending=False)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Correlation heatmap
    im1 = ax1.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xticks(range(len(correlation_matrix.columns)))
    ax1.set_yticks(range(len(correlation_matrix.columns)))
    ax1.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
    ax1.set_yticklabels(correlation_matrix.columns)
    ax1.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax1)
    
    # Feature importance
    bars = ax2.barh(range(len(target_correlations)), target_correlations.values, 
                     color='skyblue', edgecolor='black')
    ax2.set_yticks(range(len(target_correlations)))
    ax2.set_yticklabels(target_correlations.index)
    ax2.set_xlabel('Absolute Correlation with Target', fontsize=12)
    ax2.set_title('Feature Importance (Correlation with Target)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Sentiment analysis
    ax3.scatter(df_processed['Sentiment'], df_processed['Next_Day_Close'], alpha=0.6, color='green')
    ax3.set_xlabel('Sentiment Score')
    ax3.set_ylabel('Next Day Close Price')
    ax3.set_title('Sentiment vs Target Relationship', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Price features relationship
    ax4.scatter(df_processed['Close'], df_processed['Next_Day_Close'], alpha=0.6, color='blue')
    ax4.set_xlabel('Current Close Price')
    ax4.set_ylabel('Next Day Close Price')
    ax4.set_title('Current vs Next Day Close Price', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Graph saved: {save_path}")
    
    print(f"\nFeature Importance Summary:")
    print("=" * 40)
    for feature, corr in target_correlations.items():
        print(f"{feature}: {corr:.4f}")

def generate_summary_report(metrics, efficiency_metrics):
    """Generate a comprehensive summary report"""
    print("\n" + "=" * 80)
    print("LSTM MODEL COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 80)
    
    print("\n5. MODEL PERFORMANCE ASSESSMENT:")
    print("-" * 40)
    
    if metrics['R2'] > 0.8:
        print("✓ Excellent predictive power (R² > 0.8)")
    elif metrics['R2'] > 0.6:
        print("✓ Good predictive power (R² > 0.6)")
    elif metrics['R2'] > 0.4:
        print("✓ Moderate predictive power (R² > 0.4)")
    else:
        print("✗ Limited predictive power (R² < 0.4)")
    
    if metrics['Directional_Accuracy'] > 60:
        print(f"✓ Good directional accuracy ({metrics['Directional_Accuracy']:.1f}%)")
    else:
        print(f"✗ Limited directional accuracy ({metrics['Directional_Accuracy']:.1f}%)")
    
    if efficiency_metrics['Avg_Inference_Time_ms'] < 10:
        print(f"✓ Fast inference time ({efficiency_metrics['Avg_Inference_Time_ms']:.2f} ms)")
    else:
        print(f"⚠ Moderate inference time ({efficiency_metrics['Avg_Inference_Time_ms']:.2f} ms)")
    
    print("\n6. RECOMMENDATIONS:")
    print("-" * 40)
    
    if metrics['R2'] < 0.6:
        print("• Consider feature engineering or additional data sources")
        print("• Experiment with different model architectures")
    
    if metrics['Directional_Accuracy'] < 60:
        print("• Focus on improving trend prediction capabilities")
        print("• Consider ensemble methods or additional features")
    
    if efficiency_metrics['Avg_Inference_Time_ms'] > 10:
        print("• Consider model optimization for production deployment")
        print("• Evaluate model quantization or pruning")
    
    print("\n" + "=" * 80)

def main():
    """Main analysis function"""
    print("LSTM Model Comprehensive Analysis")
    print("=" * 50)
    
    # Load model and data
    model, df_processed, _ = load_model_and_data()
    
    if model is None:
        print("Failed to load model or data. Exiting.")
        return
    
    # Prepare data
    print("\nPreparing data for analysis...")
    X, y = prepare_features(df_processed)
    print(f"Feature shape: {X.shape}, Target shape: {y.shape}")
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Get predictions
    print("\nGenerating predictions...")
    y_pred = model.predict(X_test)
    y_pred = y_pred.flatten()
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_test, y_pred)
    efficiency_metrics = calculate_efficiency_metrics(model, X_test, y_test)
    
    # Create metrics table
    create_metrics_table(metrics, efficiency_metrics)
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    
    # Get dates for time series plots
    test_dates = df_processed['Date'].iloc[split_idx+10:split_idx+10+len(y_test)]
    
    # Generate all plots
    plot_prediction_vs_actual(y_test, y_pred, test_dates)
    plot_residuals(y_test, y_pred, test_dates)
    plot_scatter_predicted_vs_actual(y_test, y_pred)
    plot_error_distribution(y_test, y_pred)
    plot_metrics_comparison(metrics, efficiency_metrics)
    plot_complexity_tradeoff(metrics, efficiency_metrics)
    plot_ablation_study(df_processed)
    plot_rolling_performance(y_test, y_pred)
    plot_feature_analysis(df_processed)
    
    # Generate summary report
    generate_summary_report(metrics, efficiency_metrics)
    
    print(f"\n✓ Analysis complete! All graphs saved in 'analysis_graphs/' directory")
    print(f"✓ Generated {len([f for f in os.listdir('analysis_graphs') if f.endswith('.png')])} visualization files")

if __name__ == "__main__":
    main()
