#!/usr/bin/env python3
"""
Research Paper Graphs for Tuned Model 2.0
This script generates publication-quality graphs for the hyperparameter-tuned LSTM model
including prediction vs actual, residuals, and time series analysis.
"""

import sqlite3
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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

# Set style for publication-quality graphs
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

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
    return X_test, y_test, price_scaler, df

def load_tuned_model(input_shape):
    """Load the tuned Model 2.0"""
    print("Loading tuned Model 2.0...")
    
    model = load_model_v2(TUNED_MODEL_PATH)
    if model is None:
        print("✗ Model not found. Please run training first.")
        return None
    
    print("✓ Tuned model loaded successfully")
    return model

def inverse_transform_predictions(y_pred, y_true, price_scaler):
    """Inverse transform predictions and actual values to original scale"""
    # Create dummy arrays with correct shape for inverse transform
    y_pred_reshaped = np.zeros((len(y_pred), 7))  # 7 price features
    y_pred_reshaped[:, 3] = y_pred.flatten()  # Set Close price column
    
    y_true_reshaped = np.zeros((len(y_true), 7))
    y_true_reshaped[:, 3] = y_true
    
    # Inverse transform
    y_pred_actual = price_scaler.inverse_transform(y_pred_reshaped)[:, 3]
    y_true_actual = price_scaler.inverse_transform(y_true_reshaped)[:, 3]
    
    return y_pred_actual, y_true_actual

def plot_prediction_vs_actual(y_true, y_pred, save_path="research_graphs"):
    """Plot 1: Prediction vs Actual Values (Scatter Plot)"""
    print("Generating Prediction vs Actual plot...")
    
    # Create individual figure for scatter plot
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.6, s=30, color='steelblue', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Stock Price ($)', fontweight='bold')
    ax1.set_ylabel('Predicted Stock Price ($)', fontweight='bold')
    ax1.set_title('Prediction vs Actual Values', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add R² value
    r2 = r2_score(y_true, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/prediction_vs_actual_scatter.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create individual figure for time series plot
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Time series plot
    ax2.plot(y_true, label='Actual', linewidth=2, color='darkblue')
    ax2.plot(y_pred, label='Predicted', linewidth=2, color='red', alpha=0.8)
    ax2.set_xlabel('Time Steps', fontweight='bold')
    ax2.set_ylabel('Stock Price ($)', fontweight='bold')
    ax2.set_title('Time Series: Actual vs Predicted', fontweight='bold', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/prediction_vs_actual_timeseries.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig1, fig2

def plot_residuals_analysis(y_true, y_pred, save_path="research_graphs"):
    """Plot 2: Residuals Analysis - Individual plots"""
    print("Generating Residuals Analysis plots...")
    
    residuals = y_true - y_pred
    
    # Plot 1: Residuals vs Predicted
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.scatter(y_pred, residuals, alpha=0.6, s=30, color='steelblue', edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Values ($)', fontweight='bold')
    ax1.set_ylabel('Residuals ($)', fontweight='bold')
    ax1.set_title('Residuals vs Predicted Values', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/residuals_vs_predicted.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Residuals vs Time
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(residuals, alpha=0.7, color='darkgreen', linewidth=1.5)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Time Steps', fontweight='bold')
    ax2.set_ylabel('Residuals ($)', fontweight='bold')
    ax2.set_title('Residuals vs Time', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/residuals_vs_time.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Residuals Distribution
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ax3.hist(residuals, bins=30, alpha=0.7, color='lightcoral', edgecolor='black', linewidth=0.5)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residuals ($)', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Residuals Distribution', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/residuals_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 4: Q-Q Plot
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot of Residuals', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/residuals_qq_plot.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig1, fig2, fig3, fig4

def plot_error_metrics(y_true, y_pred, save_path="research_graphs"):
    """Plot 3: Error Metrics Visualization - Individual plots"""
    print("Generating Error Metrics plots...")
    
    # Calculate errors
    absolute_errors = np.abs(y_true - y_pred)
    percentage_errors = np.abs((y_true - y_pred) / y_true) * 100
    
    # Plot 1: Absolute Errors Distribution
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.hist(absolute_errors, bins=25, alpha=0.7, color='lightblue', edgecolor='black', linewidth=0.5)
    ax1.axvline(x=absolute_errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${absolute_errors.mean():.2f}')
    ax1.set_xlabel('Absolute Error ($)', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('Distribution of Absolute Errors', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/absolute_errors_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Percentage Errors Distribution
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.hist(percentage_errors, bins=25, alpha=0.7, color='lightgreen', edgecolor='black', linewidth=0.5)
    ax2.axvline(x=percentage_errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {percentage_errors.mean():.2f}%')
    ax2.set_xlabel('Percentage Error (%)', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Distribution of Percentage Errors', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/percentage_errors_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Cumulative Error Analysis
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    cumulative_mae = np.cumsum(absolute_errors) / np.arange(1, len(absolute_errors) + 1)
    ax3.plot(cumulative_mae, color='purple', linewidth=2)
    ax3.set_xlabel('Number of Predictions', fontweight='bold')
    ax3.set_ylabel('Cumulative MAE ($)', fontweight='bold')
    ax3.set_title('Cumulative Mean Absolute Error', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/cumulative_mae.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 4: Error vs Price Level
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    ax4.scatter(y_true, absolute_errors, alpha=0.6, s=30, color='orange', edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Actual Stock Price ($)', fontweight='bold')
    ax4.set_ylabel('Absolute Error ($)', fontweight='bold')
    ax4.set_title('Error vs Price Level', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/error_vs_price_level.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig1, fig2, fig3, fig4

def plot_directional_accuracy(y_true, y_pred, save_path="research_graphs"):
    """Plot 4: Directional Accuracy Analysis - Individual plots"""
    print("Generating Directional Accuracy plots...")
    
    # Calculate price changes
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    
    # Calculate directional accuracy
    correct_directions = (y_true_diff > 0) == (y_pred_diff > 0)
    directional_accuracy = np.mean(correct_directions) * 100
    
    # Plot 1: Directional Accuracy Pie Chart
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    correct = np.sum(correct_directions)
    incorrect = len(correct_directions) - correct
    labels = ['Correct Direction', 'Incorrect Direction']
    sizes = [correct, incorrect]
    colors = ['lightgreen', 'lightcoral']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Directional Accuracy: {directional_accuracy:.2f}%', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_path}/directional_accuracy_pie.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Price Change Direction Comparison
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    time_steps = np.arange(len(y_true_diff))
    ax2.scatter(time_steps[y_true_diff > 0], y_true_diff[y_true_diff > 0], 
                color='green', alpha=0.6, s=30, label='Actual Up', marker='^')
    ax2.scatter(time_steps[y_true_diff < 0], y_true_diff[y_true_diff < 0], 
                color='red', alpha=0.6, s=30, label='Actual Down', marker='v')
    ax2.scatter(time_steps[y_pred_diff > 0], y_pred_diff[y_pred_diff > 0], 
                color='blue', alpha=0.6, s=30, label='Predicted Up', marker='s')
    ax2.scatter(time_steps[y_pred_diff < 0], y_pred_diff[y_pred_diff < 0], 
                color='orange', alpha=0.6, s=30, label='Predicted Down', marker='o')
    
    ax2.set_xlabel('Time Steps', fontweight='bold')
    ax2.set_ylabel('Price Change ($)', fontweight='bold')
    ax2.set_title('Price Change Direction: Actual vs Predicted', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/price_change_direction.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Confusion Matrix for Direction
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_diff > 0, y_pred_diff > 0)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    ax3.set_xlabel('Predicted Direction', fontweight='bold')
    ax3.set_ylabel('Actual Direction', fontweight='bold')
    ax3.set_title('Direction Prediction Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_path}/direction_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 4: Rolling Directional Accuracy
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    window_size = 20
    rolling_accuracy = []
    for i in range(window_size, len(correct_directions)):
        rolling_accuracy.append(np.mean(correct_directions[i-window_size:i]) * 100)
    
    ax4.plot(range(window_size, len(correct_directions)), rolling_accuracy, 
             color='purple', linewidth=2)
    ax4.axhline(y=directional_accuracy, color='red', linestyle='--', 
                linewidth=2, label=f'Overall: {directional_accuracy:.2f}%')
    ax4.set_xlabel('Time Steps', fontweight='bold')
    ax4.set_ylabel('Directional Accuracy (%)', fontweight='bold')
    ax4.set_title(f'Rolling Directional Accuracy (Window: {window_size})', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/rolling_directional_accuracy.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig1, fig2, fig3, fig4

def plot_model_performance_summary(y_true, y_pred, save_path="research_graphs"):
    """Plot 5: Comprehensive Performance Summary - Individual plots"""
    print("Generating Performance Summary plots...")
    
    # Calculate all metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    # Calculate directional accuracy
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    directional_accuracy = np.mean((y_true_diff > 0) == (y_pred_diff > 0)) * 100
    
    # Calculate absolute errors for this function
    absolute_errors = np.abs(y_true - y_pred)
    
    # Plot 1: Metrics Bar Chart
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    metrics = ['MAE', 'RMSE', 'MAPE', 'R²', 'Dir. Acc.']
    values = [mae, rmse, mape, r2*100, directional_accuracy]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum']
    
    bars = ax1.bar(metrics, values, color=colors, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Value', fontweight='bold')
    ax1.set_title('Model Performance Metrics', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(values),
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/performance_metrics_bar.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Prediction Accuracy Over Time
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    window_size = 15
    rolling_mae = []
    rolling_r2 = []
    
    for i in range(window_size, len(y_true)):
        y_true_window = y_true[i-window_size:i]
        y_pred_window = y_pred[i-window_size:i]
        rolling_mae.append(mean_absolute_error(y_true_window, y_pred_window))
        rolling_r2.append(r2_score(y_true_window, y_pred_window))
    
    time_steps = range(window_size, len(y_true))
    ax2.plot(time_steps, rolling_mae, label='Rolling MAE', color='red', linewidth=2)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(time_steps, rolling_r2, label='Rolling R²', color='blue', linewidth=2)
    
    ax2.set_xlabel('Time Steps', fontweight='bold')
    ax2.set_ylabel('MAE ($)', fontweight='bold', color='red')
    ax2_twin.set_ylabel('R²', fontweight='bold', color='blue')
    ax2.set_title('Rolling Performance Metrics', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/rolling_performance_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Error Distribution by Price Range
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    price_ranges = np.linspace(y_true.min(), y_true.max(), 6)
    range_errors = []
    range_labels = []
    
    for i in range(len(price_ranges)-1):
        mask = (y_true >= price_ranges[i]) & (y_true < price_ranges[i+1])
        if np.sum(mask) > 0:
            range_errors.append(np.mean(absolute_errors[mask]))
            range_labels.append(f'${price_ranges[i]:.0f}-\n${price_ranges[i+1]:.0f}')
    
    ax3.bar(range_labels, range_errors, color='lightcoral', edgecolor='black', linewidth=1)
    ax3.set_xlabel('Price Range ($)', fontweight='bold')
    ax3.set_ylabel('Mean Absolute Error ($)', fontweight='bold')
    ax3.set_title('Error Distribution by Price Range', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{save_path}/error_by_price_range.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 4: Model Confidence Intervals
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    from scipy import stats
    confidence_level = 0.95
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Calculate prediction intervals
    residuals = y_true - y_pred
    residual_std = np.std(residuals)
    prediction_intervals = z_score * residual_std
    
    ax4.fill_between(range(len(y_true)), y_pred - prediction_intervals, 
                     y_pred + prediction_intervals, alpha=0.3, color='lightblue', 
                     label=f'{confidence_level*100:.0f}% Prediction Interval')
    ax4.plot(y_true, label='Actual', color='darkblue', linewidth=2)
    ax4.plot(y_pred, label='Predicted', color='red', linewidth=2, alpha=0.8)
    
    ax4.set_xlabel('Time Steps', fontweight='bold')
    ax4.set_ylabel('Stock Price ($)', fontweight='bold')
    ax4.set_title('Predictions with Confidence Intervals', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/predictions_with_confidence.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig1, fig2, fig3, fig4

def generate_metrics_table(y_true, y_pred, save_path="research_graphs"):
    """Generate a comprehensive metrics table"""
    print("Generating metrics table...")
    
    # Calculate all metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    # Calculate directional accuracy
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    directional_accuracy = np.mean((y_true_diff > 0) == (y_pred_diff > 0)) * 100
    
    # Create metrics table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    metrics_data = [
        ['Metric', 'Value', 'Description'],
        ['MAE', f'${mae:.4f}', 'Mean Absolute Error'],
        ['RMSE', f'${rmse:.4f}', 'Root Mean Square Error'],
        ['MAPE', f'{mape:.2f}%', 'Mean Absolute Percentage Error'],
        ['R²', f'{r2:.4f}', 'Coefficient of Determination'],
        ['Directional Accuracy', f'{directional_accuracy:.2f}%', 'Price Direction Prediction Accuracy']
    ]
    
    table = ax.table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(metrics_data)):
        for j in range(len(metrics_data[0])):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                if j == 0:  # Metric names
                    table[(i, j)].set_facecolor('#E8F5E8')
                elif j == 1:  # Values
                    table[(i, j)].set_facecolor('#FFF3E0')
                    table[(i, j)].set_text_props(weight='bold')
                else:  # Descriptions
                    table[(i, j)].set_facecolor('#F3E5F5')
    
    ax.set_title('Tuned Model 2.0 Performance Metrics', fontweight='bold', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/metrics_table.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """Main function to generate all research graphs"""
    print("="*70)
    print("RESEARCH PAPER GRAPHS GENERATION FOR TUNED MODEL 2.0")
    print("="*70)
    
    # Create output directory
    save_path = "research_graphs"
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # Load data and scalers
        df, price_scaler, sentiment_scaler = load_data_and_scalers()
        if df is None:
            return
        
        # Prepare test data
        X_test, y_test, price_scaler, df = prepare_test_data(df, price_scaler, sentiment_scaler)
        
        # Load tuned model
        model = load_tuned_model((X_test.shape[1], X_test.shape[2]))
        if model is None:
            return
        
        # Make predictions
        print("Making predictions...")
        y_pred = model.predict(X_test, verbose=0)
        
        # Inverse transform to original scale
        y_pred_actual, y_true_actual = inverse_transform_predictions(y_pred, y_test, price_scaler)
        
        print(f"✓ Predictions completed. Generating {len(y_pred_actual)} graphs...")
        
        # Generate all graphs
        print("\n" + "="*50)
        print("GENERATING RESEARCH GRAPHS")
        print("="*50)
        
        # 1. Prediction vs Actual (2 plots)
        fig1, fig2 = plot_prediction_vs_actual(y_true_actual, y_pred_actual, save_path)
        print("✓ Prediction vs Actual plots generated (2 plots)")
        
        # 2. Residuals Analysis (4 plots)
        fig1, fig2, fig3, fig4 = plot_residuals_analysis(y_true_actual, y_pred_actual, save_path)
        print("✓ Residuals Analysis plots generated (4 plots)")
        
        # 3. Error Metrics (4 plots)
        fig1, fig2, fig3, fig4 = plot_error_metrics(y_true_actual, y_pred_actual, save_path)
        print("✓ Error Metrics plots generated (4 plots)")
        
        # 4. Directional Accuracy (4 plots)
        fig1, fig2, fig3, fig4 = plot_directional_accuracy(y_true_actual, y_pred_actual, save_path)
        print("✓ Directional Accuracy plots generated (4 plots)")
        
        # 5. Performance Summary (4 plots)
        fig1, fig2, fig3, fig4 = plot_model_performance_summary(y_true_actual, y_pred_actual, save_path)
        print("✓ Performance Summary plots generated (4 plots)")
        
        # 6. Metrics Table
        fig6 = generate_metrics_table(y_true_actual, y_pred_actual, save_path)
        print("✓ Metrics Table generated")
        
        print("\n" + "="*70)
        print("ALL RESEARCH GRAPHS GENERATED SUCCESSFULLY! ✓")
        print("="*70)
        print(f"Graphs saved in: {save_path}/")
        print("\nGenerated individual plots (no subplots):")
        print("\n1. PREDICTION VS ACTUAL:")
        print("   • prediction_vs_actual_scatter.png - Scatter plot comparison")
        print("   • prediction_vs_actual_timeseries.png - Time series comparison")
        
        print("\n2. RESIDUALS ANALYSIS:")
        print("   • residuals_vs_predicted.png - Residuals vs predicted values")
        print("   • residuals_vs_time.png - Residuals over time")
        print("   • residuals_distribution.png - Residuals histogram")
        print("   • residuals_qq_plot.png - Q-Q plot for normality")
        
        print("\n3. ERROR METRICS:")
        print("   • absolute_errors_distribution.png - Absolute error distribution")
        print("   • percentage_errors_distribution.png - Percentage error distribution")
        print("   • cumulative_mae.png - Cumulative MAE over time")
        print("   • error_vs_price_level.png - Error vs price level")
        
        print("\n4. DIRECTIONAL ACCURACY:")
        print("   • directional_accuracy_pie.png - Directional accuracy pie chart")
        print("   • price_change_direction.png - Price change direction comparison")
        print("   • direction_confusion_matrix.png - Direction confusion matrix")
        print("   • rolling_directional_accuracy.png - Rolling directional accuracy")
        
        print("\n5. PERFORMANCE SUMMARY:")
        print("   • performance_metrics_bar.png - Performance metrics bar chart")
        print("   • rolling_performance_metrics.png - Rolling performance metrics")
        print("   • error_by_price_range.png - Error by price range")
        print("   • predictions_with_confidence.png - Predictions with confidence intervals")
        
        print("\n6. METRICS TABLE:")
        print("   • metrics_table.png - Comprehensive metrics summary table")
        
        print(f"\nTotal: 19 individual plots generated (no subplots)")
        print("\nThese graphs are publication-ready with high DPI (300) and professional styling.")
        print("Each plot is now a standalone figure as requested.")
        
    except Exception as e:
        print(f"\n✗ Error during graph generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 