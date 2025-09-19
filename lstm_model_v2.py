import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import numpy as np

def build_lstm_model_v2(input_shape):
    """
    Enhanced LSTM model with TUNED hyperparameters for optimal performance
    """
    model = Sequential([
        # First LSTM layer - increased units and optimized dropout
        LSTM(256, return_sequences=True, input_shape=input_shape, 
             dropout=0.2, recurrent_dropout=0.2,
             kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        
        # Second LSTM layer - optimized units and dropout
        LSTM(128, return_sequences=True, 
             dropout=0.25, recurrent_dropout=0.25,
             kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        
        # Third LSTM layer - optimized units
        LSTM(64, return_sequences=False, 
             dropout=0.3, recurrent_dropout=0.3,
             kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        
        # Dense layers with optimized architecture and regularization
        Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.25),
        Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    # Optimized learning rate and optimizer parameters
    optimizer = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
    return model

def build_lstm_model_v2_compact(input_shape):
    """
    Alternative compact version with TUNED hyperparameters for faster training
    """
    model = Sequential([
        # First LSTM layer - optimized units and dropout
        LSTM(128, return_sequences=True, input_shape=input_shape, 
             dropout=0.2, recurrent_dropout=0.2,
             kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        
        # Second LSTM layer - optimized units
        LSTM(64, return_sequences=False, 
             dropout=0.25, recurrent_dropout=0.25,
             kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        
        # Dense layers with optimized regularization
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.25),
        Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.2),
        Dense(1)
    ])
    
    # Optimized learning rate
    optimizer = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
    return model

def get_callbacks():
    """
    Get TUNED training callbacks for optimal model performance
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,  # Increased patience for better convergence
        restore_best_weights=True,
        verbose=1,
        min_delta=1e-6  # Minimum change threshold
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,  # More aggressive learning rate reduction
        patience=10,  # Increased patience
        min_lr=1e-8,  # Lower minimum learning rate
        verbose=1,
        cooldown=5  # Cooldown period
    )
    
    return [early_stopping, reduce_lr]

def predict_next_day_price(model, data):
    """Predict next day price using the model"""
    prediction = model.predict(data, verbose=0)
    return prediction[0][0]

def save_model_v2(model, filename="stock_price_model_v2.h5"):
    """Save the enhanced model"""
    model.save(filename)
    print(f"✓ Enhanced model saved as {filename}")

def load_model_v2(filename="stock_price_model_v2.h5"):
    """Load the enhanced model"""
    try:
        model = tf.keras.models.load_model(filename)
        print(f"✓ Enhanced model loaded from {filename}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

def evaluate_model_performance(model, X_test, y_test):
    """Evaluate model performance with detailed metrics"""
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Calculate metrics
    mae = tf.keras.metrics.mean_absolute_error(y_test, y_pred).numpy()
    mse = tf.keras.metrics.mean_squared_error(y_test, y_pred).numpy()
    rmse = np.sqrt(mse)
    
    # Calculate R²
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Calculate directional accuracy
    y_test_diff = np.diff(y_test)
    y_pred_diff = np.diff(y_pred.flatten())
    directional_accuracy = np.mean((y_test_diff > 0) == (y_pred_diff > 0)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    } 