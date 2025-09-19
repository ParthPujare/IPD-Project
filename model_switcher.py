#!/usr/bin/env python3
"""
Model Switcher Script
This script demonstrates how to easily switch between the original and enhanced models
in your existing pipeline without changing any other code.
"""

import numpy as np
import pandas as pd
import sqlite3
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced model loader
from lstm_model_v2 import load_model_v2

# Configuration
DB_PATH = "stock_data.db"
ORIGINAL_MODEL_PATH = "stock_price_model.h5"
ENHANCED_MODEL_PATH = "stock_price_model_v2.h5"
PRICE_SCALER_PATH = "price_scaler.pkl"
SENTIMENT_SCALER_PATH = "sentiment_scaler.pkl"
N_PAST = 10

class ModelSwitcher:
    """Class to easily switch between different LSTM models"""
    
    def __init__(self, model_type="enhanced"):
        """
        Initialize the model switcher
        
        Args:
            model_type (str): "original", "enhanced", or "auto" (chooses best performing)
        """
        self.model_type = model_type
        self.model = None
        self.price_scaler = None
        self.sentiment_scaler = None
        self.load_model_and_scalers()
    
    def load_model_and_scalers(self):
        """Load the specified model and scalers"""
        print(f"Loading {self.model_type} model...")
        
        try:
            # Load scalers
            with open(PRICE_SCALER_PATH, 'rb') as f:
                self.price_scaler = pickle.load(f)
            with open(SENTIMENT_SCALER_PATH, 'rb') as f:
                self.sentiment_scaler = pickle.load(f)
            print("✓ Scalers loaded successfully")
            
            # Load model based on type
            if self.model_type == "original":
                self.model = load_model(ORIGINAL_MODEL_PATH)
                print("✓ Original model loaded successfully")
            elif self.model_type == "enhanced":
                self.model = load_model_v2(ENHANCED_MODEL_PATH)
                print("✓ Enhanced model 2.0 loaded successfully")
            elif self.model_type == "auto":
                # Try enhanced first, fallback to original
                try:
                    self.model = load_model_v2(ENHANCED_MODEL_PATH)
                    print("✓ Enhanced model 2.0 loaded successfully (auto-selected)")
                except:
                    self.model = load_model(ORIGINAL_MODEL_PATH)
                    print("✓ Original model loaded successfully (fallback)")
            else:
                raise ValueError("model_type must be 'original', 'enhanced', or 'auto'")
                
        except Exception as e:
            print(f"✗ Error loading model/scalers: {e}")
            raise
    
    def switch_model(self, new_model_type):
        """Switch to a different model type"""
        print(f"\nSwitching from {self.model_type} to {new_model_type}...")
        self.model_type = new_model_type
        self.load_model_and_scalers()
        print(f"✓ Successfully switched to {new_model_type} model")
    
    def get_model_info(self):
        """Get information about the currently loaded model"""
        if self.model is None:
            return "No model loaded"
        
        info = {
            'model_type': self.model_type,
            'total_parameters': self.model.count_params(),
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'model_size_mb': self._get_model_size()
        }
        return info
    
    def _get_model_size(self):
        """Get model file size in MB"""
        try:
            if self.model_type == "original":
                import os
                return os.path.getsize(ORIGINAL_MODEL_PATH) / (1024 * 1024)
            elif self.model_type == "enhanced":
                import os
                return os.path.getsize(ENHANCED_MODEL_PATH) / (1024 * 1024)
            else:
                return "Unknown"
        except:
            return "Unknown"
    
    def predict(self, data):
        """Make prediction using the current model"""
        if self.model is None:
            raise ValueError("No model loaded")
        
        return self.model.predict(data, verbose=0)
    
    def evaluate(self, data, labels):
        """Evaluate the current model"""
        if self.model is None:
            raise ValueError("No model loaded")
        
        return self.model.evaluate(data, labels, verbose=0)

def demonstrate_model_switching():
    """Demonstrate how to switch between models"""
    print("="*70)
    print("MODEL SWITCHING DEMONSTRATION")
    print("="*70)
    
    # Initialize with enhanced model
    print("\n1. Starting with Enhanced Model 2.0:")
    switcher = ModelSwitcher("enhanced")
    
    # Show model info
    info = switcher.get_model_info()
    print(f"   Model Type: {info['model_type']}")
    print(f"   Parameters: {info['total_parameters']:,}")
    print(f"   Model Size: {info['model_size_mb']:.2f} MB")
    
    # Switch to original model
    print("\n2. Switching to Original Model:")
    switcher.switch_model("original")
    
    info = switcher.get_model_info()
    print(f"   Model Type: {info['model_type']}")
    print(f"   Parameters: {info['total_parameters']:,}")
    print(f"   Model Size: {info['model_size_mb']:.2f} MB")
    
    # Switch back to enhanced
    print("\n3. Switching back to Enhanced Model 2.0:")
    switcher.switch_model("enhanced")
    
    info = switcher.get_model_info()
    print(f"   Model Type: {info['model_type']}")
    print(f"   Parameters: {info['total_parameters']:,}")
    print(f"   Model Size: {info['model_size_mb']:.2f} MB")
    
    # Test auto-selection
    print("\n4. Testing Auto-Selection:")
    auto_switcher = ModelSwitcher("auto")
    
    info = auto_switcher.get_model_info()
    print(f"   Auto-Selected: {info['model_type']}")
    print(f"   Parameters: {info['total_parameters']:,}")
    
    print("\n" + "="*70)
    print("MODEL SWITCHING SUCCESSFUL! ✓")
    print("="*70)

def show_integration_examples():
    """Show examples of how to integrate the model switcher in your pipeline"""
    print("\n" + "="*70)
    print("INTEGRATION EXAMPLES")
    print("="*70)
    
    print("\n1. Simple Model Switching in Your Scripts:")
    print("""
# At the top of your script
from model_switcher import ModelSwitcher

# Choose your model
switcher = ModelSwitcher("enhanced")  # or "original" or "auto"

# Use exactly like before
predictions = switcher.predict(your_data)
evaluation = switcher.evaluate(test_data, test_labels)

# Switch models anytime
switcher.switch_model("original")
new_predictions = switcher.predict(your_data)
    """)
    
    print("\n2. Environment-Based Model Selection:")
    print("""
import os

# Choose model based on environment
model_type = os.getenv("LSTM_MODEL_TYPE", "enhanced")
switcher = ModelSwitcher(model_type)

# Your existing code works unchanged
    """)
    
    print("\n3. A/B Testing Setup:")
    print("""
# Test both models on the same data
enhanced_switcher = ModelSwitcher("enhanced")
original_switcher = ModelSwitcher("original")

enhanced_pred = enhanced_switcher.predict(test_data)
original_pred = original_switcher.predict(test_data)

# Compare results
enhanced_score = evaluate_predictions(enhanced_pred, actual)
original_score = evaluate_predictions(original_pred, actual)

print(f"Enhanced Model Score: {enhanced_score}")
print(f"Original Model Score: {original_score}")
    """)
    
    print("\n4. Production Deployment:")
    print("""
# In production, use auto-selection for reliability
switcher = ModelSwitcher("auto")

# This will automatically choose the best available model
# and fallback to original if enhanced fails to load

try:
    predictions = switcher.predict(production_data)
except Exception as e:
    print(f"Prediction failed: {e}")
    # Handle gracefully
    """)

def show_usage_tips():
    """Show practical usage tips"""
    print("\n" + "="*70)
    print("USAGE TIPS")
    print("="*70)
    
    print("\n✅ Best Practices:")
    print("• Use 'auto' mode in production for reliability")
    print("• Test both models before switching")
    print("• Monitor performance after switching")
    print("• Keep both models as backup")
    
    print("\n⚠️ Common Pitfalls:")
    print("• Don't switch models mid-prediction")
    print("• Ensure scalers are compatible")
    print("• Test with small data first")
    print("• Monitor memory usage")
    
    print("\n🚀 Performance Tips:")
    print("• Enhanced model has more parameters but better accuracy")
    print("• Original model is faster for inference")
    print("• Use enhanced for critical predictions")
    print("• Use original for high-frequency predictions")

if __name__ == "__main__":
    # Demonstrate model switching
    demonstrate_model_switching()
    
    # Show integration examples
    show_integration_examples()
    
    # Show usage tips
    show_usage_tips()
    
    print("\n" + "="*70)
    print("YOUR PIPELINE IS NOW MODEL-AGNOSTIC! 🎯")
    print("="*70)
    print("You can easily switch between models without changing any other code.")
    print("Just change the model_type parameter and everything else works the same!") 