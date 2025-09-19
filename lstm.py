import sqlite3
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ✅ Database path
DB_PATH = "stock_data.db"

# ✅ Model and scaler save paths
MODEL_PATH = "stock_price_model.h5"
PRICE_SCALER_PATH = "price_scaler.pkl"
SENTIMENT_SCALER_PATH = "sentiment_scaler.pkl"

# ✅ Number of past days to consider as input features
N_PAST = 10

# ✅ Connect to the database and fetch data
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM processed_data ORDER BY Date ASC", conn)
conn.close()

# ✅ Convert Date column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# ✅ Define feature columns
price_features = ['Open', 'High', 'Low', 'Close', 'Previous_Close', 'Moving_Avg_3d', 'Moving_Avg_7d']
sentiment_features = ['Sentiment', 'Volume']

# ✅ Separate features for scaling
price_data = df[price_features]
sentiment_data = df[sentiment_features]

# ✅ Initialize and fit scalers
price_scaler = MinMaxScaler(feature_range=(0, 1))
sentiment_scaler = MinMaxScaler(feature_range=(0, 1))

scaled_price_data = price_scaler.fit_transform(price_data)
scaled_sentiment_data = sentiment_scaler.fit_transform(sentiment_data)

# ✅ Save the scalers for future use
with open(PRICE_SCALER_PATH, "wb") as f:
    pickle.dump(price_scaler, f)

with open(SENTIMENT_SCALER_PATH, "wb") as f:
    pickle.dump(sentiment_scaler, f)

# ✅ Combine scaled price and sentiment data
scaled_data = np.hstack((scaled_price_data, scaled_sentiment_data))

# ✅ Prepare training sequences
X, y = [], []
for i in range(N_PAST, len(scaled_data)):
    X.append(scaled_data[i - N_PAST : i])  # Last N_PAST days
    y.append(scaled_price_data[i, -1])  # Target: Scaled 'Close' price

X, y = np.array(X), np.array(y)

# ✅ Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(N_PAST, X.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)  # Predicting the scaled Close price
])

# ✅ Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# ✅ Train the model
model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2)

# ✅ Save the trained model
model.save(MODEL_PATH)

print(" Model training complete. Saved model and scalers.")
