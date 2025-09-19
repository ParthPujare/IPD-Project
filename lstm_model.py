import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def predict_next_day_price(model, data):
    prediction = model.predict(data)
    return prediction[0][0]

def save_model(model, filename="lstm_model.h5"):
    model.save(filename)

def load_model(filename="lstm_model.h5"):
    return tf.keras.models.load_model(filename)
