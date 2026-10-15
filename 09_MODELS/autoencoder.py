import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def build_autoencoder(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(8, activation='relu', name='latent'),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(input_dim)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
