"""Deep learning training utilities (LSTM/GRU) using Keras/TensorFlow."""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences


def build_lstm(vocab_size, embedding_dim=100, input_length=100):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_length),
        LSTM(128, return_sequences=False),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_gru(vocab_size, embedding_dim=100, input_length=100):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_length),
        GRU(128, return_sequences=False),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
