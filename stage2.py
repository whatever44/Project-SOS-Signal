import numpy as np

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models

# Train-test split
from sklearn.model_selection import train_test_split

# Plotting (optional but useful)
import matplotlib.pyplot as plt

# HuggingFace private dataset access
from huggingface_hub import HfFileSystem, login



    # -----------------------
    # CNN + LSTM MODEL
    # -----------------------
model = models.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=(90, 126)),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),

        layers.Conv1D(128, 3, activation='relu'),
        layers.MaxPooling1D(2),

        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
