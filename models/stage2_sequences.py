import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_cnn1d(input_shape):
    """Your original baseline, slightly improved."""
    inputs = Input(shape=input_shape)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs, outputs, name="CNN_1D")

def build_bilstm(input_shape):
    """
    Bidirectional LSTM.
    Standard SOTA for temporal gestures before Transformers took over.
    Captures context from past and future frames.
    """
    inputs = Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inputs) # Ignore padded zeros
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs, outputs, name="BiLSTM")

def build_transformer(input_shape):
    """
    Transformer Encoder for Time Series.
    SOTA for handling long-range dependencies in complex gestures.
    """
    inputs = Input(shape=input_shape)
    
    # 1. Projection/Embedding (Linear or Conv)
    x = layers.Dense(64)(inputs) 
    
    # 2. Transformer Block (Multi-Head Attention)
    # Checks relationships between frame T=1 and T=90 directly
    attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Add()([x, attention_output]) # Residual
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # 3. Feed Forward Part
    ffn = layers.Dense(64, activation="relu")(x)
    ffn = layers.Dense(64)(ffn)
    x = layers.Add()([x, ffn]) # Residual
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # 4. Global Pooling & Output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model(inputs, outputs, name="Transformer")

def build_tcn(input_shape):
    """
    Temporal Convolutional Network.
    Uses dilated convolutions to see wide history without RNN slowness.
    """
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Dilated blocks: 1, 2, 4, 8
    for dilation in [1, 2, 4, 8]:
        residual = x
        x = layers.Conv1D(64, kernel_size=3, padding='same', dilation_rate=dilation, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        # Match dimensions for residual if needed
        if residual.shape[-1] != 64:
            residual = layers.Conv1D(64, 1, padding='same')(residual)
        x = layers.Add()([x, residual])
        
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs, outputs, name="TCN")