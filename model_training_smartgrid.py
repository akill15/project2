"""
SmartGrid Security Model Training Script
Trains LSTM and Autoencoder models for anomaly detection
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import joblib
import os

# ─── Configuration ─────────────────────────────────────────────────────
SEQUENCE_LENGTH = 10
FEATURE_DIM = 13  # Number of grid features
LATENT_DIM = 5
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# ─── Generate Synthetic SmartGrid Data ──────────────────────────────────
def generate_smartgrid_data(num_samples=5000):
    """
    Generates synthetic SmartGrid time-series data
    Features: Voltage, Current, Frequency, Power Factor, Active/Reactive/Apparent Power,
              Harmonic Distortion, Phase Angle, Load Demand, Generation Output, Net Flow, Temperature
    """
    print("🔄 Generating synthetic SmartGrid data...")
    
    data = []
    labels = []
    
    for i in range(num_samples):
        # Normal operational data
        if i < num_samples * 0.8:  # 80% normal
            voltage = np.random.normal(230, 5)
            current = np.random.normal(45, 3)
            frequency = np.random.normal(50, 0.1)
            power_factor = np.random.normal(0.95, 0.02)
            active_power = np.random.normal(10, 1)
            reactive_power = np.random.normal(2, 0.3)
            apparent_power = np.random.normal(10.5, 1)
            harmonic_distortion = np.random.normal(3, 0.5)
            phase_angle = np.random.normal(11, 2)
            load_demand = np.random.normal(80, 5)
            generation_output = np.random.normal(85, 5)
            net_flow = np.random.normal(5, 2)
            temperature = np.random.normal(35, 3)
            label = 0  # Normal
            
        else:  # 20% anomalous
            voltage = np.random.normal(200, 10) if np.random.rand() > 0.5 else np.random.normal(260, 10)
            current = np.random.normal(60, 5)
            frequency = np.random.uniform(48, 52)
            power_factor = np.random.normal(0.7, 0.1)
            active_power = np.random.normal(15, 3)
            reactive_power = np.random.normal(5, 2)
            apparent_power = np.random.normal(16, 3)
            harmonic_distortion = np.random.normal(8, 2)
            phase_angle = np.random.normal(25, 5)
            load_demand = np.random.normal(50, 10) if np.random.rand() > 0.5 else np.random.normal(95, 5)
            generation_output = np.random.normal(60, 10) if np.random.rand() > 0.5 else np.random.normal(100, 5)
            net_flow = np.random.normal(15, 5)
            temperature = np.random.normal(50, 5)
            label = 1  # Anomaly
        
        sample = [voltage, current, frequency, power_factor, active_power,
                  reactive_power, apparent_power, harmonic_distortion,
                  phase_angle, load_demand, generation_output, net_flow, temperature]
        data.append(sample)
        labels.append(label)
    
    X = np.array(data)
    y = np.array(labels)
    
    print(f"✅ Generated {len(X)} samples")
    print(f"   Normal samples: {np.sum(y == 0)}")
    print(f"   Anomalies: {np.sum(y == 1)}")
    
    return X, y

# ─── Create Sequences for LSTM ─────────────────────────────────────────────
def create_sequences(data, seq_length=SEQUENCE_LENGTH):
    """
    Creates sequences for LSTM training
    """
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

# ─── Build LSTM Anomaly Detector ──────────────────────────────────────────────
def build_lstm_model(input_shape):
    """
    LSTM-based anomaly detection model
    """
    print("🔨 Building LSTM model...")
    
    model = keras.Sequential([
        layers.LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, activation='relu', return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output: 0-1 anomaly score
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['mae', 'accuracy']
    )
    
    print(model.summary())
    return model

# ─── Build Autoencoder ────────────────────────────────────────────────────────
def build_autoencoder(input_dim):
    """
    Autoencoder for reconstruction-based anomaly detection
    """
    print("🔨 Building Autoencoder...")
    
    # Encoder
    inputs = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(8, activation='relu')(inputs)
    encoded = layers.Dense(LATENT_DIM, activation='relu')(encoded)
    
    # Decoder
    decoded = layers.Dense(8, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(
        optimizer='adam',
        loss='mse'
    )
    
    print(autoencoder.summary())
    return autoencoder

# ─── Prepare Data ──────────────────────────────────────────────────────────────
def prepare_data():
    """
    Generate and preprocess data
    """
    # Generate data
    X, y = generate_smartgrid_data(5000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create LSTM sequences
    X_train_seq = create_sequences(X_train_scaled)
    X_test_seq = create_sequences(X_test_scaled)
    
    y_train_seq = y_train[SEQUENCE_LENGTH:]
    y_test_seq = y_test[SEQUENCE_LENGTH:]
    
    return {
        'X_train_lstm': X_train_seq,
        'X_test_lstm': X_test_seq,
        'y_train_lstm': y_train_seq,
        'y_test_lstm': y_test_seq,
        'X_train_ae': X_train_scaled,
        'X_test_ae': X_test_scaled,
        'scaler': scaler
    }

# ─── Train Models ──────────────────────────────────────────────────────────────
def train_models(data):
    """
    Train LSTM and Autoencoder models
    """
    print("\n" + "="*60)
    print("TRAINING LSTM MODEL")
    print("="*60)
    
    lstm_model = build_lstm_model((SEQUENCE_LENGTH, FEATURE_DIM))
    
    lstm_history = lstm_model.fit(
        data['X_train_lstm'], data['y_train_lstm'],
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # Evaluate LSTM
    lstm_loss, lstm_mae, lstm_acc = lstm_model.evaluate(data['X_test_lstm'], data['y_test_lstm'])
    print(f"\n✅ LSTM Test Accuracy: {lstm_acc:.4f}")
    print(f"   Test Loss: {lstm_loss:.4f}")
    print(f"   Test MAE: {lstm_mae:.4f}")
    
    print("\n" + "="*60)
    print("TRAINING AUTOENCODER MODEL")
    print("="*60)
    
    autoencoder = build_autoencoder(FEATURE_DIM)
    
    ae_history = autoencoder.fit(
        data['X_train_ae'], data['X_train_ae'],
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # Evaluate Autoencoder
    ae_loss = autoencoder.evaluate(data['X_test_ae'], data['X_test_ae'])
    print(f"\n✅ Autoencoder Test Loss (MSE): {ae_loss:.4f}")
    
    return lstm_model, autoencoder

# ─── Save Models ──────────────────────────────────────────────────────────────
def save_models(lstm_model, autoencoder, scaler):
    """
    Save trained models
    """
    models_dir = "backend/models"
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\n💾 Saving models to {models_dir}/...")
    
    lstm_model.save(f"{models_dir}/lstm_anomaly_detector.h5")
    autoencoder.save(f"{models_dir}/autoencoder.h5")
    joblib.dump(scaler, f"{models_dir}/scaler.pkl")
    
    print("✅ Models saved successfully!")

# ─── Main Training Pipeline ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SmartGrid Security - Model Training Pipeline")
    print("="*60)
    
    # Prepare data
    print("\n📊 Preparing data...")
    data = prepare_data()
    
    # Train models
    lstm_model, autoencoder = train_models(data)
    
    # Save models
    save_models(lstm_model, autoencoder, data['scaler'])
    
    print("\n" + "="*60)
    print("🎉 Training Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run the backend: python app_smartgrid.py")
    print("2. Open frontend: Open smartgrid.html in your browser")
    print("3. Analyze grid data with deep learning predictions")
