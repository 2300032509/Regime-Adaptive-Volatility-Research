"""
Script 7B: LSTM Model - Extended Time Period
Re-train LSTM to cover 2015-2024 including COVID crisis
This gives us crisis samples for the adaptive framework!

Author: Volatility Research Project - Extended Version
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

def load_data():
    """Load feature matrix with all engineered features"""
    print("Loading feature data...")
    
    train = pd.read_csv('data/features/train_data.csv', index_col=0, parse_dates=True)
    val = pd.read_csv('data/features/val_data.csv', index_col=0, parse_dates=True)
    test = pd.read_csv('data/features/test_data.csv', index_col=0, parse_dates=True)
    
    print(f"Train: {train.shape} ({train.index.min()} to {train.index.max()})")
    print(f"Val:   {val.shape} ({val.index.min()} to {val.index.max()})")
    print(f"Test:  {test.shape} ({test.index.min()} to {test.index.max()})")
    
    return train, val, test

def prepare_lstm_data(train, val, test, lookback=20):
    """
    Prepare data for LSTM with lookback window
    """
    print(f"\nPreparing LSTM sequences (lookback={lookback})...")
    
    # Separate features and target
    X_train = train.drop('target_volatility', axis=1)
    y_train = train['target_volatility']
    
    X_val = val.drop('target_volatility', axis=1)
    y_val = val['target_volatility']
    
    X_test = test.drop('target_volatility', axis=1)
    y_test = test['target_volatility']
    
    # Scale features
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scale target
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    
    # Create sequences
    def create_sequences(X, y, lookback):
        X_seq, y_seq, dates = [], [], []
        for i in range(lookback, len(X)):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, lookback)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, lookback)
    
    # Get corresponding dates (after lookback)
    train_dates = train.index[lookback:]
    val_dates = val.index[lookback:]
    test_dates = test.index[lookback:]
    
    print(f"Train sequences: {X_train_seq.shape}")
    print(f"Val sequences:   {X_val_seq.shape}")
    print(f"Test sequences:  {X_test_seq.shape}")
    
    return (X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq, 
            scaler_X, scaler_y, train_dates, val_dates, test_dates)

def build_lstm_model(input_shape, units=100, dropout=0.2):
    """
    Build LSTM architecture
    Simpler than before to handle COVID regime shift
    """
    print(f"\nBuilding LSTM model...")
    print(f"  Units: {units}")
    print(f"  Dropout: {dropout}")
    
    model = Sequential([
        # First LSTM layer
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        
        # Second LSTM layer
        LSTM(units//2, return_sequences=False),
        Dropout(dropout),
        
        # Dense layers
        Dense(50, activation='relu'),
        Dropout(dropout/2),
        
        Dense(1)  # Output
    ])
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    return model

def train_lstm(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Train LSTM with early stopping
    """
    print(f"\nTraining LSTM...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    return model, history

def generate_full_forecast(model, X_train_seq, X_val_seq, X_test_seq, 
                          scaler_y, train_dates, val_dates, test_dates):
    """
    Generate forecasts for ALL periods (train, val, test)
    This gives us COVID period forecasts!
    """
    print("\nGenerating forecasts for all periods...")
    
    # Predict on all sets
    y_train_pred_scaled = model.predict(X_train_seq, verbose=0)
    y_val_pred_scaled = model.predict(X_val_seq, verbose=0)
    y_test_pred_scaled = model.predict(X_test_seq, verbose=0)
    
    # Inverse transform
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).flatten()
    y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled).flatten()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()
    
    # Combine into single series
    all_forecasts = np.concatenate([y_train_pred, y_val_pred, y_test_pred])
    all_dates = train_dates.tolist() + val_dates.tolist() + test_dates.tolist()
    
    forecast_series = pd.Series(all_forecasts, index=all_dates)
    
    print(f"  Total forecasts: {len(forecast_series)}")
    print(f"  Date range: {forecast_series.index.min()} to {forecast_series.index.max()}")
    print(f"  Includes 2020: {(forecast_series.index.year == 2020).any()}")
    
    # Count by year
    print(f"\nForecasts by year:")
    for year in range(2015, 2025):
        count = (forecast_series.index.year == year).sum()
        print(f"  {year}: {count} samples")
    
    return forecast_series

def evaluate_model(model, X_test, y_test, scaler_y, test_dates, actual_vol):
    """
    Evaluate LSTM on test set
    """
    print("\nEvaluating LSTM on test set...")
    
    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = scaler_y.inverse_transform(y_pred_scaled).flatten()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
    
    # Directional accuracy
    actual_direction = np.sign(np.diff(y_test_orig))
    pred_direction = np.sign(np.diff(y_pred_orig))
    da = np.mean(actual_direction == pred_direction) * 100
    
    print(f"\nLSTM Performance (Test Period):")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  DA:   {da:.2f}%")
    
    metrics = {
        'Model': 'LSTM (Extended)',
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'DA': da,
        'N': len(y_test_orig)
    }
    
    return metrics

def plot_training_history(history, save_path='results/figures/18_lstm_extended_training.png'):
    """
    Plot training history
    """
    print("\nPlotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title('LSTM Training Loss (Extended Period)', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE
    ax2.plot(history.history['mae'], label='Train MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    ax2.set_title('LSTM Training MAE (Extended Period)', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def main():
    print("="*60)
    print("LSTM MODEL - EXTENDED TIME PERIOD")
    print("Including COVID Crisis Data (2020)")
    print("="*60)
    
    # Load data
    train, val, test = load_data()
    
    print("\n" + "="*60)
    print("STEP 1: PREPARE DATA")
    print("="*60)
    
    # Prepare sequences
    (X_train_seq, y_train_seq, X_val_seq, y_val_seq, 
     X_test_seq, y_test_seq, scaler_X, scaler_y,
     train_dates, val_dates, test_dates) = prepare_lstm_data(train, val, test, lookback=20)
    
    print("\n" + "="*60)
    print("STEP 2: BUILD MODEL")
    print("="*60)
    
    # Build model
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    model = build_lstm_model(input_shape, units=100, dropout=0.2)
    
    print("\n" + "="*60)
    print("STEP 3: TRAIN MODEL")
    print("="*60)
    
    # Train
    model, history = train_lstm(
        model, X_train_seq, y_train_seq, 
        X_val_seq, y_val_seq,
        epochs=100, batch_size=32
    )
    
    # Plot training history
    plot_training_history(history)
    
    print("\n" + "="*60)
    print("STEP 4: EVALUATE ON TEST SET")
    print("="*60)
    
    # Load actual volatility for evaluation
    vol_actual = pd.read_csv('data/processed/volatility_clean.csv', 
                             index_col=0, parse_dates=True)['SPY']
    
    # Evaluate
    metrics = evaluate_model(model, X_test_seq, y_test_seq, scaler_y, 
                            test_dates, vol_actual)
    
    print("\n" + "="*60)
    print("STEP 5: GENERATE FULL FORECAST (ALL PERIODS)")
    print("="*60)
    
    # Generate forecasts for ALL periods including COVID
    full_forecast = generate_full_forecast(
        model, X_train_seq, X_val_seq, X_test_seq,
        scaler_y, train_dates, val_dates, test_dates
    )
    
    # Save model
    model.save('results/saved_models/lstm_model_extended.keras')
    print("\nSaved model: results/saved_models/lstm_model_extended.keras")
    
    # Save full forecast
    full_forecast.to_csv('results/tables/forecast_lstm_extended.csv')
    print("Saved forecast: results/tables/forecast_lstm_extended.csv")
    
    print("\n" + "="*60)
    print("✅ LSTM EXTENDED MODEL COMPLETE!")
    print("="*60)
    
    print("\n🎯 KEY ACHIEVEMENT:")
    print(f"  ✓ Full forecast from {full_forecast.index.min()} to {full_forecast.index.max()}")
    print(f"  ✓ Includes COVID crisis period (2020)")
    
    # Count crisis samples
    prices = pd.read_csv('data/processed/prices_clean.csv', index_col=0, parse_dates=True)
    vix = prices['^VIX']
    
    # Align VIX with forecast
    common_idx = vix.index.intersection(full_forecast.index)
    vix_aligned = vix.loc[common_idx]
    
    crisis_samples = (vix_aligned >= 30).sum()
    covid_2020_samples = ((vix_aligned >= 30) & (vix_aligned.index.year == 2020)).sum()
    
    print(f"  ✓ Total crisis samples (VIX>30): {crisis_samples}")
    print(f"  ✓ COVID 2020 crisis samples: {covid_2020_samples}")
    
    print("\n🚀 Ready to re-run adaptive framework with COVID data!")
    print("Next: py -3.10 scripts\\11_regime_adaptive_extended.py")

if __name__ == "__main__":
    main()
