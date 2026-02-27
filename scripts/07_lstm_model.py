"""
Script 7: LSTM Model - Standalone
Implement Long Short-Term Memory neural network for volatility forecasting
Author: Volatility Research Project - Week 3
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

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_data():
    """Load feature matrix with all engineered features"""
    print("Loading feature data...")
    
    train = pd.read_csv('data/features/train_data.csv', index_col=0, parse_dates=True)
    val = pd.read_csv('data/features/val_data.csv', index_col=0, parse_dates=True)
    test = pd.read_csv('data/features/test_data.csv', index_col=0, parse_dates=True)
    
    print(f"Train: {train.shape}")
    print(f"Val:   {val.shape}")
    print(f"Test:  {test.shape}")
    
    return train, val, test

def prepare_lstm_data(train, val, test, lookback=20):
    """
    Prepare data for LSTM
    - Use lookback window of volatility values
    - Scale data to [0,1] range
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
    
    # Create sequences for LSTM
    def create_sequences(X, y, lookback):
        X_seq, y_seq = [], []
        for i in range(lookback, len(X)):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, lookback)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, lookback)
    
    print(f"Train sequences: {X_train_seq.shape}")
    print(f"Val sequences:   {X_val_seq.shape}")
    print(f"Test sequences:  {X_test_seq.shape}")
    
    return (X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq, 
            scaler_X, scaler_y)

def build_lstm_model(input_shape, units=100, dropout=0.2):
    """
    Build LSTM architecture
    """
    print(f"\nBuilding LSTM model...")
    print(f"  Units: {units}")
    print(f"  Dropout: {dropout}")
    
    model = Sequential([
        # First LSTM layer with return sequences
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        
        # Second LSTM layer
        LSTM(units//2, return_sequences=False),
        Dropout(dropout),
        
        # Dense layers
        Dense(50, activation='relu'),
        Dropout(dropout/2),
        
        Dense(1)  # Output layer
    ])
    
    # Compile model
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
    Train LSTM model with early stopping
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

def plot_training_history(history, save_path='results/figures/09_lstm_training.png'):
    """
    Plot training and validation loss
    """
    print("\nPlotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title('Model Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE
    ax2.plot(history.history['mae'], label='Train MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    ax2.set_title('Model MAE', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def evaluate_model(model, X_test, y_test, scaler_y, test_dates):
    """
    Evaluate LSTM on test set
    """
    print("\nEvaluating LSTM model...")
    
    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform to original scale
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
    
    # Create forecast series
    forecast = pd.Series(y_pred_orig, index=test_dates)
    actual = pd.Series(y_test_orig, index=test_dates)
    
    metrics = {
        'Model': 'LSTM',
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'DA': da,
        'N': len(y_test_orig)
    }
    
    return forecast, actual, metrics

def plot_lstm_forecast(actual, forecast, save_path='results/figures/10_lstm_forecast.png'):
    """
    Plot LSTM forecast vs actual
    """
    print("\nPlotting LSTM forecast...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot actual
    ax.plot(actual.index, actual, label='Actual Volatility', 
            color='black', linewidth=1.5, alpha=0.7)
    
    # Plot forecast
    ax.plot(forecast.index, forecast, label='LSTM Forecast', 
            color='purple', linewidth=1, alpha=0.8)
    
    ax.set_title('LSTM Volatility Forecasts', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Annualized Volatility', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def compare_with_baseline(lstm_metrics):
    """
    Compare LSTM with traditional models
    """
    print("\n" + "="*60)
    print("COMPARISON: LSTM vs TRADITIONAL MODELS")
    print("="*60)
    
    try:
        # Load traditional models results
        trad_metrics = pd.read_csv('results/tables/traditional_models_comparison.csv')
        
        # Add LSTM
        lstm_df = pd.DataFrame([lstm_metrics])
        all_metrics = pd.concat([lstm_df, trad_metrics], ignore_index=True)
        
        # Sort by RMSE
        all_metrics_sorted = all_metrics.sort_values('RMSE')
        
        print("\nAll Models Performance (sorted by RMSE):")
        print(all_metrics_sorted.to_string(index=False))
        
        # Calculate improvement
        best_trad_rmse = trad_metrics['RMSE'].min()
        lstm_rmse = lstm_metrics['RMSE']
        improvement = ((best_trad_rmse - lstm_rmse) / best_trad_rmse) * 100
        
        print(f"\n📊 LSTM Improvement over Best Traditional Model:")
        print(f"  Traditional best: {best_trad_rmse:.6f}")
        print(f"  LSTM:            {lstm_rmse:.6f}")
        print(f"  Improvement:     {improvement:.2f}%")
        
        # Save
        all_metrics_sorted.to_csv('results/tables/all_models_comparison.csv', index=False)
        print(f"\nSaved: results/tables/all_models_comparison.csv")
        
        return all_metrics_sorted
        
    except FileNotFoundError:
        print("\nTraditional models results not found.")
        return pd.DataFrame([lstm_metrics])

def main():
    print("="*60)
    print("LSTM MODEL - DEEP LEARNING VOLATILITY FORECASTING")
    print("="*60)
    
    # Load data
    train, val, test = load_data()
    
    # Store test dates for later
    test_dates = test.index[20:]  # After lookback window
    
    print("\n" + "="*60)
    print("STEP 1: PREPARE DATA FOR LSTM")
    print("="*60)
    
    # Prepare sequences
    (X_train_seq, y_train_seq, X_val_seq, y_val_seq, 
     X_test_seq, y_test_seq, scaler_X, scaler_y) = prepare_lstm_data(
        train, val, test, lookback=20
    )
    
    print("\n" + "="*60)
    print("STEP 2: BUILD LSTM MODEL")
    print("="*60)
    
    # Build model
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    model = build_lstm_model(input_shape, units=100, dropout=0.2)
    
    print("\n" + "="*60)
    print("STEP 3: TRAIN LSTM")
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
    
    # Evaluate
    forecast, actual, metrics = evaluate_model(
        model, X_test_seq, y_test_seq, scaler_y, test_dates
    )
    
    # Plot forecast
    plot_lstm_forecast(actual, forecast)
    
    # Save model
    model.save('results/saved_models/lstm_model.keras')
    print("\nSaved model: results/saved_models/lstm_model.keras")
    
    # Save forecast
    forecast.to_csv('results/tables/forecast_lstm.csv')
    print("Saved forecast: results/tables/forecast_lstm.csv")
    
    # Compare with baseline
    all_metrics = compare_with_baseline(metrics)
    
    print("\n" + "="*60)
    print("✅ LSTM MODEL COMPLETE!")
    print("="*60)
    
    # Check if LSTM beat baseline
    best_model = all_metrics.iloc[0]['Model']
    if best_model == 'LSTM':
        print("\n🎉 LSTM IS THE NEW CHAMPION!")
        print(f"  LSTM beat all traditional models!")
    else:
        print(f"\n📊 Current best: {best_model}")
        print(f"  LSTM performed well but traditional model still leads")
    
    print("\nWeek 3 Progress: LSTM standalone DONE ✅")
    print("Next: Hyperparameter tuning to optimize LSTM")

if __name__ == "__main__":
    # Create saved_models directory if it doesn't exist
    import os
    os.makedirs('results/saved_models', exist_ok=True)
    
    main()
