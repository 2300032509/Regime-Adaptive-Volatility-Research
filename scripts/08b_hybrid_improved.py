"""
Script 8B: Hybrid GARCH-LSTM Model (IMPROVED)
More training samples by starting GARCH forecasts earlier
Author: Volatility Research Project - Week 4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

def load_data():
    """Load returns and actual volatility"""
    print("Loading data...")
    
    returns = pd.read_csv('data/processed/returns_clean.csv', index_col=0, parse_dates=True)
    vol_actual = pd.read_csv('data/processed/volatility_clean.csv', index_col=0, parse_dates=True)
    
    spy_returns = returns['SPY'].dropna() * 100
    spy_vol = vol_actual['SPY'].dropna()
    
    return spy_returns, spy_vol

def generate_garch_forecasts_and_residuals(returns, window=500):
    """
    Generate GARCH forecasts starting EARLIER
    window=500 instead of 1200 gives us more training data
    """
    print("\n" + "="*60)
    print("GENERATING GARCH FORECASTS (IMPROVED)")
    print("="*60)
    
    print(f"\nStarting GARCH forecasts after {window} days...")
    print("(This gives LSTM more training samples!)")
    
    garch_forecasts = []
    forecast_dates = []
    
    for i in range(window, len(returns)):
        train_data = returns.iloc[:i]
        
        try:
            model = arch_model(train_data, vol='Garch', p=1, q=1, rescale=False)
            result = model.fit(disp='off', show_warning=False)
            
            forecast = result.forecast(horizon=1)
            variance = forecast.variance.values[-1, 0]
            volatility = np.sqrt(variance) * np.sqrt(252) / 100
            
            garch_forecasts.append(volatility)
            forecast_dates.append(returns.index[i])
            
            if (i - window) % 100 == 0:
                pct = ((i - window) / (len(returns) - window)) * 100
                print(f"  Progress: {pct:.1f}%", end='\r')
        except:
            if len(garch_forecasts) > 0:
                garch_forecasts.append(garch_forecasts[-1])
                forecast_dates.append(returns.index[i])
    
    print(f"\n  Generated {len(garch_forecasts)} GARCH forecasts")
    
    garch_series = pd.Series(garch_forecasts, index=forecast_dates)
    
    return garch_series

def calculate_residuals(vol_actual, garch_forecasts):
    """Calculate GARCH residuals"""
    print("\nCalculating GARCH residuals...")
    
    common_idx = vol_actual.index.intersection(garch_forecasts.index)
    actual_aligned = vol_actual.loc[common_idx]
    garch_aligned = garch_forecasts.loc[common_idx]
    
    residuals = actual_aligned - garch_aligned
    
    print(f"  Residuals: {len(residuals)} values")
    print(f"  Mean: {residuals.mean():.6f}")
    print(f"  Std: {residuals.std():.6f}")
    
    return residuals

def prepare_lstm_for_residuals(residuals, lookback=20):
    """Prepare LSTM data with proper train/val/test split"""
    print("\n" + "="*60)
    print("PREPARING LSTM DATA")
    print("="*60)
    
    print(f"\nCreating sequences (lookback={lookback})...")
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(residuals)):
        X.append(residuals.iloc[i-lookback:i].values)
        y.append(residuals.iloc[i])
    
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    
    # Split by date
    train_end = '2019-12-31'
    val_end = '2021-12-31'
    
    dates = residuals.index[lookback:]
    train_mask = dates <= train_end
    val_mask = (dates > train_end) & (dates <= val_end)
    test_mask = dates > val_end
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\n  Train: {X_train.shape} samples")
    print(f"  Val:   {X_val.shape} samples")
    print(f"  Test:  {X_test.shape} samples")
    
    # Check if we have enough training data
    if len(X_train) < 100:
        print(f"\n  ⚠️  WARNING: Only {len(X_train)} training samples!")
        print("  Consider reducing window size further")
    else:
        print(f"\n  ✓ Good: {len(X_train)} training samples!")
    
    # Scale
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, lookback)).reshape(-1, lookback, 1)
    X_val_scaled = scaler.transform(X_val.reshape(-1, lookback)).reshape(-1, lookback, 1)
    X_test_scaled = scaler.transform(X_test.reshape(-1, lookback)).reshape(-1, lookback, 1)
    
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    return (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 
            X_test_scaled, y_test_scaled, scaler, scaler_y, dates[test_mask])

def build_residual_lstm(lookback, units=50):
    """Build LSTM for residual learning"""
    print("\n" + "="*60)
    print("BUILDING LSTM MODEL")
    print("="*60)
    
    print(f"\nArchitecture: {units} units, dropout 0.2")
    
    model = Sequential([
        LSTM(units, input_shape=(lookback, 1)),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dropout(0.1),
        Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    model.summary()
    
    return model

def train_residual_lstm(model, X_train, y_train, X_val, y_val, epochs=100):
    """Train LSTM on residuals"""
    print("\n" + "="*60)
    print("TRAINING LSTM ON RESIDUALS")
    print("="*60)
    
    print(f"\nEpochs: {epochs}")
    
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
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    return model, history

def plot_training_history(history, save_path='results/figures/12_hybrid_improved_training.png'):
    """Plot training history"""
    print("\nPlotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title('LSTM Training Loss (Residuals)', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['mae'], label='Train MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    ax2.set_title('LSTM Training MAE (Residuals)', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def generate_hybrid_forecasts(garch_forecasts, lstm_residual_predictions, test_dates):
    """Combine GARCH + LSTM"""
    print("\n" + "="*60)
    print("COMBINING GARCH + LSTM")
    print("="*60)
    
    garch_test = garch_forecasts.loc[test_dates]
    hybrid_forecast = garch_test.values + lstm_residual_predictions
    hybrid_series = pd.Series(hybrid_forecast, index=test_dates)
    
    print(f"\n  Generated {len(hybrid_series)} hybrid forecasts")
    print(f"  GARCH baseline mean: {garch_test.mean():.6f}")
    print(f"  LSTM correction mean: {lstm_residual_predictions.mean():.6f}")
    print(f"  Hybrid final mean: {hybrid_series.mean():.6f}")
    
    return hybrid_series, garch_test

def evaluate_hybrid(actual, hybrid_forecast, garch_forecast):
    """Evaluate hybrid model"""
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    common_idx = actual.index.intersection(hybrid_forecast.index)
    actual_aligned = actual.loc[common_idx]
    hybrid_aligned = hybrid_forecast.loc[common_idx]
    garch_aligned = garch_forecast.loc[common_idx]
    
    # Hybrid metrics
    errors_hybrid = actual_aligned - hybrid_aligned
    rmse_hybrid = np.sqrt(np.mean(errors_hybrid**2))
    mae_hybrid = np.mean(np.abs(errors_hybrid))
    mape_hybrid = np.mean(np.abs(errors_hybrid / actual_aligned)) * 100
    
    actual_dir = np.sign(actual_aligned.diff())
    hybrid_dir = np.sign(hybrid_aligned.diff())
    da_hybrid = np.mean(actual_dir[1:] == hybrid_dir[1:]) * 100
    
    # GARCH metrics (for comparison)
    errors_garch = actual_aligned - garch_aligned
    rmse_garch = np.sqrt(np.mean(errors_garch**2))
    
    print(f"\nHybrid GARCH-LSTM (Improved):")
    print(f"  RMSE: {rmse_hybrid:.6f}")
    print(f"  MAE:  {mae_hybrid:.6f}")
    print(f"  MAPE: {mape_hybrid:.2f}%")
    print(f"  DA:   {da_hybrid:.2f}%")
    
    improvement = ((rmse_garch - rmse_hybrid) / rmse_garch) * 100
    print(f"\nImprovement over GARCH:")
    print(f"  GARCH RMSE: {rmse_garch:.6f}")
    print(f"  Hybrid RMSE: {rmse_hybrid:.6f}")
    print(f"  Improvement: {improvement:.2f}%")
    
    metrics = {
        'Model': 'Hybrid GARCH-LSTM v2',
        'RMSE': rmse_hybrid,
        'MAE': mae_hybrid,
        'MAPE': mape_hybrid,
        'DA': da_hybrid,
        'N': len(common_idx)
    }
    
    return metrics

def plot_hybrid_comparison(actual, garch, hybrid, save_path='results/figures/13_hybrid_improved_forecast.png'):
    """Plot comparison"""
    print("\nCreating visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(actual.index, actual, label='Actual', color='black', linewidth=1.5, alpha=0.7)
    ax.plot(garch.index, garch, label='GARCH', color='red', linewidth=1, alpha=0.6)
    ax.plot(hybrid.index, hybrid, label='Hybrid v2 (More Training Data)', 
            color='purple', linewidth=1.2, alpha=0.8)
    
    ax.set_title('Improved Hybrid GARCH-LSTM (More Training Samples)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Annualized Volatility')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def compare_all_models(hybrid_metrics):
    """Compare with all models"""
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    
    try:
        all_models = pd.read_csv('results/tables/all_models_with_hybrid.csv')
        
        # Add improved hybrid
        hybrid_df = pd.DataFrame([hybrid_metrics])
        all_models = pd.concat([hybrid_df, all_models], ignore_index=True)
        
        # Sort
        all_models_sorted = all_models.sort_values('RMSE')
        
        print("\nAll Models (sorted by RMSE):")
        print(all_models_sorted.to_string(index=False))
        
        # Save
        all_models_sorted.to_csv('results/tables/all_models_final.csv', index=False)
        print(f"\nSaved: results/tables/all_models_final.csv")
        
        # Check winner
        best_model = all_models_sorted.iloc[0]['Model']
        best_rmse = all_models_sorted.iloc[0]['RMSE']
        
        if 'Hybrid' in best_model:
            print("\n🎉🎉🎉 HYBRID MODEL WINS! 🎉🎉🎉")
            print(f"  {best_model}")
            print(f"  RMSE: {best_rmse:.6f}")
        else:
            print(f"\n📊 Best model: {best_model}")
            print(f"  RMSE: {best_rmse:.6f}")
            
            if 'Hybrid' in hybrid_metrics['Model']:
                hybrid_rmse = hybrid_metrics['RMSE']
                gap = ((hybrid_rmse - best_rmse) / best_rmse) * 100
                print(f"\n  Hybrid v2 RMSE: {hybrid_rmse:.6f}")
                print(f"  Gap to leader: {gap:.2f}%")
        
        return all_models_sorted
        
    except FileNotFoundError:
        return pd.DataFrame([hybrid_metrics])

def main():
    print("="*60)
    print("HYBRID GARCH-LSTM MODEL v2 (IMPROVED)")
    print("More Training Data = Better LSTM Learning")
    print("="*60)
    
    # Load data
    returns, vol_actual = load_data()
    
    # Generate GARCH forecasts (starting earlier!)
    print("\nKey improvement: Starting GARCH forecasts at day 500 instead of 1200")
    print("This gives LSTM ~700-800 training samples instead of 38!\n")
    
    garch_forecasts = generate_garch_forecasts_and_residuals(returns, window=500)
    
    # Calculate residuals
    residuals = calculate_residuals(vol_actual, garch_forecasts)
    
    # Prepare LSTM data
    (X_train, y_train, X_val, y_val, X_test, y_test, 
     scaler, scaler_y, test_dates) = prepare_lstm_for_residuals(residuals, lookback=20)
    
    # Build and train LSTM
    lstm_model = build_residual_lstm(lookback=20, units=50)
    lstm_model, history = train_residual_lstm(lstm_model, X_train, y_train, X_val, y_val)
    
    # Plot training
    plot_training_history(history)
    
    # Predict
    print("\nPredicting residuals on test set...")
    residual_pred_scaled = lstm_model.predict(X_test, verbose=0)
    residual_pred = scaler_y.inverse_transform(residual_pred_scaled).flatten()
    
    # Combine
    hybrid_forecast, garch_test = generate_hybrid_forecasts(
        garch_forecasts, residual_pred, test_dates
    )
    
    # Evaluate
    vol_test = vol_actual[vol_actual.index >= '2022-01-03']
    metrics = evaluate_hybrid(vol_test, hybrid_forecast, garch_test)
    
    # Visualize
    plot_hybrid_comparison(vol_test, garch_test, hybrid_forecast)
    
    # Save
    hybrid_forecast.to_csv('results/tables/forecast_hybrid_v2.csv')
    print("\nSaved: results/tables/forecast_hybrid_v2.csv")
    
    # Compare
    all_models = compare_all_models(metrics)
    
    print("\n" + "="*60)
    print("✅ IMPROVED HYBRID MODEL COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
