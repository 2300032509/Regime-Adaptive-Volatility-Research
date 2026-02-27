"""
Script 8: Hybrid GARCH-LSTM Model
Combine GARCH baseline with LSTM residual learning
This is the CORE INNOVATION of the research!

Two-stage approach:
1. GARCH captures baseline volatility patterns
2. LSTM learns from GARCH residuals (what GARCH missed)
3. Final forecast = GARCH + LSTM correction

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
from tensorflow.keras.callbacks import EarlyStopping
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
    
    spy_returns = returns['SPY'].dropna() * 100  # Percentage
    spy_vol = vol_actual['SPY'].dropna()
    
    return spy_returns, spy_vol

def split_data(data, train_end='2019-12-31', val_end='2021-12-31'):
    """Split into train/val/test"""
    train = data[data.index <= train_end]
    val = data[(data.index > train_end) & (data.index <= val_end)]
    test = data[data.index > val_end]
    
    return train, val, test

def fit_garch_model(returns_train):
    """
    Stage 1: Fit GARCH model
    """
    print("\n" + "="*60)
    print("STAGE 1: FIT GARCH MODEL")
    print("="*60)
    
    print("\nFitting GARCH(1,1) on training data...")
    model = arch_model(returns_train, vol='Garch', p=1, q=1, rescale=False)
    result = model.fit(disp='off', show_warning=False)
    
    print(f"  AIC: {result.aic:.2f}")
    print(f"  Parameters:")
    print(result.params)
    
    return result

def generate_garch_forecasts_and_residuals(returns, garch_result, window=1200):
    """
    Generate GARCH forecasts and calculate residuals
    Residual = Actual volatility - GARCH forecast
    """
    print("\n" + "="*60)
    print("STAGE 2: GENERATE GARCH FORECASTS & RESIDUALS")
    print("="*60)
    
    print("\nGenerating GARCH rolling forecasts...")
    
    garch_forecasts = []
    forecast_dates = []
    
    for i in range(window, len(returns)):
        train_data = returns.iloc[:i]
        
        try:
            model = arch_model(train_data, vol='Garch', p=1, q=1, rescale=False)
            result = model.fit(disp='off', show_warning=False)
            
            # Forecast next day
            forecast = result.forecast(horizon=1)
            variance = forecast.variance.values[-1, 0]
            volatility = np.sqrt(variance) * np.sqrt(252) / 100  # Annualized
            
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
    """
    Calculate GARCH residuals
    Residual = Actual - GARCH forecast
    """
    print("\nCalculating GARCH residuals...")
    
    # Align dates
    common_idx = vol_actual.index.intersection(garch_forecasts.index)
    actual_aligned = vol_actual.loc[common_idx]
    garch_aligned = garch_forecasts.loc[common_idx]
    
    # Residuals
    residuals = actual_aligned - garch_aligned
    
    print(f"  Residuals calculated: {len(residuals)} values")
    print(f"  Mean residual: {residuals.mean():.6f}")
    print(f"  Std residual: {residuals.std():.6f}")
    
    return residuals

def prepare_lstm_for_residuals(residuals, lookback=20):
    """
    Prepare residual data for LSTM
    """
    print("\n" + "="*60)
    print("STAGE 3: PREPARE LSTM FOR RESIDUAL LEARNING")
    print("="*60)
    
    print(f"\nCreating sequences (lookback={lookback})...")
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(residuals)):
        X.append(residuals.iloc[i-lookback:i].values)
        y.append(residuals.iloc[i])
    
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    
    # Split into train/val/test
    # Use same dates as original splits
    train_end = '2019-12-31'
    val_end = '2021-12-31'
    
    train_mask = residuals.index[lookback:] <= train_end
    val_mask = (residuals.index[lookback:] > train_end) & (residuals.index[lookback:] <= val_end)
    test_mask = residuals.index[lookback:] > val_end
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Scale
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, lookback)).reshape(-1, lookback, 1)
    X_val_scaled = scaler.transform(X_val.reshape(-1, lookback)).reshape(-1, lookback, 1)
    X_test_scaled = scaler.transform(X_test.reshape(-1, lookback)).reshape(-1, lookback, 1)
    
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"  Train: {X_train_scaled.shape}")
    print(f"  Val:   {X_val_scaled.shape}")
    print(f"  Test:  {X_test_scaled.shape}")
    
    return (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 
            X_test_scaled, y_test_scaled, scaler, scaler_y,
            residuals.index[lookback:][test_mask])

def build_residual_lstm(lookback, units=50):
    """
    Build LSTM for residual learning
    Simpler architecture since we're only learning residuals
    """
    print("\nBuilding residual LSTM...")
    print(f"  Units: {units}")
    
    model = Sequential([
        LSTM(units, input_shape=(lookback, 1)),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    return model

def train_residual_lstm(model, X_train, y_train, X_val, y_val, epochs=50):
    """
    Train LSTM on residuals
    """
    print("\n" + "="*60)
    print("STAGE 4: TRAIN LSTM ON RESIDUALS")
    print("="*60)
    
    print(f"\nTraining LSTM on GARCH residuals...")
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    return model, history

def generate_hybrid_forecasts(garch_forecasts, lstm_residual_predictions, test_dates):
    """
    Combine GARCH + LSTM predictions
    Final forecast = GARCH baseline + LSTM correction
    """
    print("\n" + "="*60)
    print("STAGE 5: COMBINE GARCH + LSTM")
    print("="*60)
    
    print("\nGenerating hybrid forecasts...")
    
    # Get GARCH forecasts for test period
    garch_test = garch_forecasts.loc[test_dates]
    
    # Combine: Hybrid = GARCH + LSTM_residual
    hybrid_forecast = garch_test.values + lstm_residual_predictions
    hybrid_series = pd.Series(hybrid_forecast, index=test_dates)
    
    print(f"  Generated {len(hybrid_series)} hybrid forecasts")
    print(f"  GARCH mean: {garch_test.mean():.6f}")
    print(f"  LSTM correction mean: {lstm_residual_predictions.mean():.6f}")
    print(f"  Hybrid mean: {hybrid_series.mean():.6f}")
    
    return hybrid_series, garch_test

def evaluate_hybrid(actual, hybrid_forecast, garch_forecast, model_name='Hybrid GARCH-LSTM'):
    """
    Evaluate hybrid model
    """
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    # Align
    common_idx = actual.index.intersection(hybrid_forecast.index)
    actual_aligned = actual.loc[common_idx]
    hybrid_aligned = hybrid_forecast.loc[common_idx]
    garch_aligned = garch_forecast.loc[common_idx]
    
    # Calculate metrics for hybrid
    errors_hybrid = actual_aligned - hybrid_aligned
    rmse_hybrid = np.sqrt(np.mean(errors_hybrid**2))
    mae_hybrid = np.mean(np.abs(errors_hybrid))
    mape_hybrid = np.mean(np.abs(errors_hybrid / actual_aligned)) * 100
    
    # Directional accuracy
    actual_dir = np.sign(actual_aligned.diff())
    hybrid_dir = np.sign(hybrid_aligned.diff())
    da_hybrid = np.mean(actual_dir[1:] == hybrid_dir[1:]) * 100
    
    # Calculate metrics for GARCH (for comparison)
    errors_garch = actual_aligned - garch_aligned
    rmse_garch = np.sqrt(np.mean(errors_garch**2))
    
    print(f"\n{model_name} Performance:")
    print(f"  RMSE: {rmse_hybrid:.6f}")
    print(f"  MAE:  {mae_hybrid:.6f}")
    print(f"  MAPE: {mape_hybrid:.2f}%")
    print(f"  DA:   {da_hybrid:.2f}%")
    
    print(f"\nImprovement over GARCH alone:")
    improvement = ((rmse_garch - rmse_hybrid) / rmse_garch) * 100
    print(f"  GARCH RMSE: {rmse_garch:.6f}")
    print(f"  Hybrid RMSE: {rmse_hybrid:.6f}")
    print(f"  Improvement: {improvement:.2f}%")
    
    metrics = {
        'Model': model_name,
        'RMSE': rmse_hybrid,
        'MAE': mae_hybrid,
        'MAPE': mape_hybrid,
        'DA': da_hybrid,
        'N': len(common_idx)
    }
    
    return metrics

def plot_hybrid_comparison(actual, garch, hybrid, save_path='results/figures/11_hybrid_model.png'):
    """
    Plot actual vs GARCH vs Hybrid
    """
    print("\nCreating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top: Full comparison
    ax1.plot(actual.index, actual, label='Actual', color='black', linewidth=1.5, alpha=0.7)
    ax1.plot(garch.index, garch, label='GARCH', color='red', linewidth=1, alpha=0.6)
    ax1.plot(hybrid.index, hybrid, label='Hybrid GARCH-LSTM', color='purple', linewidth=1.2, alpha=0.8)
    
    ax1.set_title('Hybrid GARCH-LSTM Model Performance', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Annualized Volatility')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Forecast errors
    garch_errors = actual.loc[garch.index] - garch
    hybrid_errors = actual.loc[hybrid.index] - hybrid
    
    ax2.plot(garch_errors.index, garch_errors, label='GARCH Error', color='red', alpha=0.6)
    ax2.plot(hybrid_errors.index, hybrid_errors, label='Hybrid Error', color='purple', alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    ax2.set_title('Forecast Errors', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Forecast Error')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def compare_all_models(hybrid_metrics):
    """
    Compare hybrid with all previous models
    """
    print("\n" + "="*60)
    print("COMPARISON: HYBRID vs ALL MODELS")
    print("="*60)
    
    try:
        all_models = pd.read_csv('results/tables/all_models_comparison.csv')
        
        # Add hybrid
        hybrid_df = pd.DataFrame([hybrid_metrics])
        all_models = pd.concat([hybrid_df, all_models], ignore_index=True)
        
        # Sort by RMSE
        all_models_sorted = all_models.sort_values('RMSE')
        
        print("\nAll Models (sorted by RMSE):")
        print(all_models_sorted.to_string(index=False))
        
        # Save
        all_models_sorted.to_csv('results/tables/all_models_with_hybrid.csv', index=False)
        print(f"\nSaved: results/tables/all_models_with_hybrid.csv")
        
        # Check if hybrid is best
        best_model = all_models_sorted.iloc[0]['Model']
        best_rmse = all_models_sorted.iloc[0]['RMSE']
        
        if best_model == 'Hybrid GARCH-LSTM':
            print("\n🎉 HYBRID MODEL IS THE WINNER!")
            print(f"  RMSE: {best_rmse:.6f}")
        else:
            print(f"\n📊 Current champion: {best_model}")
            print(f"  Best RMSE: {best_rmse:.6f}")
            print(f"  Hybrid RMSE: {hybrid_metrics['RMSE']:.6f}")
            print(f"  Gap: {((hybrid_metrics['RMSE'] - best_rmse) / best_rmse * 100):.2f}%")
        
        return all_models_sorted
        
    except FileNotFoundError:
        print("\nPrevious results not found.")
        return pd.DataFrame([hybrid_metrics])

def main():
    print("="*60)
    print("HYBRID GARCH-LSTM MODEL")
    print("Core Innovation: Two-Stage Volatility Forecasting")
    print("="*60)
    
    # Load data
    returns, vol_actual = load_data()
    
    # Split
    returns_train, returns_val, returns_test = split_data(returns)
    
    # Stage 1: Fit GARCH
    garch_result = fit_garch_model(returns_train)
    
    # Stage 2: Generate GARCH forecasts
    garch_forecasts = generate_garch_forecasts_and_residuals(returns, garch_result)
    
    # Calculate residuals
    residuals = calculate_residuals(vol_actual, garch_forecasts)
    
    # Stage 3: Prepare LSTM data
    (X_train, y_train, X_val, y_val, X_test, y_test, 
     scaler, scaler_y, test_dates) = prepare_lstm_for_residuals(residuals)
    
    # Build LSTM for residuals
    lstm_model = build_residual_lstm(lookback=20, units=50)
    
    # Stage 4: Train LSTM
    lstm_model, history = train_residual_lstm(lstm_model, X_train, y_train, X_val, y_val)
    
    # Predict residuals on test set
    print("\nPredicting residuals on test set...")
    residual_pred_scaled = lstm_model.predict(X_test, verbose=0)
    residual_pred = scaler_y.inverse_transform(residual_pred_scaled).flatten()
    
    # Stage 5: Combine GARCH + LSTM
    hybrid_forecast, garch_test = generate_hybrid_forecasts(
        garch_forecasts, residual_pred, test_dates
    )
    
    # Evaluate
    vol_test = vol_actual[vol_actual.index >= '2022-01-03']
    metrics = evaluate_hybrid(vol_test, hybrid_forecast, garch_test)
    
    # Visualize
    plot_hybrid_comparison(vol_test, garch_test, hybrid_forecast)
    
    # Save
    hybrid_forecast.to_csv('results/tables/forecast_hybrid_garch_lstm.csv')
    print("\nSaved: results/tables/forecast_hybrid_garch_lstm.csv")
    
    # Compare with all models
    all_models = compare_all_models(metrics)
    
    print("\n" + "="*60)
    print("✅ HYBRID MODEL COMPLETE!")
    print("="*60)
    print("\nCore Innovation Implemented:")
    print("  ✓ GARCH captures baseline volatility")
    print("  ✓ LSTM learns residual patterns")
    print("  ✓ Hybrid combines both strengths")
    print("\nWeek 4 Progress: Hybrid model DONE ✅")
    print("Next: Adaptive regime-switching framework")

if __name__ == "__main__":
    main()
