"""
Script 5: GARCH Family Models
Implement GARCH(1,1), EGARCH(1,1), and GJR-GARCH for volatility forecasting
Author: Volatility Research Project - Week 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load preprocessed volatility and returns data"""
    print("Loading data...")
    
    # Load returns (GARCH uses returns, not volatility directly)
    returns = pd.read_csv('data/processed/returns_clean.csv', index_col=0, parse_dates=True)
    
    # Load actual volatility for comparison
    vol_actual = pd.read_csv('data/processed/volatility_clean.csv', index_col=0, parse_dates=True)
    
    # Focus on SPY (S&P 500) as main target
    spy_returns = returns['SPY'].dropna() * 100  # GARCH library expects percentage returns
    spy_vol_actual = vol_actual['SPY'].dropna()
    
    print(f"Returns shape: {spy_returns.shape}")
    print(f"Date range: {spy_returns.index[0]} to {spy_returns.index[-1]}")
    
    return spy_returns, spy_vol_actual

def create_data_splits(returns, train_end='2019-12-31', val_end='2021-12-31'):
    """Split data into train, validation, test sets"""
    train = returns[returns.index <= train_end]
    val = returns[(returns.index > train_end) & (returns.index <= val_end)]
    test = returns[returns.index > val_end]
    
    print(f"\nData splits:")
    print(f"  Train: {len(train)} days ({train.index[0]} to {train.index[-1]})")
    print(f"  Val:   {len(val)} days ({val.index[0]} to {val.index[-1]})")
    print(f"  Test:  {len(test)} days ({test.index[0]} to {test.index[-1]})")
    
    return train, val, test

def fit_garch(returns, p=1, q=1, model_type='GARCH'):
    """
    Fit GARCH model
    p = ARCH order (lagged squared errors)
    q = GARCH order (lagged variance)
    """
    print(f"\nFitting {model_type}({p},{q})...")
    
    try:
        # Create model
        if model_type == 'GARCH':
            model = arch_model(returns, vol='Garch', p=p, q=q, rescale=False)
        elif model_type == 'EGARCH':
            model = arch_model(returns, vol='EGARCH', p=p, q=q, rescale=False)
        elif model_type == 'GJR-GARCH':
            model = arch_model(returns, vol='Garch', p=p, o=1, q=q, rescale=False)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Fit model
        result = model.fit(disp='off', show_warning=False)
        
        print(f"  Fitted successfully!")
        print(f"  AIC: {result.aic:.2f}")
        print(f"  BIC: {result.bic:.2f}")
        
        return result
    
    except Exception as e:
        print(f"  Error fitting {model_type}: {e}")
        return None

def rolling_forecast(returns, model_type='GARCH', window=1200):
    """
    Generate rolling window forecasts
    Use expanding window (all past data) for each forecast
    """
    print(f"\nGenerating {model_type} rolling forecasts...")
    
    forecasts = []
    forecast_dates = []
    
    # Start forecasting after minimum window
    start_idx = window
    
    for i in range(start_idx, len(returns)):
        # Use all data up to current point
        train_data = returns.iloc[:i]
        
        # Fit model
        try:
            if model_type == 'GARCH':
                model = arch_model(train_data, vol='Garch', p=1, q=1, rescale=False)
            elif model_type == 'EGARCH':
                model = arch_model(train_data, vol='EGARCH', p=1, q=1, rescale=False)
            elif model_type == 'GJR-GARCH':
                model = arch_model(train_data, vol='Garch', p=1, o=1, q=1, rescale=False)
            
            result = model.fit(disp='off', show_warning=False)
            
            # Forecast next day's volatility
            forecast = result.forecast(horizon=1)
            variance_forecast = forecast.variance.values[-1, 0]
            
            # Convert to annualized volatility (same scale as realized vol)
            volatility_forecast = np.sqrt(variance_forecast) * np.sqrt(252) / 100
            
            forecasts.append(volatility_forecast)
            forecast_dates.append(returns.index[i])
            
            # Progress indicator
            if (i - start_idx) % 100 == 0:
                pct_complete = ((i - start_idx) / (len(returns) - start_idx)) * 100
                print(f"  Progress: {pct_complete:.1f}%", end='\r')
        
        except Exception as e:
            # If fitting fails, use last forecast
            if len(forecasts) > 0:
                forecasts.append(forecasts[-1])
                forecast_dates.append(returns.index[i])
    
    print(f"\n  Generated {len(forecasts)} forecasts")
    
    # Create forecast series
    forecast_series = pd.Series(forecasts, index=forecast_dates)
    
    return forecast_series

def evaluate_forecasts(actual, forecast, model_name):
    """
    Evaluate forecast accuracy
    """
    # Align actual and forecast (same dates)
    common_idx = actual.index.intersection(forecast.index)
    actual_aligned = actual.loc[common_idx]
    forecast_aligned = forecast.loc[common_idx]
    
    # Calculate metrics
    errors = actual_aligned - forecast_aligned
    
    # RMSE
    rmse = np.sqrt(np.mean(errors**2))
    
    # MAE
    mae = np.mean(np.abs(errors))
    
    # MAPE
    mape = np.mean(np.abs(errors / actual_aligned)) * 100
    
    # Directional Accuracy
    actual_direction = np.sign(actual_aligned.diff())
    forecast_direction = np.sign(forecast_aligned.diff())
    da = np.mean(actual_direction[1:] == forecast_direction[1:]) * 100
    
    results = {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'DA': da,
        'N': len(common_idx)
    }
    
    return results

def plot_forecasts(actual, forecasts_dict, save_path='results/figures/07_garch_forecasts.png'):
    """
    Plot actual volatility vs GARCH forecasts
    """
    print("\nCreating forecast visualization...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    models = list(forecasts_dict.keys())
    
    for i, (model_name, forecast) in enumerate(forecasts_dict.items()):
        ax = axes[i]
        
        # Plot actual
        ax.plot(actual.index, actual, label='Actual Volatility', 
                color='black', linewidth=1.5, alpha=0.7)
        
        # Plot forecast
        ax.plot(forecast.index, forecast, label=f'{model_name} Forecast', 
                color='red', linewidth=1, alpha=0.8)
        
        # Highlight COVID crisis
        ax.axvspan('2020-03-01', '2020-06-01', alpha=0.2, color='orange', label='COVID Crisis')
        
        ax.set_title(f'{model_name} Volatility Forecasts', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Annualized Volatility')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def save_results(forecasts_dict, metrics_df):
    """Save forecast results"""
    print("\nSaving results...")
    
    # Save forecasts
    for model_name, forecast in forecasts_dict.items():
        filename = f'results/tables/forecast_{model_name.lower().replace("-", "_")}.csv'
        forecast.to_csv(filename)
        print(f"  Saved: {filename}")
    
    # Save metrics
    metrics_df.to_csv('results/tables/garch_models_performance.csv', index=False)
    print(f"  Saved: results/tables/garch_models_performance.csv")

def main():
    print("="*60)
    print("GARCH FAMILY MODELS - TRADITIONAL VOLATILITY FORECASTING")
    print("="*60)
    
    # Load data
    returns, vol_actual = load_data()
    
    # Create splits
    train_returns, val_returns, test_returns = create_data_splits(returns)
    
    print("\n" + "="*60)
    print("STEP 1: FIT GARCH MODELS ON TRAINING DATA")
    print("="*60)
    
    # Fit models on training data
    garch_result = fit_garch(train_returns, p=1, q=1, model_type='GARCH')
    egarch_result = fit_garch(train_returns, p=1, q=1, model_type='EGARCH')
    gjr_result = fit_garch(train_returns, p=1, q=1, model_type='GJR-GARCH')
    
    # Print parameter estimates
    if garch_result:
        print("\nGARCH(1,1) Parameters:")
        print(garch_result.params)
    
    print("\n" + "="*60)
    print("STEP 2: GENERATE ROLLING FORECASTS")
    print("="*60)
    
    # Generate forecasts using rolling window on ALL data
    # (This simulates real-time forecasting)
    forecast_garch = rolling_forecast(returns, model_type='GARCH', window=1200)
    forecast_egarch = rolling_forecast(returns, model_type='EGARCH', window=1200)
    forecast_gjr = rolling_forecast(returns, model_type='GJR-GARCH', window=1200)
    
    forecasts_dict = {
        'GARCH': forecast_garch,
        'EGARCH': forecast_egarch,
        'GJR-GARCH': forecast_gjr
    }
    
    print("\n" + "="*60)
    print("STEP 3: EVALUATE FORECAST ACCURACY")
    print("="*60)
    
    # Evaluate on test period only
    test_start = '2022-01-03'
    vol_actual_test = vol_actual[vol_actual.index >= test_start]
    
    metrics_list = []
    
    for model_name, forecast in forecasts_dict.items():
        forecast_test = forecast[forecast.index >= test_start]
        metrics = evaluate_forecasts(vol_actual_test, forecast_test, model_name)
        metrics_list.append(metrics)
        
        print(f"\n{model_name} Performance (Test Period):")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  MAE:  {metrics['MAE']:.6f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  DA:   {metrics['DA']:.2f}%")
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(metrics_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("STEP 4: VISUALIZE FORECASTS")
    print("="*60)
    
    # Plot forecasts
    plot_forecasts(vol_actual, forecasts_dict)
    
    # Save results
    save_results(forecasts_dict, metrics_df)
    
    print("\n" + "="*60)
    print("✅ GARCH MODELS COMPLETE!")
    print("="*60)
    print("\nKey Findings:")
    print(f"  • Best model (lowest RMSE): {metrics_df.loc[metrics_df['RMSE'].idxmin(), 'Model']}")
    print(f"  • Best RMSE: {metrics_df['RMSE'].min():.6f}")
    print(f"  • Best directional accuracy: {metrics_df['DA'].max():.2f}%")
    print("\nNext: Run ARIMA models for comparison")
    print("Command: py -3.10 scripts\\06_arima_models.py")

if __name__ == "__main__":
    main()
