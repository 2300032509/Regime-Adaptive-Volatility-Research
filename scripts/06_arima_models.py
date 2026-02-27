"""
Script 6: ARIMA Models
Implement ARIMA for realized volatility forecasting (baseline comparison)
Author: Volatility Research Project - Week 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load realized volatility data"""
    print("Loading volatility data...")
    
    vol = pd.read_csv('data/processed/volatility_clean.csv', index_col=0, parse_dates=True)
    spy_vol = vol['SPY'].dropna()
    
    print(f"Volatility shape: {spy_vol.shape}")
    print(f"Date range: {spy_vol.index[0]} to {spy_vol.index[-1]}")
    
    return spy_vol

def test_stationarity(data):
    """
    Test if time series is stationary using Augmented Dickey-Fuller test
    """
    print("\nTesting stationarity (ADF test)...")
    
    result = adfuller(data.dropna())
    
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  Critical values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.4f}")
    
    if result[1] < 0.05:
        print("  → Series is STATIONARY (p < 0.05)")
        return True
    else:
        print("  → Series is NON-STATIONARY (p >= 0.05)")
        print("  → Will use differencing (d=1)")
        return False

def find_optimal_arima_params(data, max_p=5, max_q=5):
    """
    Find optimal ARIMA parameters using AIC
    """
    print("\nSearching for optimal ARIMA parameters...")
    
    best_aic = np.inf
    best_params = None
    
    # Test stationarity to determine d
    is_stationary = test_stationarity(data)
    d_values = [0] if is_stationary else [0, 1]
    
    results = []
    
    for d in d_values:
        for p in range(0, max_p + 1):
            for q in range(0, max_q + 1):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    result = model.fit()
                    
                    results.append({
                        'p': p, 'd': d, 'q': q,
                        'AIC': result.aic,
                        'BIC': result.bic
                    })
                    
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_params = (p, d, q)
                
                except:
                    continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).sort_values('AIC')
    
    print(f"\nTop 5 ARIMA models by AIC:")
    print(results_df.head().to_string(index=False))
    
    print(f"\nOptimal ARIMA parameters: {best_params}")
    print(f"  AIC: {best_aic:.2f}")
    
    return best_params

def fit_arima(data, order):
    """
    Fit ARIMA model with given order
    """
    print(f"\nFitting ARIMA{order}...")
    
    try:
        model = ARIMA(data, order=order)
        result = model.fit()
        
        print(f"  Fitted successfully!")
        print(f"  AIC: {result.aic:.2f}")
        print(f"  BIC: {result.bic:.2f}")
        
        return result
    
    except Exception as e:
        print(f"  Error fitting ARIMA: {e}")
        return None

def rolling_forecast_arima(data, order, window=1200):
    """
    Generate rolling window forecasts using ARIMA
    """
    print(f"\nGenerating ARIMA{order} rolling forecasts...")
    
    forecasts = []
    forecast_dates = []
    
    # Start forecasting after minimum window
    start_idx = window
    
    for i in range(start_idx, len(data)):
        # Use all data up to current point
        train_data = data.iloc[:i]
        
        try:
            # Fit ARIMA
            model = ARIMA(train_data, order=order)
            result = model.fit()
            
            # Forecast next day
            forecast = result.forecast(steps=1)
            
            forecasts.append(forecast.iloc[0])
            forecast_dates.append(data.index[i])
            
            # Progress indicator
            if (i - start_idx) % 100 == 0:
                pct_complete = ((i - start_idx) / (len(data) - start_idx)) * 100
                print(f"  Progress: {pct_complete:.1f}%", end='\r')
        
        except Exception as e:
            # If fitting fails, use last forecast
            if len(forecasts) > 0:
                forecasts.append(forecasts[-1])
                forecast_dates.append(data.index[i])
    
    print(f"\n  Generated {len(forecasts)} forecasts")
    
    # Create forecast series
    forecast_series = pd.Series(forecasts, index=forecast_dates)
    
    return forecast_series

def evaluate_forecasts(actual, forecast, model_name):
    """
    Evaluate forecast accuracy
    """
    # Align actual and forecast
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

def plot_forecast(actual, forecast, model_name, save_path='results/figures/08_arima_forecast.png'):
    """
    Plot actual vs ARIMA forecast
    """
    print("\nCreating ARIMA forecast visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot actual
    ax.plot(actual.index, actual, label='Actual Volatility', 
            color='black', linewidth=1.5, alpha=0.7)
    
    # Plot forecast
    ax.plot(forecast.index, forecast, label=f'{model_name} Forecast', 
            color='blue', linewidth=1, alpha=0.8)
    
    # Highlight COVID crisis
    ax.axvspan('2020-03-01', '2020-06-01', alpha=0.2, color='orange', label='COVID Crisis')
    
    ax.set_title(f'{model_name} Volatility Forecasts', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Annualized Volatility', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def compare_with_garch(arima_metrics):
    """
    Load GARCH results and compare with ARIMA
    """
    print("\n" + "="*60)
    print("COMPARISON: ARIMA vs GARCH MODELS")
    print("="*60)
    
    try:
        # Load GARCH results
        garch_metrics = pd.read_csv('results/tables/garch_models_performance.csv')
        
        # Combine with ARIMA
        all_metrics = pd.concat([garch_metrics, arima_metrics], ignore_index=True)
        
        # Sort by RMSE
        all_metrics_sorted = all_metrics.sort_values('RMSE')
        
        print("\nAll Traditional Models Performance (sorted by RMSE):")
        print(all_metrics_sorted.to_string(index=False))
        
        # Save combined results
        all_metrics_sorted.to_csv('results/tables/traditional_models_comparison.csv', index=False)
        print("\nSaved: results/tables/traditional_models_comparison.csv")
        
        return all_metrics_sorted
    
    except FileNotFoundError:
        print("\nGARCH results not found. Run 05_garch_models.py first.")
        return arima_metrics

def main():
    print("="*60)
    print("ARIMA MODELS - BASELINE VOLATILITY FORECASTING")
    print("="*60)
    
    # Load data
    vol_data = load_data()
    
    print("\n" + "="*60)
    print("STEP 1: FIND OPTIMAL ARIMA PARAMETERS")
    print("="*60)
    
    # Use first 80% for parameter search
    train_size = int(len(vol_data) * 0.8)
    train_data = vol_data.iloc[:train_size]
    
    # Find optimal parameters
    optimal_params = find_optimal_arima_params(train_data, max_p=5, max_q=5)
    
    # Also test some common configurations
    arima_configs = [
        optimal_params,
        (1, 0, 1),  # ARIMA(1,0,1)
        (1, 1, 1),  # ARIMA(1,1,1)
        (2, 1, 2),  # ARIMA(2,1,2)
    ]
    
    # Remove duplicates
    arima_configs = list(set(arima_configs))
    
    print(f"\nWill test these ARIMA configurations: {arima_configs}")
    
    print("\n" + "="*60)
    print("STEP 2: GENERATE ROLLING FORECASTS")
    print("="*60)
    
    # Generate forecasts for each configuration
    forecasts_dict = {}
    
    for params in arima_configs:
        model_name = f"ARIMA{params}"
        forecast = rolling_forecast_arima(vol_data, order=params, window=1200)
        forecasts_dict[model_name] = forecast
    
    print("\n" + "="*60)
    print("STEP 3: EVALUATE FORECAST ACCURACY")
    print("="*60)
    
    # Evaluate on test period
    test_start = '2022-01-03'
    vol_test = vol_data[vol_data.index >= test_start]
    
    metrics_list = []
    
    for model_name, forecast in forecasts_dict.items():
        forecast_test = forecast[forecast.index >= test_start]
        metrics = evaluate_forecasts(vol_test, forecast_test, model_name)
        metrics_list.append(metrics)
        
        print(f"\n{model_name} Performance (Test Period):")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  MAE:  {metrics['MAE']:.6f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  DA:   {metrics['DA']:.2f}%")
    
    # Create metrics DataFrame
    arima_metrics = pd.DataFrame(metrics_list)
    
    # Find best ARIMA model
    best_idx = arima_metrics['RMSE'].idxmin()
    best_model = arima_metrics.loc[best_idx, 'Model']
    best_forecast = forecasts_dict[best_model]
    
    print("\n" + "="*60)
    print("STEP 4: VISUALIZE BEST ARIMA MODEL")
    print("="*60)
    
    # Plot best model
    plot_forecast(vol_data, best_forecast, best_model)
    
    # Save ARIMA results
    arima_metrics.to_csv('results/tables/arima_models_performance.csv', index=False)
    print("\nSaved: results/tables/arima_models_performance.csv")
    
    # Save best forecast
    best_forecast.to_csv(f'results/tables/forecast_{best_model.lower().replace(",", "_").replace("(", "").replace(")", "")}.csv')
    
    # Compare with GARCH
    all_metrics = compare_with_garch(arima_metrics)
    
    print("\n" + "="*60)
    print("✅ ARIMA MODELS COMPLETE!")
    print("="*60)
    print("\nKey Findings:")
    print(f"  • Best ARIMA model: {best_model}")
    print(f"  • Best RMSE: {arima_metrics['RMSE'].min():.6f}")
    print(f"  • Best overall traditional model: {all_metrics.iloc[0]['Model']}")
    print(f"  • Best traditional RMSE: {all_metrics.iloc[0]['RMSE']:.6f}")
    print("\n📊 Traditional models baseline established!")
    print("Next: Implement ML models to beat this baseline")
    print("\nWeek 2 Progress: Traditional models DONE ✅")
    print("Coming in Week 3: LSTM models")

if __name__ == "__main__":
    main()
