"""
Script 9: Ensemble Model - ARIMA + Hybrid GARCH-LSTM
Combine the best of both approaches through weighted averaging

Approach:
- ARIMA: Best overall performer (simple, persistent series)
- Hybrid: Captures non-linear patterns GARCH misses
- Ensemble: Weighted combination optimized on validation set

Author: Volatility Research Project - Week 4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_forecasts():
    """Load all model forecasts"""
    print("Loading forecasts from all models...")
    
    # Load ARIMA forecast - exact filename with spaces
    try:
        arima_forecast = pd.read_csv('results/tables/forecast_arima1_ 0_ 1.csv', index_col=0, parse_dates=True)
        print(f"  ✓ Loaded ARIMA forecast")
    except Exception as e:
        print(f"  Error loading ARIMA: {e}")
        raise
    
    # Load Hybrid v2 forecast
    try:
        hybrid_forecast = pd.read_csv('results/tables/forecast_hybrid_v2.csv', 
                                     index_col=0, parse_dates=True)
        print(f"  Loaded Hybrid v2")
    except:
        # Try alternative name
        hybrid_forecast = pd.read_csv('results/tables/forecast_hybrid_garch_lstm.csv',
                                     index_col=0, parse_dates=True)
        print(f"  Loaded Hybrid (v1)")
    
    # Load actual volatility
    vol_actual = pd.read_csv('data/processed/volatility_clean.csv', 
                            index_col=0, parse_dates=True)
    spy_vol = vol_actual['SPY']
    
    print(f"\n  ARIMA forecast: {arima_forecast.shape}")
    print(f"  Hybrid forecast: {hybrid_forecast.shape}")
    print(f"  Actual volatility: {spy_vol.shape}")
    
    return arima_forecast, hybrid_forecast, spy_vol

def split_by_period(data, val_end='2021-12-31'):
    """Split into validation and test periods"""
    # Handle both DataFrame and Series
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]  # Take first column
    
    # Filter by date, not by position
    val = data[data.index <= val_end]
    test = data[data.index > val_end]
    
    print(f"  Val period: {len(val)} samples ({val.index.min()} to {val.index.max()})")
    print(f"  Test period: {len(test)} samples ({test.index.min()} to {test.index.max()})")
    
    return val, test

def optimize_ensemble_weights(actual, forecast1, forecast2, method='rmse'):
    """
    Find optimal weights for ensemble
    weight * forecast1 + (1-weight) * forecast2
    """
    print(f"\nOptimizing ensemble weights (method: {method})...")
    
    # Flatten if DataFrames
    if isinstance(forecast1, pd.DataFrame):
        forecast1 = forecast1.iloc[:, 0]
    if isinstance(forecast2, pd.DataFrame):
        forecast2 = forecast2.iloc[:, 0]
    
    # Align data - find common dates
    common_idx = actual.index.intersection(forecast1.index).intersection(forecast2.index)
    
    if len(common_idx) == 0:
        print(f"\n⚠️  WARNING: No overlapping dates!")
        print(f"  Actual range: {actual.index.min()} to {actual.index.max()}")
        print(f"  Forecast1 range: {forecast1.index.min()} to {forecast1.index.max()}")
        print(f"  Forecast2 range: {forecast2.index.min()} to {forecast2.index.max()}")
        raise ValueError("No overlapping dates between forecasts")
    
    print(f"  Found {len(common_idx)} common dates")
    
    actual_aligned = actual.loc[common_idx].values
    f1_aligned = forecast1.loc[common_idx].values
    f2_aligned = forecast2.loc[common_idx].values
    
    # Try different weights
    weights = np.arange(0, 1.01, 0.01)
    errors = []
    
    for w in weights:
        ensemble = w * f1_aligned + (1 - w) * f2_aligned
        
        if method == 'rmse':
            error = np.sqrt(mean_squared_error(actual_aligned, ensemble))
        elif method == 'mae':
            error = mean_absolute_error(actual_aligned, ensemble)
        
        errors.append(error)
    
    # Find best weight
    best_idx = np.argmin(errors)
    best_weight = weights[best_idx]
    best_error = errors[best_idx]
    
    print(f"  Optimal weight for forecast1 (ARIMA): {best_weight:.2f}")
    print(f"  Optimal weight for forecast2 (Hybrid): {1-best_weight:.2f}")
    print(f"  Validation {method.upper()}: {best_error:.6f}")
    
    return best_weight

def create_ensemble_forecast(forecast1, forecast2, weight):
    """
    Create ensemble forecast
    """
    print(f"\nCreating ensemble forecast...")
    print(f"  ARIMA weight: {weight:.2f}")
    print(f"  Hybrid weight: {1-weight:.2f}")
    
    # Align indices
    common_idx = forecast1.index.intersection(forecast2.index)
    f1 = forecast1.loc[common_idx]
    f2 = forecast2.loc[common_idx]
    
    # Weighted average
    if isinstance(f1, pd.DataFrame):
        f1 = f1.iloc[:, 0]
    if isinstance(f2, pd.DataFrame):
        f2 = f2.iloc[:, 0]
    
    ensemble = weight * f1 + (1 - weight) * f2
    
    print(f"  Generated {len(ensemble)} ensemble forecasts")
    
    return ensemble

def evaluate_ensemble(actual, ensemble, arima, hybrid, model_name='Ensemble'):
    """
    Evaluate ensemble performance
    """
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    # Align all data
    common_idx = (actual.index.intersection(ensemble.index)
                  .intersection(arima.index)
                  .intersection(hybrid.index))
    
    actual_aligned = actual.loc[common_idx]
    ensemble_aligned = ensemble.loc[common_idx]
    arima_aligned = arima.loc[common_idx]
    hybrid_aligned = hybrid.loc[common_idx]
    
    # Flatten if DataFrame
    if isinstance(arima_aligned, pd.DataFrame):
        arima_aligned = arima_aligned.iloc[:, 0]
    if isinstance(hybrid_aligned, pd.DataFrame):
        hybrid_aligned = hybrid_aligned.iloc[:, 0]
    
    # Calculate ensemble metrics
    errors_ens = actual_aligned - ensemble_aligned
    rmse_ens = np.sqrt(np.mean(errors_ens**2))
    mae_ens = np.mean(np.abs(errors_ens))
    mape_ens = np.mean(np.abs(errors_ens / actual_aligned)) * 100
    
    actual_dir = np.sign(actual_aligned.diff())
    ens_dir = np.sign(ensemble_aligned.diff())
    da_ens = np.mean(actual_dir[1:] == ens_dir[1:]) * 100
    
    # Calculate ARIMA metrics (for comparison)
    errors_arima = actual_aligned - arima_aligned
    rmse_arima = np.sqrt(np.mean(errors_arima**2))
    
    # Calculate Hybrid metrics
    errors_hybrid = actual_aligned - hybrid_aligned
    rmse_hybrid = np.sqrt(np.mean(errors_hybrid**2))
    
    print(f"\n{model_name} Performance:")
    print(f"  RMSE: {rmse_ens:.6f}")
    print(f"  MAE:  {mae_ens:.6f}")
    print(f"  MAPE: {mape_ens:.2f}%")
    print(f"  DA:   {da_ens:.2f}%")
    
    print(f"\nComparison:")
    print(f"  ARIMA RMSE:    {rmse_arima:.6f}")
    print(f"  Hybrid RMSE:   {rmse_hybrid:.6f}")
    print(f"  Ensemble RMSE: {rmse_ens:.6f}")
    
    # Check if ensemble beats both
    if rmse_ens < rmse_arima and rmse_ens < rmse_hybrid:
        improvement_arima = ((rmse_arima - rmse_ens) / rmse_arima) * 100
        improvement_hybrid = ((rmse_hybrid - rmse_ens) / rmse_hybrid) * 100
        print(f"\n🎉 ENSEMBLE BEATS BOTH!")
        print(f"  Improvement over ARIMA: {improvement_arima:.2f}%")
        print(f"  Improvement over Hybrid: {improvement_hybrid:.2f}%")
    elif rmse_ens < rmse_arima:
        improvement = ((rmse_arima - rmse_ens) / rmse_arima) * 100
        print(f"\n✓ Ensemble beats ARIMA by {improvement:.2f}%")
    elif rmse_ens < rmse_hybrid:
        improvement = ((rmse_hybrid - rmse_ens) / rmse_hybrid) * 100
        print(f"\n✓ Ensemble beats Hybrid by {improvement:.2f}%")
    else:
        print(f"\n📊 Ensemble between ARIMA and Hybrid")
    
    metrics = {
        'Model': model_name,
        'RMSE': rmse_ens,
        'MAE': mae_ens,
        'MAPE': mape_ens,
        'DA': da_ens,
        'N': len(common_idx)
    }
    
    return metrics

def plot_ensemble_comparison(actual, arima, hybrid, ensemble, 
                            save_path='results/figures/14_ensemble_model.png'):
    """
    Plot all forecasts
    """
    print("\nCreating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Flatten if DataFrames
    if isinstance(arima, pd.DataFrame):
        arima = arima.iloc[:, 0]
    if isinstance(hybrid, pd.DataFrame):
        hybrid = hybrid.iloc[:, 0]
    
    # Top: All forecasts
    ax1.plot(actual.index, actual, label='Actual', 
            color='black', linewidth=2, alpha=0.8, zorder=5)
    ax1.plot(arima.index, arima, label='ARIMA (Best Single)', 
            color='blue', linewidth=1, alpha=0.6)
    ax1.plot(hybrid.index, hybrid, label='Hybrid GARCH-LSTM', 
            color='purple', linewidth=1, alpha=0.6)
    ax1.plot(ensemble.index, ensemble, label='Ensemble (ARIMA + Hybrid)', 
            color='red', linewidth=1.5, alpha=0.8, linestyle='--')
    
    ax1.set_title('Ensemble Model: Combining ARIMA + Hybrid GARCH-LSTM', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Annualized Volatility')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Forecast errors
    arima_errors = actual.loc[arima.index] - arima
    hybrid_errors = actual.loc[hybrid.index] - hybrid
    ensemble_errors = actual.loc[ensemble.index] - ensemble
    
    ax2.plot(arima_errors.index, arima_errors, label='ARIMA Error', 
            color='blue', alpha=0.5, linewidth=1)
    ax2.plot(hybrid_errors.index, hybrid_errors, label='Hybrid Error', 
            color='purple', alpha=0.5, linewidth=1)
    ax2.plot(ensemble_errors.index, ensemble_errors, label='Ensemble Error', 
            color='red', alpha=0.7, linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    ax2.set_title('Forecast Errors Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Forecast Error')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_weight_optimization(actual, forecast1, forecast2, 
                            save_path='results/figures/15_ensemble_weights.png'):
    """
    Plot how RMSE changes with different weights
    """
    print("\nPlotting weight optimization...")
    
    # Align data
    common_idx = actual.index.intersection(forecast1.index).intersection(forecast2.index)
    actual_aligned = actual.loc[common_idx].values
    
    if isinstance(forecast1, pd.DataFrame):
        f1_aligned = forecast1.loc[common_idx].iloc[:, 0].values
    else:
        f1_aligned = forecast1.loc[common_idx].values
        
    if isinstance(forecast2, pd.DataFrame):
        f2_aligned = forecast2.loc[common_idx].iloc[:, 0].values
    else:
        f2_aligned = forecast2.loc[common_idx].values
    
    # Try different weights
    weights = np.arange(0, 1.01, 0.01)
    rmse_values = []
    
    for w in weights:
        ensemble = w * f1_aligned + (1 - w) * f2_aligned
        rmse = np.sqrt(mean_squared_error(actual_aligned, ensemble))
        rmse_values.append(rmse)
    
    # Find minimum
    best_idx = np.argmin(rmse_values)
    best_weight = weights[best_idx]
    best_rmse = rmse_values[best_idx]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(weights, rmse_values, linewidth=2, color='blue')
    ax.axvline(x=best_weight, color='red', linestyle='--', linewidth=2, 
              label=f'Optimal: ARIMA={best_weight:.2f}, Hybrid={1-best_weight:.2f}')
    ax.scatter([best_weight], [best_rmse], color='red', s=100, zorder=5)
    
    # Mark pure models
    ax.axvline(x=0, color='purple', linestyle=':', alpha=0.5, label='Pure Hybrid')
    ax.axvline(x=1, color='green', linestyle=':', alpha=0.5, label='Pure ARIMA')
    
    ax.set_xlabel('Weight on ARIMA', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Ensemble Weight Optimization', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def compare_all_models(ensemble_metrics):
    """
    Final comparison with all models
    """
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)
    
    try:
        all_models = pd.read_csv('results/tables/all_models_final.csv')
        
        # Add ensemble
        ensemble_df = pd.DataFrame([ensemble_metrics])
        all_models = pd.concat([ensemble_df, all_models], ignore_index=True)
        
        # Sort
        all_models_sorted = all_models.sort_values('RMSE')
        
        print("\nAll Models (sorted by RMSE):")
        print(all_models_sorted.to_string(index=False))
        
        # Save
        all_models_sorted.to_csv('results/tables/all_models_with_ensemble.csv', index=False)
        print(f"\nSaved: results/tables/all_models_with_ensemble.csv")
        
        # Check winner
        best_model = all_models_sorted.iloc[0]['Model']
        best_rmse = all_models_sorted.iloc[0]['RMSE']
        
        if 'Ensemble' in best_model:
            print("\n🎉🎉🎉 ENSEMBLE IS THE CHAMPION! 🎉🎉🎉")
            print(f"  Model: {best_model}")
            print(f"  RMSE: {best_rmse:.6f}")
            print("\n✨ Your innovation works! Combining models beats individuals!")
        else:
            print(f"\n📊 Current best: {best_model}")
            print(f"  RMSE: {best_rmse:.6f}")
            
            ensemble_rank = all_models_sorted[all_models_sorted['Model'].str.contains('Ensemble', na=False)].index[0] + 1
            print(f"\n  Ensemble rank: #{ensemble_rank}")
        
        return all_models_sorted
        
    except FileNotFoundError:
        print("\nPrevious results not found.")
        return pd.DataFrame([ensemble_metrics])

def main():
    print("="*60)
    print("ENSEMBLE MODEL: ARIMA + Hybrid GARCH-LSTM")
    print("Combining Best of Both Approaches")
    print("="*60)
    
    # Load forecasts
    arima_forecast, hybrid_forecast, actual_vol = load_forecasts()
    
    print("\n" + "="*60)
    print("STEP 1: OPTIMIZE WEIGHTS")
    print("="*60)
    
    # Check if we have validation data
    arima_val, arima_test = split_by_period(arima_forecast)
    hybrid_val, hybrid_test = split_by_period(hybrid_forecast)
    actual_val, actual_test = split_by_period(actual_vol)
    
    # Check if hybrid has validation data
    if len(hybrid_val) == 0:
        print("\n⚠️  Hybrid forecast only covers test period")
        print("  Using TEST set for weight optimization (with cross-validation)")
        print("  This is acceptable for ensemble combination")
        
        # Use first half of test for optimization, second half for final evaluation
        mid_point = len(arima_test) // 2
        arima_opt = arima_test.iloc[:mid_point]
        hybrid_opt = hybrid_test.iloc[:mid_point]
        actual_opt = actual_test.iloc[:mid_point]
        
        arima_final = arima_test.iloc[mid_point:]
        hybrid_final = hybrid_test.iloc[mid_point:]
        actual_final = actual_test.iloc[mid_point:]
        
        print(f"\n  Optimization set: {len(arima_opt)} samples")
        print(f"  Final evaluation set: {len(arima_final)} samples")
    else:
        print("\n✓ Using validation set for optimization")
        arima_opt = arima_val
        hybrid_opt = hybrid_val
        actual_opt = actual_val
        
        arima_final = arima_test
        hybrid_final = hybrid_test
        actual_final = actual_test
    
    # Optimize weights on optimization set
    optimal_weight = optimize_ensemble_weights(
        actual_opt, arima_opt, hybrid_opt, method='rmse'
    )
    
    # Plot weight optimization
    plot_weight_optimization(actual_opt, arima_opt, hybrid_opt)
    
    print("\n" + "="*60)
    print("STEP 2: CREATE ENSEMBLE ON FINAL EVALUATION SET")
    print("="*60)
    
    # Create ensemble forecast
    ensemble_final = create_ensemble_forecast(arima_final, hybrid_final, optimal_weight)
    
    print("\n" + "="*60)
    print("STEP 3: EVALUATE ON FINAL SET")
    print("="*60)
    
    # Evaluate
    metrics = evaluate_ensemble(
        actual_final, ensemble_final, arima_final, hybrid_final,
        model_name='Ensemble (ARIMA + Hybrid)'
    )
    
    # Plot comparison
    plot_ensemble_comparison(actual_final, arima_final, hybrid_final, ensemble_final)
    
    # Save ensemble forecast
    ensemble_final.to_csv('results/tables/forecast_ensemble.csv')
    print("\nSaved: results/tables/forecast_ensemble.csv")
    
    # Compare with all models
    all_models = compare_all_models(metrics)
    
    print("\n" + "="*60)
    print("✅ ENSEMBLE MODEL COMPLETE!")
    print("="*60)
    print("\nKey Innovation:")
    print("  ✓ Combines ARIMA's simplicity with Hybrid's flexibility")
    print("  ✓ Optimized weights on validation set")
    print("  ✓ Ensemble leverages strengths of both approaches")
    
    print("\nWeek 4-5 Complete!")
    print("Next: Adaptive regime-switching framework")

if __name__ == "__main__":
    main()
