"""
Script 12: Asymmetric Loss Evaluation
Risk-Aware Performance Assessment

Standard RMSE treats all errors equally.
In risk management, UNDER-predicting volatility is FAR WORSE than over-predicting.

We evaluate models using asymmetric loss:
  Loss = α * (actual - predicted)² if actual > predicted  (under-prediction)
       + β * (predicted - actual)² if predicted > actual  (over-prediction)

Where α > β (typically α = 3, β = 1)

This transforms a "forecasting paper" into a "risk management paper"!

Author: Volatility Research Project - Risk-Aware Evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_all_forecasts():
    """Load all model forecasts"""
    print("Loading all forecasts...")
    
    forecasts = {}
    
    # ARIMA
    forecasts['ARIMA'] = pd.read_csv('results/tables/forecast_arima1_ 0_ 1.csv', 
                                     index_col=0, parse_dates=True).iloc[:, 0]
    
    # LSTM Extended
    forecasts['LSTM'] = pd.read_csv('results/tables/forecast_lstm_extended.csv', 
                                    index_col=0, parse_dates=True).iloc[:, 0]
    
    # Adaptive
    forecasts['Adaptive'] = pd.read_csv('results/tables/forecast_regime_adaptive_extended.csv', 
                                        index_col=0, parse_dates=True).iloc[:, 0]
    
    # Actual volatility
    vol_actual = pd.read_csv('data/processed/volatility_clean.csv', 
                             index_col=0, parse_dates=True)['SPY']
    
    # VIX for regime
    prices = pd.read_csv('data/processed/prices_clean.csv', 
                        index_col=0, parse_dates=True)
    vix = prices['^VIX']
    
    print(f"  Loaded {len(forecasts)} models")
    
    return forecasts, vol_actual, vix

def classify_regime(vix):
    """Classify market regime"""
    regime = pd.Series(index=vix.index, dtype=str)
    regime[vix < 20] = 'Normal'
    regime[(vix >= 20) & (vix < 30)] = 'Stress'
    regime[vix >= 30] = 'Crisis'
    return regime

def compute_asymmetric_loss(actual, predicted, alpha=3.0, beta=1.0):
    """
    Compute asymmetric loss function
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    alpha : float
        Penalty for under-prediction (actual > predicted)
    beta : float
        Penalty for over-prediction (predicted > actual)
    
    Returns:
    --------
    asymmetric_loss : float
        Mean asymmetric loss
    """
    errors = actual - predicted
    
    # Under-prediction (actual > predicted, error > 0)
    under_pred_mask = errors > 0
    under_pred_loss = alpha * (errors[under_pred_mask] ** 2)
    
    # Over-prediction (predicted > actual, error < 0)
    over_pred_mask = errors <= 0
    over_pred_loss = beta * (errors[over_pred_mask] ** 2)
    
    # Total loss
    total_loss = np.concatenate([under_pred_loss, over_pred_loss])
    asymmetric_loss = np.sqrt(np.mean(total_loss))  # Root mean asymmetric loss
    
    return asymmetric_loss

def evaluate_model_asymmetric(actual, forecast, model_name, alpha=3.0, beta=1.0):
    """
    Evaluate model using both symmetric and asymmetric loss
    """
    # Align data
    common_idx = actual.index.intersection(forecast.index)
    actual_aligned = actual.loc[common_idx]
    forecast_aligned = forecast.loc[common_idx]
    
    errors = actual_aligned - forecast_aligned
    
    # Symmetric RMSE (baseline)
    rmse = np.sqrt(np.mean(errors ** 2))
    
    # Asymmetric loss
    asym_loss = compute_asymmetric_loss(actual_aligned.values, 
                                        forecast_aligned.values, 
                                        alpha, beta)
    
    # Under-prediction statistics
    under_pred_mask = errors > 0
    under_pred_count = under_pred_mask.sum()
    under_pred_pct = (under_pred_count / len(errors)) * 100
    under_pred_mean_error = errors[under_pred_mask].mean() if under_pred_count > 0 else 0
    
    # Over-prediction statistics
    over_pred_mask = errors <= 0
    over_pred_count = over_pred_mask.sum()
    over_pred_pct = (over_pred_count / len(errors)) * 100
    over_pred_mean_error = abs(errors[over_pred_mask].mean()) if over_pred_count > 0 else 0
    
    results = {
        'Model': model_name,
        'RMSE': rmse,
        'Asymmetric_Loss': asym_loss,
        'Under_Pred_%': under_pred_pct,
        'Under_Pred_Mean': under_pred_mean_error,
        'Over_Pred_%': over_pred_pct,
        'Over_Pred_Mean': over_pred_mean_error,
        'N': len(common_idx)
    }
    
    return results

def evaluate_by_regime_asymmetric(actual, forecasts_dict, regime, alpha=3.0, beta=1.0):
    """
    Evaluate all models by regime using asymmetric loss
    """
    print("\n" + "="*80)
    print(f"ASYMMETRIC LOSS EVALUATION (α={alpha}, β={beta})")
    print("Under-prediction penalty = {:.1f}x over-prediction".format(alpha/beta))
    print("="*80)
    
    results = []
    
    regimes = ['Normal', 'Stress', 'Crisis', 'Overall']
    
    for reg in regimes:
        print(f"\n{reg} Regime:")
        print("-" * 80)
        
        for model_name, forecast in forecasts_dict.items():
            # Align data
            common_idx = actual.index.intersection(forecast.index).intersection(regime.index)
            
            if len(common_idx) == 0:
                continue
            
            # Filter by regime
            if reg == 'Overall':
                mask = pd.Series(True, index=common_idx)
            else:
                mask = regime.loc[common_idx] == reg
            
            if mask.sum() == 0:
                continue
            
            actual_reg = actual.loc[common_idx][mask]
            forecast_reg = forecast.loc[common_idx][mask]
            
            # Evaluate
            result = evaluate_model_asymmetric(actual_reg, forecast_reg, 
                                              model_name, alpha, beta)
            result['Regime'] = reg
            
            results.append(result)
            
            print(f"  {model_name:15s} | RMSE: {result['RMSE']:.6f} | "
                  f"Asym Loss: {result['Asymmetric_Loss']:.6f} | "
                  f"Under-pred: {result['Under_Pred_%']:5.1f}% | "
                  f"N: {result['N']}")
    
    results_df = pd.DataFrame(results)
    return results_df

def compare_rankings(results_df):
    """
    Compare model rankings under RMSE vs Asymmetric Loss
    """
    print("\n" + "="*80)
    print("RANKING COMPARISON: RMSE vs ASYMMETRIC LOSS")
    print("="*80)
    
    for regime in ['Normal', 'Stress', 'Crisis', 'Overall']:
        regime_data = results_df[results_df['Regime'] == regime].copy()
        
        if len(regime_data) == 0:
            continue
        
        # Rank by RMSE
        regime_data['RMSE_Rank'] = regime_data['RMSE'].rank()
        
        # Rank by Asymmetric Loss
        regime_data['Asym_Rank'] = regime_data['Asymmetric_Loss'].rank()
        
        # Rank change
        regime_data['Rank_Change'] = regime_data['RMSE_Rank'] - regime_data['Asym_Rank']
        
        print(f"\n{regime} Regime:")
        print("-" * 80)
        
        # Sort by asymmetric loss
        regime_sorted = regime_data.sort_values('Asymmetric_Loss')
        
        for _, row in regime_sorted.iterrows():
            rank_change = row['Rank_Change']
            change_str = ""
            if rank_change > 0:
                change_str = f"↑ improved {int(rank_change)} places"
            elif rank_change < 0:
                change_str = f"↓ dropped {int(abs(rank_change))} places"
            else:
                change_str = "→ no change"
            
            print(f"  {int(row['Asym_Rank'])}. {row['Model']:15s} "
                  f"(RMSE rank: {int(row['RMSE_Rank'])}) {change_str}")
    
    return results_df

def plot_asymmetric_comparison(results_df, save_path='results/figures/21_asymmetric_loss.png'):
    """
    Visualize RMSE vs Asymmetric Loss rankings
    """
    print("\nCreating asymmetric loss visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    regimes = ['Normal', 'Stress', 'Crisis', 'Overall']
    
    for idx, regime in enumerate(regimes):
        ax = axes[idx // 2, idx % 2]
        
        regime_data = results_df[results_df['Regime'] == regime].copy()
        
        if len(regime_data) == 0:
            ax.text(0.5, 0.5, f'No data for {regime}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{regime} Regime')
            continue
        
        # Sort by asymmetric loss
        regime_data = regime_data.sort_values('Asymmetric_Loss')
        
        x = np.arange(len(regime_data))
        width = 0.35
        
        # Plot both metrics
        bars1 = ax.bar(x - width/2, regime_data['RMSE'], width, 
                      label='RMSE', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, regime_data['Asymmetric_Loss'], width, 
                      label='Asymmetric Loss (α=3, β=1)', alpha=0.8, color='coral')
        
        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title(f'{regime} Regime', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(regime_data['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_under_over_prediction(results_df, save_path='results/figures/22_prediction_bias.png'):
    """
    Plot under-prediction vs over-prediction rates
    """
    print("\nCreating prediction bias visualization...")
    
    # Filter overall regime
    overall_data = results_df[results_df['Regime'] == 'Overall'].copy()
    
    if len(overall_data) == 0:
        print("  No overall data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    models = overall_data['Model']
    x = np.arange(len(models))
    
    # Left: Under vs Over prediction rates
    ax1.barh(x, overall_data['Under_Pred_%'], alpha=0.8, color='red', label='Under-prediction')
    ax1.barh(x, -overall_data['Over_Pred_%'], alpha=0.8, color='green', label='Over-prediction')
    ax1.axvline(x=0, color='black', linewidth=1)
    ax1.set_yticks(x)
    ax1.set_yticklabels(models)
    ax1.set_xlabel('Percentage (%)')
    ax1.set_title('Under-prediction vs Over-prediction Rates', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Right: Mean error magnitude
    ax2.barh(x, overall_data['Under_Pred_Mean'], alpha=0.8, color='red', label='Under-pred mean error')
    ax2.barh(x, -overall_data['Over_Pred_Mean'], alpha=0.8, color='green', label='Over-pred mean error')
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.set_yticks(x)
    ax2.set_yticklabels(models)
    ax2.set_xlabel('Mean Error Magnitude')
    ax2.set_title('Average Error When Wrong', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def sensitivity_analysis(actual, forecasts_dict, regime):
    """
    Test different alpha/beta ratios
    """
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: Different α/β Ratios")
    print("="*80)
    
    # Test different penalty ratios
    ratios = [(1, 1), (2, 1), (3, 1), (5, 1), (10, 1)]
    
    sensitivity_results = []
    
    for alpha, beta in ratios:
        print(f"\n--- α={alpha}, β={beta} (under-pred penalty = {alpha/beta}x) ---")
        
        # Evaluate overall regime only
        for model_name, forecast in forecasts_dict.items():
            common_idx = actual.index.intersection(forecast.index).intersection(regime.index)
            
            if len(common_idx) == 0:
                continue
            
            actual_aligned = actual.loc[common_idx]
            forecast_aligned = forecast.loc[common_idx]
            
            asym_loss = compute_asymmetric_loss(actual_aligned.values, 
                                               forecast_aligned.values, 
                                               alpha, beta)
            
            sensitivity_results.append({
                'Alpha': alpha,
                'Beta': beta,
                'Ratio': f'{alpha}:{beta}',
                'Model': model_name,
                'Asymmetric_Loss': asym_loss
            })
    
    sensitivity_df = pd.DataFrame(sensitivity_results)
    
    # Show rankings for each ratio
    for ratio_name in sensitivity_df['Ratio'].unique():
        ratio_data = sensitivity_df[sensitivity_df['Ratio'] == ratio_name].sort_values('Asymmetric_Loss')
        print(f"\n{ratio_name} Ranking:")
        for i, row in enumerate(ratio_data.itertuples(), 1):
            print(f"  {i}. {row.Model:15s} - {row.Asymmetric_Loss:.6f}")
    
    return sensitivity_df

def main():
    print("="*80)
    print("ASYMMETRIC LOSS EVALUATION")
    print("Risk-Aware Model Performance Assessment")
    print("="*80)
    
    # Load data
    forecasts, vol_actual, vix = load_all_forecasts()
    
    # Classify regimes
    regime = classify_regime(vix)
    
    print("\n" + "="*80)
    print("STEP 1: EVALUATE WITH ASYMMETRIC LOSS (α=3, β=1)")
    print("="*80)
    
    # Evaluate all models
    results_df = evaluate_by_regime_asymmetric(vol_actual, forecasts, regime, 
                                               alpha=3.0, beta=1.0)
    
    # Save results
    results_df.to_csv('results/tables/asymmetric_loss_results.csv', index=False)
    print(f"\nSaved: results/tables/asymmetric_loss_results.csv")
    
    print("\n" + "="*80)
    print("STEP 2: COMPARE RANKINGS")
    print("="*80)
    
    # Compare rankings
    results_with_ranks = compare_rankings(results_df)
    
    print("\n" + "="*80)
    print("STEP 3: VISUALIZATIONS")
    print("="*80)
    
    # Create plots
    plot_asymmetric_comparison(results_df)
    plot_under_over_prediction(results_df)
    
    print("\n" + "="*80)
    print("STEP 4: SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Test different alpha/beta ratios
    sensitivity_df = sensitivity_analysis(vol_actual, forecasts, regime)
    sensitivity_df.to_csv('results/tables/asymmetric_sensitivity.csv', index=False)
    print(f"\nSaved: results/tables/asymmetric_sensitivity.csv")
    
    print("\n" + "="*80)
    print("✅ ASYMMETRIC LOSS EVALUATION COMPLETE!")
    print("="*80)
    
    # Highlight key insights
    print("\n🎯 KEY INSIGHTS:")
    
    # Overall winners
    overall_results = results_df[results_df['Regime'] == 'Overall'].copy()
    
    if len(overall_results) > 0:
        # Best by RMSE
        best_rmse = overall_results.loc[overall_results['RMSE'].idxmin()]
        print(f"\n  RMSE Winner: {best_rmse['Model']}")
        print(f"    RMSE: {best_rmse['RMSE']:.6f}")
        
        # Best by Asymmetric Loss
        best_asym = overall_results.loc[overall_results['Asymmetric_Loss'].idxmin()]
        print(f"\n  Asymmetric Loss Winner: {best_asym['Model']}")
        print(f"    Asymmetric Loss: {best_asym['Asymmetric_Loss']:.6f}")
        print(f"    Under-prediction rate: {best_asym['Under_Pred_%']:.1f}%")
        
        if best_rmse['Model'] != best_asym['Model']:
            print(f"\n  🔥 WINNER CHANGED! {best_asym['Model']} wins under risk-aware metric!")
        else:
            print(f"\n  → Same winner, but gap may have changed")
    
    # Crisis performance
    crisis_results = results_df[results_df['Regime'] == 'Crisis'].copy()
    
    if len(crisis_results) > 0:
        best_crisis = crisis_results.loc[crisis_results['Asymmetric_Loss'].idxmin()]
        print(f"\n  Crisis Period (Asymmetric Loss): {best_crisis['Model']}")
        print(f"    Loss: {best_crisis['Asymmetric_Loss']:.6f}")
        print(f"    This matters most for risk management!")
    
    print("\n📝 PAPER IMPACT:")
    print("  ✓ Converts forecasting → risk management")
    print("  ✓ Shows under-prediction is costly")
    print("  ✓ May change model rankings")
    print("  ✓ Practical value for practitioners")
    print("  ✓ Differentiates your work!")

if __name__ == "__main__":
    main()
