"""
Script 13: Regime Transition Analysis
Evaluate model performance during regime changes

Most papers evaluate static regimes.
We analyze TRANSITIONS between regimes:
- What happens in the 5 days before a regime change?
- What happens in the 5 days after?
- Which model adapts fastest?

This is NOVEL - very few papers look at transitions!

Key insights:
- Models that work in stable regimes may fail during transitions
- Adaptive framework should adjust weights during transitions
- Shows "temporal intelligence"

Author: Volatility Research Project - Temporal Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

def load_all_data():
    """Load forecasts, actuals, and regime data"""
    print("Loading data...")
    
    # Forecasts
    arima = pd.read_csv('results/tables/forecast_arima1_ 0_ 1.csv', 
                       index_col=0, parse_dates=True).iloc[:, 0]
    lstm = pd.read_csv('results/tables/forecast_lstm_extended.csv', 
                      index_col=0, parse_dates=True).iloc[:, 0]
    adaptive = pd.read_csv('results/tables/forecast_regime_adaptive_extended.csv', 
                          index_col=0, parse_dates=True).iloc[:, 0]
    
    # Actual volatility
    vol_actual = pd.read_csv('data/processed/volatility_clean.csv', 
                            index_col=0, parse_dates=True)['SPY']
    
    # VIX for regime
    prices = pd.read_csv('data/processed/prices_clean.csv', 
                        index_col=0, parse_dates=True)
    vix = prices['^VIX']
    
    print(f"  ARIMA: {len(arima)} forecasts")
    print(f"  LSTM: {len(lstm)} forecasts")
    print(f"  Adaptive: {len(adaptive)} forecasts")
    
    return arima, lstm, adaptive, vol_actual, vix

def classify_regime(vix):
    """Classify market regime"""
    regime = pd.Series(index=vix.index, dtype=str)
    regime[vix < 20] = 'Normal'
    regime[(vix >= 20) & (vix < 30)] = 'Stress'
    regime[vix >= 30] = 'Crisis'
    return regime

def detect_regime_transitions(regime):
    """
    Detect when regime changes occur
    
    Returns DataFrame with transition information
    """
    print("\nDetecting regime transitions...")
    
    # Find where regime changes
    regime_shift = regime != regime.shift(1)
    transition_dates = regime.index[regime_shift]
    
    # Skip first date (no previous regime)
    transition_dates = transition_dates[1:]
    
    transitions = []
    
    for date in transition_dates:
        idx = regime.index.get_loc(date)
        
        if idx > 0:
            from_regime = regime.iloc[idx - 1]
            to_regime = regime.iloc[idx]
            
            transitions.append({
                'date': date,
                'from': from_regime,
                'to': to_regime,
                'type': f"{from_regime}→{to_regime}"
            })
    
    transitions_df = pd.DataFrame(transitions)
    
    print(f"  Found {len(transitions_df)} regime transitions")
    print(f"\nTransition types:")
    print(transitions_df['type'].value_counts())
    
    return transitions_df

def get_window_around_transition(date, data, window=5):
    """
    Get data in window around transition date
    
    Parameters:
    -----------
    date : datetime
        Transition date
    data : Series
        Data to extract
    window : int
        Number of days before/after
    
    Returns:
    --------
    before : Series
        Data from t-window to t-1
    after : Series
        Data from t to t+window-1
    """
    # Get index position
    if date not in data.index:
        return None, None
    
    idx = data.index.get_loc(date)
    
    # Before transition (t-window to t-1)
    start_before = max(0, idx - window)
    before = data.iloc[start_before:idx]
    
    # After transition (t to t+window-1)
    end_after = min(len(data), idx + window)
    after = data.iloc[idx:end_after]
    
    return before, after

def evaluate_transition_window(actual, forecast, window_type='before'):
    """
    Evaluate forecast performance in a window
    """
    if len(actual) == 0 or len(forecast) == 0:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'n': 0
        }
    
    # Align
    common_idx = actual.index.intersection(forecast.index)
    
    if len(common_idx) == 0:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'n': 0
        }
    
    actual_aligned = actual.loc[common_idx]
    forecast_aligned = forecast.loc[common_idx]
    
    errors = actual_aligned - forecast_aligned
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    
    return {
        'rmse': rmse,
        'mae': mae,
        'n': len(common_idx)
    }

def analyze_all_transitions(transitions_df, actual, forecasts_dict, window=5):
    """
    Analyze model performance around all transitions
    """
    print(f"\n" + "="*80)
    print(f"ANALYZING TRANSITIONS (±{window} days window)")
    print("="*80)
    
    results = []
    
    for _, trans in transitions_df.iterrows():
        date = trans['date']
        trans_type = trans['type']
        
        # Get actual volatility windows
        actual_before, actual_after = get_window_around_transition(date, actual, window)
        
        if actual_before is None or actual_after is None:
            continue
        
        # Evaluate each model
        for model_name, forecast in forecasts_dict.items():
            # Get forecast windows
            forecast_before, forecast_after = get_window_around_transition(date, forecast, window)
            
            if forecast_before is None or forecast_after is None:
                continue
            
            # Evaluate before
            perf_before = evaluate_transition_window(actual_before, forecast_before, 'before')
            
            # Evaluate after
            perf_after = evaluate_transition_window(actual_after, forecast_after, 'after')
            
            # Store results
            results.append({
                'date': date,
                'transition': trans_type,
                'from_regime': trans['from'],
                'to_regime': trans['to'],
                'model': model_name,
                'period': 'before',
                **perf_before
            })
            
            results.append({
                'date': date,
                'transition': trans_type,
                'from_regime': trans['from'],
                'to_regime': trans['to'],
                'model': model_name,
                'period': 'after',
                **perf_after
            })
    
    results_df = pd.DataFrame(results)
    
    print(f"\nAnalyzed {len(transitions_df)} transitions")
    print(f"Generated {len(results_df)} performance records")
    
    return results_df

def summarize_by_transition_type(results_df):
    """
    Aggregate performance by transition type and model
    """
    print("\n" + "="*80)
    print("PERFORMANCE BY TRANSITION TYPE")
    print("="*80)
    
    # Group by transition type and model
    transition_types = results_df['transition'].unique()
    
    summary = []
    
    for trans_type in transition_types:
        trans_data = results_df[results_df['transition'] == trans_type]
        
        print(f"\n{trans_type}:")
        print("-" * 80)
        
        for model in trans_data['model'].unique():
            model_data = trans_data[trans_data['model'] == model]
            
            # Before transition
            before_data = model_data[model_data['period'] == 'before']
            rmse_before = before_data['rmse'].mean()
            n_before = before_data['n'].sum()
            
            # After transition
            after_data = model_data[model_data['period'] == 'after']
            rmse_after = after_data['rmse'].mean()
            n_after = after_data['n'].sum()
            
            # Change
            rmse_change = rmse_after - rmse_before
            pct_change = (rmse_change / rmse_before * 100) if rmse_before > 0 else 0
            
            print(f"  {model:15s} | Before: {rmse_before:.6f} | "
                  f"After: {rmse_after:.6f} | "
                  f"Change: {rmse_change:+.6f} ({pct_change:+.1f}%)")
            
            summary.append({
                'transition': trans_type,
                'model': model,
                'rmse_before': rmse_before,
                'rmse_after': rmse_after,
                'rmse_change': rmse_change,
                'pct_change': pct_change,
                'n_before': n_before,
                'n_after': n_after
            })
    
    summary_df = pd.DataFrame(summary)
    return summary_df

def analyze_adaptation_speed(results_df):
    """
    Which model adapts fastest after regime change?
    """
    print("\n" + "="*80)
    print("ADAPTATION SPEED ANALYSIS")
    print("="*80)
    
    # Focus on "after" period
    after_data = results_df[results_df['period'] == 'after'].copy()
    
    # Calculate average RMSE increase after transition
    adaptation_scores = []
    
    for model in after_data['model'].unique():
        model_data = after_data[after_data['model'] == model]
        
        avg_rmse = model_data['rmse'].mean()
        
        # Compare to "before" performance
        before_data = results_df[
            (results_df['model'] == model) & 
            (results_df['period'] == 'before')
        ]
        avg_rmse_before = before_data['rmse'].mean()
        
        degradation = avg_rmse - avg_rmse_before
        
        adaptation_scores.append({
            'model': model,
            'avg_rmse_after': avg_rmse,
            'avg_rmse_before': avg_rmse_before,
            'degradation': degradation
        })
    
    adaptation_df = pd.DataFrame(adaptation_scores).sort_values('degradation')
    
    print("\nModel Adaptation Ranking (lower degradation = better):")
    print("-" * 80)
    for i, row in enumerate(adaptation_df.itertuples(), 1):
        print(f"  {i}. {row.model:15s} - Degradation: {row.degradation:+.6f}")
    
    return adaptation_df

def plot_transition_performance(summary_df, 
                               save_path='results/figures/23_transition_performance.png'):
    """
    Visualize performance before/after transitions
    """
    print("\nCreating transition performance visualization...")
    
    # Get unique transition types
    transition_types = summary_df['transition'].unique()
    
    # Create subplots
    n_trans = len(transition_types)
    fig, axes = plt.subplots(n_trans, 1, figsize=(12, 4*n_trans))
    
    if n_trans == 1:
        axes = [axes]
    
    for idx, trans_type in enumerate(transition_types):
        ax = axes[idx]
        
        trans_data = summary_df[summary_df['transition'] == trans_type]
        
        models = trans_data['model'].values
        x = np.arange(len(models))
        width = 0.35
        
        # Plot before/after
        ax.bar(x - width/2, trans_data['rmse_before'], width, 
              label='Before Transition', alpha=0.8, color='steelblue')
        ax.bar(x + width/2, trans_data['rmse_after'], width, 
              label='After Transition', alpha=0.8, color='coral')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('RMSE')
        ax.set_title(f'Performance Around {trans_type} Transitions', 
                    fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_adaptation_speed(adaptation_df, 
                         save_path='results/figures/24_adaptation_speed.png'):
    """
    Visualize which models adapt fastest
    """
    print("\nCreating adaptation speed visualization...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = adaptation_df['model'].values
    degradation = adaptation_df['degradation'].values
    
    colors = ['green' if d < 0 else 'red' for d in degradation]
    
    bars = ax.barh(models, degradation, color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=1)
    
    ax.set_xlabel('Performance Degradation After Transition (RMSE)', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title('Model Adaptation to Regime Changes\n(Lower = Better)', 
                fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (model, deg) in enumerate(zip(models, degradation)):
        ax.text(deg + 0.0005 * np.sign(deg), i, f'{deg:+.4f}', 
               va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def analyze_specific_transitions(results_df, actual, vix):
    """
    Deep dive into specific important transitions (e.g., COVID onset)
    """
    print("\n" + "="*80)
    print("ANALYSIS OF SPECIFIC CRITICAL TRANSITIONS")
    print("="*80)
    
    # Find COVID crisis onset (March 2020)
    covid_dates = results_df[
        (results_df['date'] >= '2020-03-01') & 
        (results_df['date'] <= '2020-03-31') &
        (results_df['to_regime'] == 'Crisis')
    ]['date'].unique()
    
    if len(covid_dates) > 0:
        print(f"\nCOVID Crisis Onset (March 2020):")
        print("-" * 80)
        
        for date in covid_dates[:3]:  # Show first 3 transitions
            print(f"\nTransition on {date.strftime('%Y-%m-%d')}:")
            
            trans_data = results_df[results_df['date'] == date]
            
            for period in ['before', 'after']:
                period_data = trans_data[trans_data['period'] == period]
                print(f"\n  {period.capitalize()} transition:")
                
                for _, row in period_data.iterrows():
                    print(f"    {row['model']:15s}: RMSE = {row['rmse']:.6f}")
    
    # Find regime normalizations (Crisis → Normal)
    normalizations = results_df[
        (results_df['from_regime'] == 'Crisis') & 
        (results_df['to_regime'] == 'Normal')
    ]['date'].unique()
    
    if len(normalizations) > 0:
        print(f"\n\nCrisis → Normal Transitions:")
        print("-" * 80)
        print(f"Found {len(normalizations)} transitions")
        
        # Show summary
        norm_data = results_df[
            (results_df['from_regime'] == 'Crisis') & 
            (results_df['to_regime'] == 'Normal')
        ]
        
        for model in norm_data['model'].unique():
            model_data = norm_data[norm_data['model'] == model]
            
            before = model_data[model_data['period'] == 'before']['rmse'].mean()
            after = model_data[model_data['period'] == 'after']['rmse'].mean()
            
            print(f"  {model:15s}: {before:.6f} → {after:.6f} "
                  f"(change: {after-before:+.6f})")

def main():
    print("="*80)
    print("REGIME TRANSITION ANALYSIS")
    print("Temporal Intelligence in Model Performance")
    print("="*80)
    
    # Load data
    arima, lstm, adaptive, vol_actual, vix = load_all_data()
    
    # Classify regimes
    regime = classify_regime(vix)
    
    print("\n" + "="*80)
    print("STEP 1: DETECT REGIME TRANSITIONS")
    print("="*80)
    
    # Detect transitions
    transitions_df = detect_regime_transitions(regime)
    
    # Save transition dates
    transitions_df.to_csv('results/tables/regime_transitions.csv', index=False)
    print(f"\nSaved: results/tables/regime_transitions.csv")
    
    print("\n" + "="*80)
    print("STEP 2: ANALYZE PERFORMANCE AROUND TRANSITIONS")
    print("="*80)
    
    # Prepare forecasts
    forecasts = {
        'ARIMA': arima,
        'LSTM': lstm,
        'Adaptive': adaptive
    }
    
    # Analyze transitions
    results_df = analyze_all_transitions(transitions_df, vol_actual, forecasts, window=5)
    
    # Save detailed results
    results_df.to_csv('results/tables/transition_analysis_detailed.csv', index=False)
    print(f"\nSaved: results/tables/transition_analysis_detailed.csv")
    
    print("\n" + "="*80)
    print("STEP 3: SUMMARIZE BY TRANSITION TYPE")
    print("="*80)
    
    # Summarize
    summary_df = summarize_by_transition_type(results_df)
    
    # Save summary
    summary_df.to_csv('results/tables/transition_analysis_summary.csv', index=False)
    print(f"\nSaved: results/tables/transition_analysis_summary.csv")
    
    print("\n" + "="*80)
    print("STEP 4: ADAPTATION SPEED ANALYSIS")
    print("="*80)
    
    # Analyze adaptation
    adaptation_df = analyze_adaptation_speed(results_df)
    
    # Save
    adaptation_df.to_csv('results/tables/adaptation_speed.csv', index=False)
    print(f"\nSaved: results/tables/adaptation_speed.csv")
    
    print("\n" + "="*80)
    print("STEP 5: SPECIFIC TRANSITION ANALYSIS")
    print("="*80)
    
    # Analyze specific critical transitions
    analyze_specific_transitions(results_df, vol_actual, vix)
    
    print("\n" + "="*80)
    print("STEP 6: VISUALIZATIONS")
    print("="*80)
    
    # Create visualizations
    plot_transition_performance(summary_df)
    plot_adaptation_speed(adaptation_df)
    
    print("\n" + "="*80)
    print("✅ REGIME TRANSITION ANALYSIS COMPLETE!")
    print("="*80)
    
    print("\n🎯 KEY INSIGHTS:")
    
    # Best adapter
    best_adapter = adaptation_df.iloc[0]
    print(f"\n  Best Adapter: {best_adapter['model']}")
    print(f"    Performance degradation: {best_adapter['degradation']:+.6f}")
    print(f"    (Lower = adapts faster to regime changes)")
    
    # Worst transition
    worst_trans = summary_df.loc[summary_df['pct_change'].idxmax()]
    print(f"\n  Hardest Transition: {worst_trans['transition']}")
    print(f"    Model: {worst_trans['model']}")
    print(f"    Performance drop: {worst_trans['pct_change']:.1f}%")
    
    print("\n📝 PAPER CONTRIBUTION:")
    print("  ✓ Novel temporal analysis (few papers do this)")
    print("  ✓ Shows adaptive intelligence")
    print("  ✓ Reveals which models handle transitions well")
    print("  ✓ COVID crisis transition analyzed")
    print("  ✓ Practical value: know when to switch models")

if __name__ == "__main__":
    main()
