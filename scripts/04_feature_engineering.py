"""
Script 4: Feature Engineering
Create lagged features, rolling statistics, and regime indicators for ML models
Author: Volatility Research Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_clean_data():
    """Load preprocessed data"""
    print("Loading clean data...")
    prices = pd.read_csv('data/processed/prices_clean.csv', index_col=0, parse_dates=True)
    returns = pd.read_csv('data/processed/returns_clean.csv', index_col=0, parse_dates=True)
    vol = pd.read_csv('data/processed/volatility_clean.csv', index_col=0, parse_dates=True)
    regimes = pd.read_csv('data/processed/market_regimes.csv', index_col=0, parse_dates=True)
    
    return prices, returns, vol, regimes

def create_lagged_features(data, lags=[1, 2, 3, 5, 10, 20], prefix='lag'):
    """
    Create lagged features
    For volatility forecasting, we use past volatility as features
    """
    print(f"\nCreating lagged features (lags: {lags})...")
    
    features = pd.DataFrame(index=data.index)
    
    for col in data.columns:
        for lag in lags:
            features[f'{prefix}_{col}_{lag}'] = data[col].shift(lag)
    
    print(f"Created {len(features.columns)} lagged features")
    return features

def create_rolling_features(data, windows=[5, 10, 20, 60], stats=['mean', 'std', 'min', 'max']):
    """
    Create rolling window statistics
    Capture short-term and long-term trends
    """
    print(f"\nCreating rolling features (windows: {windows})...")
    
    features = pd.DataFrame(index=data.index)
    
    for col in data.columns:
        for window in windows:
            if 'mean' in stats:
                features[f'roll_mean_{col}_{window}'] = data[col].rolling(window).mean()
            if 'std' in stats:
                features[f'roll_std_{col}_{window}'] = data[col].rolling(window).std()
            if 'min' in stats:
                features[f'roll_min_{col}_{window}'] = data[col].rolling(window).min()
            if 'max' in stats:
                features[f'roll_max_{col}_{window}'] = data[col].rolling(window).max()
    
    print(f"Created {len(features.columns)} rolling features")
    return features

def create_volatility_features(vol):
    """
    Create volatility-specific features
    """
    print("\nCreating volatility-specific features...")
    
    features = pd.DataFrame(index=vol.index)
    
    # For S&P 500 (SPY) as main target
    spy_vol = vol['SPY']
    
    # Volatility change (momentum)
    features['vol_change_1d'] = spy_vol.diff(1)
    features['vol_change_5d'] = spy_vol.diff(5)
    
    # Volatility ratios (relative to moving average)
    features['vol_ratio_ma5'] = spy_vol / spy_vol.rolling(5).mean()
    features['vol_ratio_ma20'] = spy_vol / spy_vol.rolling(20).mean()
    
    # Volatility percentile (relative to historical)
    features['vol_percentile_60d'] = spy_vol.rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )
    
    # High volatility indicator
    features['high_vol_flag'] = (spy_vol > spy_vol.rolling(60).quantile(0.75)).astype(int)
    
    print(f"Created {len(features.columns)} volatility features")
    return features

def create_regime_features(regimes):
    """
    Create regime-based features (one-hot encoding)
    """
    print("\nCreating regime features...")
    
    # One-hot encode regimes
    regime_dummies = pd.get_dummies(regimes.iloc[:, 0], prefix='regime')
    
    print(f"Created {len(regime_dummies.columns)} regime features")
    return regime_dummies

def create_vix_features(prices):
    """
    Create VIX-based features for regime detection
    """
    print("\nCreating VIX features...")
    
    vix = prices['^VIX']
    features = pd.DataFrame(index=prices.index)
    
    # VIX levels
    features['vix_level'] = vix
    
    # VIX change
    features['vix_change_1d'] = vix.diff(1)
    features['vix_change_5d'] = vix.diff(5)
    
    # VIX relative to MA
    features['vix_ratio_ma20'] = vix / vix.rolling(20).mean()
    
    # VIX term structure (if we had VIX futures, but we'll use approximation)
    features['vix_ma_spread'] = vix.rolling(5).mean() - vix.rolling(20).mean()
    
    print(f"Created {len(features.columns)} VIX features")
    return features

def create_cross_asset_features(vol):
    """
    Create features from relationships between assets
    """
    print("\nCreating cross-asset features...")
    
    features = pd.DataFrame(index=vol.index)
    
    # Correlation between SPY and QQQ volatility (rolling 20-day)
    features['spy_qqq_vol_corr'] = vol['SPY'].rolling(20).corr(vol['QQQ'])
    
    # Volatility spread
    features['spy_qqq_vol_spread'] = vol['SPY'] - vol['QQQ']
    
    # Average volatility across tech stocks
    tech_stocks = ['AAPL', 'MSFT', 'TSLA']
    features['tech_avg_vol'] = vol[tech_stocks].mean(axis=1)
    
    # Market-wide volatility (average of all stocks)
    features['market_avg_vol'] = vol[['SPY', 'QQQ', 'AAPL', 'MSFT', 'JPM', 'XOM']].mean(axis=1)
    
    print(f"Created {len(features.columns)} cross-asset features")
    return features

def combine_all_features(vol, returns, prices, regimes, target_ticker='SPY'):
    """
    Combine all feature sets into final feature matrix
    """
    print("\n" + "="*60)
    print("COMBINING ALL FEATURES")
    print("="*60)
    
    # Focus on SPY (S&P 500) as main target
    target = vol[target_ticker].shift(-1)  # Next day's volatility
    target.name = 'target_volatility'
    
    # Create all features
    lag_features_vol = create_lagged_features(vol[[target_ticker]], lags=[1, 2, 3, 5, 10, 20], prefix='vol_lag')
    lag_features_ret = create_lagged_features(returns[[target_ticker]], lags=[1, 2, 3, 5], prefix='ret_lag')
    
    roll_features = create_rolling_features(vol[[target_ticker]], windows=[5, 10, 20], stats=['mean', 'std'])
    
    vol_features = create_volatility_features(vol)
    regime_features = create_regime_features(regimes)
    vix_features = create_vix_features(prices)
    cross_features = create_cross_asset_features(vol)
    
    # Combine all
    feature_matrix = pd.concat([
        lag_features_vol,
        lag_features_ret,
        roll_features,
        vol_features,
        regime_features,
        vix_features,
        cross_features,
        target
    ], axis=1)
    
    # Drop rows with NaN (due to lagging and rolling)
    print(f"\nBefore cleaning: {feature_matrix.shape}")
    feature_matrix_clean = feature_matrix.dropna()
    print(f"After cleaning: {feature_matrix_clean.shape}")
    
    return feature_matrix_clean

def split_features_target(data):
    """Separate features and target"""
    X = data.drop('target_volatility', axis=1)
    y = data['target_volatility']
    
    return X, y

def visualize_feature_importance_preview(X, y):
    """
    Quick correlation analysis to preview important features
    """
    print("\nAnalyzing feature correlations with target...")
    
    # Calculate correlations
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    
    # Plot top 20
    fig, ax = plt.subplots(figsize=(10, 8))
    correlations.head(20).plot(kind='barh', ax=ax)
    ax.set_title('Top 20 Features by Correlation with Target Volatility', fontweight='bold')
    ax.set_xlabel('Absolute Correlation')
    plt.tight_layout()
    plt.savefig('results/figures/06_feature_correlations.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/06_feature_correlations.png")
    plt.close()
    
    print("\nTop 10 most correlated features:")
    print(correlations.head(10))

def save_features(data, filename):
    """Save feature matrix"""
    filepath = f'data/features/{filename}'
    data.to_csv(filepath)
    print(f"\nSaved: {filepath}")

def main():
    print("="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    
    # Load data
    prices, returns, vol, regimes = load_clean_data()
    
    # Create feature matrix
    feature_matrix = combine_all_features(vol, returns, prices, regimes, target_ticker='SPY')
    
    print("\n" + "="*60)
    print("FEATURE MATRIX SUMMARY")
    print("="*60)
    print(f"Total features: {feature_matrix.shape[1] - 1}")  # -1 for target
    print(f"Total samples: {feature_matrix.shape[0]}")
    print(f"Date range: {feature_matrix.index[0]} to {feature_matrix.index[-1]}")
    
    # Split features and target
    X, y = split_features_target(feature_matrix)
    
    # Visualize feature importance preview
    visualize_feature_importance_preview(X, y)
    
    # Save feature matrix
    save_features(feature_matrix, 'feature_matrix_full.csv')
    save_features(X, 'features_X.csv')
    save_features(y, 'target_y.csv')
    
    # Create train/val/test splits
    print("\n" + "="*60)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("="*60)
    
    train_end = '2019-12-31'
    val_end = '2021-12-31'
    
    train_data = feature_matrix[feature_matrix.index <= train_end]
    val_data = feature_matrix[(feature_matrix.index > train_end) & (feature_matrix.index <= val_end)]
    test_data = feature_matrix[feature_matrix.index > val_end]
    
    print(f"Train: {train_data.shape[0]} samples")
    print(f"Val:   {val_data.shape[0]} samples")
    print(f"Test:  {test_data.shape[0]} samples")
    
    # Save splits
    save_features(train_data, 'train_data.csv')
    save_features(val_data, 'val_data.csv')
    save_features(test_data, 'test_data.csv')
    
    print("\n" + "="*60)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("="*60)
    print("\nYou now have:")
    print("  ✓ Feature matrix with", X.shape[1], "features")
    print("  ✓ Train/validation/test splits ready")
    print("  ✓ Ready to build models!")
    print("\nNext: Start implementing GARCH models (Week 2)")

if __name__ == "__main__":
    main()
