"""
Script 3: Data Preprocessing
Clean data, handle missing values, and create train/val/test splits
Author: Volatility Research Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_raw_data():
    """Load the downloaded data"""
    print("Loading raw data...")
    prices = pd.read_csv('data/raw/prices_raw.csv', index_col=0, parse_dates=True)
    returns = pd.read_csv('data/raw/returns_raw.csv', index_col=0, parse_dates=True)
    vol = pd.read_csv('data/raw/realized_volatility_raw.csv', index_col=0, parse_dates=True)
    
    print(f"Prices shape: {prices.shape}")
    print(f"Returns shape: {returns.shape}")
    print(f"Volatility shape: {vol.shape}")
    
    return prices, returns, vol

def check_missing_values(data, name):
    """Check for missing values"""
    missing = data.isnull().sum()
    missing_pct = (missing / len(data)) * 100
    
    print(f"\n{name} - Missing Values:")
    print(missing_pct[missing_pct > 0])
    
    return missing

def handle_missing_values(data):
    """Handle missing values using forward fill"""
    print("\nHandling missing values...")
    
    # Forward fill (carry last valid observation forward)
    data_filled = data.fillna(method='ffill')
    
    # If any remaining NaN at the start, use backward fill
    data_filled = data_filled.fillna(method='bfill')
    
    # Check if any NaN remaining
    remaining_nan = data_filled.isnull().sum().sum()
    print(f"Remaining NaN values: {remaining_nan}")
    
    return data_filled

def detect_outliers(data, threshold=5):
    """
    Detect outliers using z-score method
    threshold=5 means values more than 5 std deviations from mean
    """
    print(f"\nDetecting outliers (threshold={threshold} std)...")
    
    outliers = {}
    for col in data.columns:
        z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
        outlier_mask = z_scores > threshold
        n_outliers = outlier_mask.sum()
        
        if n_outliers > 0:
            outliers[col] = n_outliers
            print(f"  {col}: {n_outliers} outliers")
    
    return outliers

def create_data_splits(data, train_end='2019-12-31', val_end='2021-12-31'):
    """
    Split data into train/validation/test sets
    Train: 2015-2019 (normal period before COVID)
    Validation: 2020-2021 (includes COVID crisis)
    Test: 2022-2024 (post-COVID period)
    """
    print("\nCreating data splits...")
    
    train = data[data.index <= train_end]
    val = data[(data.index > train_end) & (data.index <= val_end)]
    test = data[data.index > val_end]
    
    print(f"Train set: {train.index[0]} to {train.index[-1]} ({len(train)} days)")
    print(f"Val set:   {val.index[0]} to {val.index[-1]} ({len(val)} days)")
    print(f"Test set:  {test.index[0]} to {test.index[-1]} ({len(test)} days)")
    
    return train, val, test

def identify_regime_periods(data):
    """
    Identify market regimes based on VIX levels
    Normal: VIX < 20
    Stress: VIX 20-30
    Crisis: VIX > 30
    """
    print("\nIdentifying market regimes...")
    
    # Load prices to get VIX
    prices = pd.read_csv('data/raw/prices_raw.csv', index_col=0, parse_dates=True)
    vix = prices['^VIX']
    
    # Create regime labels
    regime = pd.Series(index=vix.index, dtype=str)
    regime[vix < 20] = 'Normal'
    regime[(vix >= 20) & (vix < 30)] = 'Stress'
    regime[vix >= 30] = 'Crisis'
    
    # Count days in each regime
    regime_counts = regime.value_counts()
    print("\nRegime distribution:")
    print(regime_counts)
    print(f"\nPercentage:")
    print((regime_counts / len(regime) * 100).round(2))
    
    return regime

def visualize_regimes(regime):
    """Visualize market regimes over time"""
    print("\nCreating regime visualization...")
    
    # Load VIX for plotting
    prices = pd.read_csv('data/raw/prices_raw.csv', index_col=0, parse_dates=True)
    vix = prices['^VIX']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot VIX
    ax.plot(vix.index, vix, linewidth=1, color='black', alpha=0.7)
    
    # Color background by regime
    normal_mask = regime == 'Normal'
    stress_mask = regime == 'Stress'
    crisis_mask = regime == 'Crisis'
    
    # Fill regions
    for i in range(len(regime)-1):
        if crisis_mask.iloc[i]:
            ax.axvspan(regime.index[i], regime.index[i+1], alpha=0.3, color='red')
        elif stress_mask.iloc[i]:
            ax.axvspan(regime.index[i], regime.index[i+1], alpha=0.2, color='orange')
        else:
            ax.axvspan(regime.index[i], regime.index[i+1], alpha=0.1, color='green')
    
    # Add horizontal lines for thresholds
    ax.axhline(y=20, color='orange', linestyle='--', linewidth=1, label='Stress threshold')
    ax.axhline(y=30, color='red', linestyle='--', linewidth=1, label='Crisis threshold')
    
    ax.set_title('Market Regimes Based on VIX Levels', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('VIX Level', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add text annotations
    ax.text(0.02, 0.95, 'Green = Normal (VIX < 20)', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.text(0.02, 0.88, 'Orange = Stress (VIX 20-30)', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
    ax.text(0.02, 0.81, 'Red = Crisis (VIX > 30)', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('results/figures/05_market_regimes.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/05_market_regimes.png")
    plt.close()

def save_processed_data(data, filename):
    """Save processed data"""
    filepath = f'data/processed/{filename}'
    data.to_csv(filepath)
    print(f"Saved: {filepath}")

def main():
    print("="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    # Load data
    prices, returns, vol = load_raw_data()
    
    print("\n" + "="*60)
    print("STEP 1: CHECK DATA QUALITY")
    print("="*60)
    
    # Check missing values
    check_missing_values(prices, "Prices")
    check_missing_values(returns, "Returns")
    check_missing_values(vol, "Volatility")
    
    # Detect outliers
    outliers_returns = detect_outliers(returns, threshold=5)
    outliers_vol = detect_outliers(vol, threshold=5)
    
    print("\n" + "="*60)
    print("STEP 2: HANDLE MISSING VALUES")
    print("="*60)
    
    # Fill missing values
    prices_clean = handle_missing_values(prices)
    returns_clean = handle_missing_values(returns)
    vol_clean = handle_missing_values(vol)
    
    print("\n" + "="*60)
    print("STEP 3: CREATE DATA SPLITS")
    print("="*60)
    
    # Split volatility data (our main target)
    vol_train, vol_val, vol_test = create_data_splits(vol_clean)
    
    # Split returns data (for features)
    returns_train, returns_val, returns_test = create_data_splits(returns_clean)
    
    print("\n" + "="*60)
    print("STEP 4: IDENTIFY MARKET REGIMES")
    print("="*60)
    
    # Create regime labels
    regime = identify_regime_periods(prices_clean)
    
    # Visualize regimes
    visualize_regimes(regime)
    
    print("\n" + "="*60)
    print("STEP 5: SAVE PROCESSED DATA")
    print("="*60)
    
    # Save cleaned data
    save_processed_data(prices_clean, 'prices_clean.csv')
    save_processed_data(returns_clean, 'returns_clean.csv')
    save_processed_data(vol_clean, 'volatility_clean.csv')
    
    # Save regime labels
    save_processed_data(regime, 'market_regimes.csv')
    
    # Save splits
    save_processed_data(vol_train, 'vol_train.csv')
    save_processed_data(vol_val, 'vol_val.csv')
    save_processed_data(vol_test, 'vol_test.csv')
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*60)
    print("\nNext step: Run feature engineering script")
    print("Command: py -3.10 scripts\\04_feature_engineering.py")

if __name__ == "__main__":
    main()
