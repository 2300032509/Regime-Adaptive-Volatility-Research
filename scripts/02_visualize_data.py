"""
Script 2: Data Visualization
Create initial plots to understand your data
Author: Volatility Research Project
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load the data we downloaded"""
    prices = pd.read_csv('data/raw/prices_raw.csv', index_col=0, parse_dates=True)
    returns = pd.read_csv('data/raw/returns_raw.csv', index_col=0, parse_dates=True)
    vol = pd.read_csv('data/raw/realized_volatility_raw.csv', index_col=0, parse_dates=True)
    
    return prices, returns, vol

def plot_price_series(prices):
    """Plot price time series for all assets"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle('Price Time Series (2015-2024)', fontsize=16, fontweight='bold')
    
    tickers = prices.columns
    axes = axes.flatten()
    
    for i, ticker in enumerate(tickers):
        if i < len(axes):
            axes[i].plot(prices.index, prices[ticker], linewidth=1)
            axes[i].set_title(ticker, fontweight='bold')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Price ($)')
            axes[i].grid(True, alpha=0.3)
            
            # Highlight COVID crash
            axes[i].axvspan('2020-03-01', '2020-06-01', alpha=0.2, color='red', label='COVID Crisis')
    
    plt.tight_layout()
    plt.savefig('results/figures/01_price_series.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/01_price_series.png")
    plt.close()

def plot_returns_distribution(returns):
    """Plot return distributions"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Daily Returns Distribution', fontsize=16, fontweight='bold')
    
    tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'JPM', 'XOM', 'TSLA', '^VIX']
    axes = axes.flatten()
    
    for i, ticker in enumerate(tickers):
        if ticker in returns.columns:
            data = returns[ticker].dropna()
            
            axes[i].hist(data, bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{ticker}', fontweight='bold')
            axes[i].set_xlabel('Daily Return')
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(0, color='red', linestyle='--', linewidth=1)
            
            # Add statistics
            mean = data.mean()
            std = data.std()
            axes[i].text(0.05, 0.95, f'μ={mean:.4f}\nσ={std:.4f}', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('results/figures/02_returns_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/02_returns_distribution.png")
    plt.close()

def plot_volatility_series(vol):
    """Plot realized volatility over time"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot SPY volatility as main focus
    ax.plot(vol.index, vol['SPY']*100, linewidth=2, label='S&P 500 (SPY)', color='blue')
    ax.plot(vol.index, vol['QQQ']*100, linewidth=1.5, label='NASDAQ (QQQ)', color='green', alpha=0.7)
    
    # Highlight COVID crisis
    ax.axvspan('2020-03-01', '2020-06-01', alpha=0.2, color='red', label='COVID Crisis')
    
    ax.set_title('Realized Volatility Over Time (21-day window)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Annualized Volatility (%)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/03_volatility_series.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/03_volatility_series.png")
    plt.close()

def plot_vix_comparison(vol):
    """Compare calculated volatility with VIX"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot both
    ax.plot(vol.index, vol['SPY']*100, linewidth=2, label='Realized Vol (SPY)', color='blue')
    
    # VIX is already in percentage points, so we plot it directly
    # But we need to load VIX from prices since it's an index
    prices = pd.read_csv('data/raw/prices_raw.csv', index_col=0, parse_dates=True)
    ax.plot(prices.index, prices['^VIX'], linewidth=2, label='VIX Index', color='red', alpha=0.7)
    
    ax.set_title('Realized Volatility vs VIX', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Volatility', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Highlight crisis
    ax.axvspan('2020-03-01', '2020-06-01', alpha=0.2, color='orange', label='COVID Crisis')
    
    plt.tight_layout()
    plt.savefig('results/figures/04_vix_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/04_vix_comparison.png")
    plt.close()

def generate_summary_stats(returns, vol):
    """Generate summary statistics table"""
    
    summary = pd.DataFrame({
        'Mean Return': returns.mean() * 252 * 100,  # Annualized
        'Volatility': returns.std() * np.sqrt(252) * 100,  # Annualized
        'Sharpe Ratio': (returns.mean() / returns.std()) * np.sqrt(252),
        'Min Return': returns.min() * 100,
        'Max Return': returns.max() * 100,
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis()
    })
    
    summary = summary.round(3)
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary)
    
    # Save to CSV
    summary.to_csv('results/tables/summary_statistics.csv')
    print("\nSaved: results/tables/summary_statistics.csv")

def main():
    print("="*60)
    print("VISUALIZING YOUR DATA")
    print("="*60)
    
    # Create results folders if they don't exist
    import os
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    prices, returns, vol = load_data()
    
    # Generate plots
    print("\nGenerating visualizations...")
    print("1. Price series...")
    plot_price_series(prices)
    
    print("2. Returns distribution...")
    plot_returns_distribution(returns)
    
    print("3. Volatility time series...")
    plot_volatility_series(vol)
    
    print("4. VIX comparison...")
    plot_vix_comparison(vol)
    
    # Summary statistics
    generate_summary_stats(returns, vol)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print("\nCheck the 'results/figures/' folder for your plots!")

if __name__ == "__main__":
    main()
