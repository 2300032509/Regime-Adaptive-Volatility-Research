"""
Script 1: Download Financial Data (FIXED VERSION)
Downloads stock price data and VIX from Yahoo Finance
Author: Volatility Research Project
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Configuration
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'

# Assets to download
TICKERS = {
    'indices': ['SPY', 'QQQ', '^VIX'],  # S&P 500, NASDAQ, VIX
    'stocks': ['AAPL', 'MSFT', 'JPM', 'XOM', 'TSLA']
}

def download_data(tickers, start_date, end_date):
    """Download historical price data"""
    print(f"Downloading data from {start_date} to {end_date}...")
    
    all_tickers = tickers['indices'] + tickers['stocks']
    
    # Download data
    data = yf.download(
        all_tickers,
        start=start_date,
        end=end_date,
        progress=True
    )
    
    return data

def calculate_returns(prices):
    """Calculate log returns"""
    returns = np.log(prices / prices.shift(1))
    return returns

def calculate_realized_volatility(returns, window=21):
    """
    Calculate realized volatility
    window=21 means approximately 1 month (21 trading days)
    """
    # Annualized volatility
    realized_vol = returns.rolling(window=window).std() * np.sqrt(252)
    return realized_vol

def save_data(data, filename):
    """Save data to CSV"""
    filepath = f'data/raw/{filename}'
    data.to_csv(filepath)
    print(f"Saved: {filepath}")

def main():
    print("="*60)
    print("STEP 1: DOWNLOADING FINANCIAL DATA")
    print("="*60)
    
    # Download price data
    prices = download_data(TICKERS, START_DATE, END_DATE)
    
    # FIX: Handle different data structures from yfinance
    # Check if data has multi-level columns
    if isinstance(prices.columns, pd.MultiIndex):
        # Multi-ticker download - extract Adj Close
        adj_close = prices['Adj Close']
    else:
        # Single ticker or already simplified
        adj_close = prices
    
    print(f"\nData structure: {adj_close.shape}")
    print(f"Columns: {list(adj_close.columns)}")
    
    # Save raw prices
    save_data(adj_close, 'prices_raw.csv')
    
    print("\n" + "="*60)
    print("STEP 2: CALCULATING RETURNS")
    print("="*60)
    
    # Calculate returns
    returns = calculate_returns(adj_close)
    save_data(returns, 'returns_raw.csv')
    
    print("\n" + "="*60)
    print("STEP 3: CALCULATING REALIZED VOLATILITY")
    print("="*60)
    
    # Calculate realized volatility
    realized_vol = calculate_realized_volatility(returns, window=21)
    save_data(realized_vol, 'realized_volatility_raw.csv')
    
    print("\n" + "="*60)
    print("DATA DOWNLOAD COMPLETE!")
    print("="*60)
    print(f"\nData shape: {adj_close.shape}")
    print(f"Date range: {adj_close.index[0]} to {adj_close.index[-1]}")
    print(f"Trading days: {len(adj_close)}")
    
    # Quick summary
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    print("\nMean Daily Returns (%):")
    print((returns.mean() * 100).round(3))
    print("\nAnnualized Volatility (%):")
    print((returns.std() * np.sqrt(252) * 100).round(2))
    
    print("\n" + "="*60)
    print("SUCCESS! You're ready for visualization!")
    print("="*60)

if __name__ == "__main__":
    main()