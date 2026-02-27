"""
Script 1: Download Financial Data (FINAL VERSION)
Downloads stock price data and VIX from Yahoo Finance
Author: Volatility Research Project
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os

# Configuration
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'

# Assets to download
TICKERS = ['SPY', 'QQQ', '^VIX', 'AAPL', 'MSFT', 'JPM', 'XOM', 'TSLA']

def download_single_ticker(ticker, start_date, end_date):
    """Download data for a single ticker"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if 'Adj Close' in data.columns:
            return data['Adj Close']
        elif 'Close' in data.columns:
            return data['Close']
        else:
            return data
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None

def download_all_data(tickers, start_date, end_date):
    """Download data for all tickers one by one"""
    print(f"Downloading data from {start_date} to {end_date}...")
    
    all_data = {}
    
    for ticker in tickers:
        print(f"  Downloading {ticker}...", end=" ")
        data = download_single_ticker(ticker, start_date, end_date)
        if data is not None and len(data) > 0:
            all_data[ticker] = data
            print("✓")
        else:
            print("✗")
    
    # Combine into single DataFrame (using concat to handle different lengths)
    df = pd.concat(all_data, axis=1)
    df.columns = list(all_data.keys())
    
    return df

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
    prices = download_all_data(TICKERS, START_DATE, END_DATE)
    
    print(f"\nData downloaded successfully!")
    print(f"Shape: {prices.shape}")
    print(f"Columns: {list(prices.columns)}")
    
    # Save raw prices
    save_data(prices, 'prices_raw.csv')
    
    print("\n" + "="*60)
    print("STEP 2: CALCULATING RETURNS")
    print("="*60)
    
    # Calculate returns
    returns = calculate_returns(prices)
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
    print(f"\nData shape: {prices.shape}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"Trading days: {len(prices)}")
    
    # Quick summary
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    print("\nMean Daily Returns (%):")
    print((returns.mean() * 100).round(3))
    print("\nAnnualized Volatility (%):")
    print((returns.std() * np.sqrt(252) * 100).round(2))
    
    print("\n" + "="*60)
    print("✅ SUCCESS! Now run: py -3.10 scripts\\02_visualize_data.py")
    print("="*60)

if __name__ == "__main__":
    main()
