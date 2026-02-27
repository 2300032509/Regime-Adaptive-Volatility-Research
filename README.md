# Regime-Adaptive Volatility Research

A comprehensive research project on regime-adaptive volatility forecasting using traditional time series models, deep learning, and hybrid approaches.

## Overview

This project implements and compares various volatility forecasting models including GARCH, ARIMA, LSTM neural networks, and ensemble methods with regime-switching capabilities.

## Project Structure

```
volatility_research/
├── scripts/
│   ├── 01_download_data.py         # Data download
│   ├── 02_visualize_data.py       # Data visualization
│   ├── 03_data_preprocessing.py   # Data cleaning and preprocessing
│   ├── 04_feature_engineering.py # Feature engineering
│   ├── 05_garch_models.py        # GARCH family models
│   ├── 06_arima_models.py        # ARIMA models
│   ├── 07_lstm_model.py          # LSTM neural network
│   ├── 08_hybrid_garch_lstm.py   # Hybrid GARCH-LSTM model
│   ├── 09_ensemble_model.py      # Ensemble forecasting
│   ├── 10_regime_adaptive.py     # Regime-adaptive switching
│   ├── 11_regime_adaptive_extended.py  # Extended regime model
│   ├── 12_asymmetric_loss.py     # Asymmetric loss analysis
│   ├── 13_transition_analysis.py # Regime transition analysis
│   └── 14_cross_market.py        # Cross-market analysis
├── data/
│   ├── raw/                       # Raw data files
│   ├── processed/                # Processed data
│   └── features/                 # Engineered features
├── results/
│   ├── figures/                  # Visualization outputs
│   ├── tables/                   # Forecast results
│   └── saved_models/             # Trained models
└── .gitignore
```

## Models Implemented

### Traditional Models
- **GARCH**: Generalized Autoregressive Conditional Heteroskedasticity
- **EGARCH**: Exponential GARCH
- **GJR-GARCH**: Asymmetric GARCH
- **ARIMA**: AutoRegressive Integrated Moving Average

### Machine Learning Models
- **LSTM**: Long Short-Term Memory neural network

### Hybrid & Ensemble
- **Hybrid GARCH-LSTM**: Combines traditional GARCH with deep learning
- **Ensemble Model**: Weighted combination of multiple forecasts
- **Regime-Adaptive**: Switches between models based on market regime

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
arch
statsmodels
tensorflow
keras
```

## Usage

1. Download data:
```bash
python scripts/01_download_data.py
```

2. Preprocess data:
```bash
python scripts/03_data_preprocessing.py
```

3. Run models:
```bash
python scripts/05_garch_models.py
python scripts/06_arima_models.py
python scripts/07_lstm_model.py
```

4. Run regime-adaptive model:
```bash
python scripts/10_regime_adaptive.py
```

## Results

The project includes comprehensive analysis of:
- Volatility forecasting accuracy
- Regime detection and transition analysis
- Model comparison across different market conditions
- Asymmetric loss sensitivity
- Cross-market performance

## License

MIT License
