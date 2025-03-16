# Bitcoin_Risk_Models
Bitcoin Risk Prediction Algorithms Prototype.

## Overview
This is an open-source prototype **Bitcoin Risk Management Library** in Python, built using **NumPy, SciPy, Pandas, Matplotlib, and XGBoost**. It provides essential risk metrics, statistical analysis, machine learning-based volatility predictions, and visualizations for analyzing Bitcoin's historical price performance and risk characteristics. The bigger goal is to create an unstoppable Bitcoin Risk Management system ,that will be AI agent friendly as we firmly believe that Bitcoin will be the beloved AI agents "money".

## Features
- **Risk Metrics:**
  - Daily Returns & Cumulative Returns
  - Sharpe Ratio
  - Sortino Ratio
  - Value at Risk (VaR) & Conditional VaR (CVaR)
  - Skewness & Kurtosis
  - Volatility (Annualized)
  - Maximum Drawdown
  - Omega Ratio
  - Compound Return
  - **Cornish-Fisher VaR Calculation**
- **Bitcoin Volatility Magnitude Prediction:**
  - Feature Engineering (Rolling Statistics, Percent Changes, Time Features)
  - Data Splitting (Train, Validation, Test)
  - XGBoost Model for Bitcoin Volatility Prediction
  - Grid Search for Hyperparameter Optimization
  - Results [**MAE**: 3.96, **RMSE**: 5.38, **Accuracy**: 0.74, **Precision**: 0.63, **Recall**: 0.84, **AUC**: 0.75, **Volatility Threshold used for classification metrics**: 5.00, **Confusion Matrix**: (232, 114, 37, 193) ]
    
- **Visualizations:**
  - Return Distribution
  - Drawdowns
  - Rolling Volatility
  - Closing Price Over Time
  - Actual vs. Predicted Volatility
  - Correlation Heatmap
 
 **Others:**
   
   -More pragmatic updates to come, so stay tuned. Also, we eagerly request your critics and suggestions. 

## Installation
This library requires Python 3.7+ and the following dependencies:

```bash
pip install numpy pandas scipy matplotlib seaborn statsmodels xgboost scikit-learn
```

## Usage
Import the library and initialize it with a **pandas Series** of Bitcoin closing prices (data was obtained from Yahoo Finance) or a CSV file:

```python
import pandas as pd
import majestify as ma  # Updated import statement

# Load historical BTC price data
analysis = ma.BitcoinDataAnalysis("bitcoin_prices.csv")

# Display Data Overview
analysis.data_overview()

# Compute Volatility and Test for Stationarity
analysis.compute_volatility()
analysis.test_stationarity()

# Train a machine learning model for volatility prediction
analysis.split_data()
model = analysis.train_xgboost_model()
analysis.plot_actual_vs_predicted(model)

# Generate risk visualization plots
analysis.plot_closing_price()
analysis.compute_correlation_matrix()
analysis.plot_drawdown()
analysis.plot_rolling_volatility()
```

## License
This project is released under a **Dual License**:
- **MIT License** for permissive open-source use.
- **GPL-3.0** for those who prefer stricter copyleft terms.

## Contributions
We welcome contributions! Feel free to submit pull requests or open issues.

## Data Source
This library is designed to work with **Bitcoin price data** from various sources, including **CoinGecko, Yahoo Finance, and Crypto exchanges**.

## Community

Join Us on Discord: https://discord.gg/hNhjAdRm

---
**Developed with first-principles thinking to provide robust Bitcoin risk management !** 🚀
