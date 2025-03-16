import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

class BitcoinRiskMetrics:
    """
    A library for calculating risk metrics for Bitcoin, including:
    - Returns (daily & cumulative)
    - Sharpe Ratio
    - Sortino Ratio
    - Value at Risk (VaR) & Conditional VaR (CVaR)
    - Cornish-Fisher Value at Risk (VaR)
    - Skewness & Kurtosis
    - Volatility (Annualized)
    - Maximum Drawdown
    - Omega Ratio
    - Compound Return
    - Bitcoin Closing Price Over Time Visualization
    """
    
    def __init__(self, price_series):
        """Initialize with a pandas Series of Bitcoin closing prices."""
        self.prices = price_series
        self.returns = self.prices.pct_change().dropna()
        self.dates = price_series.index
    
    def plot_closing_price(self):
        """Plots Bitcoin Closing Price over Time."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.dates, self.prices, label='BTC Closing Price', color='blue')
        plt.xlabel("Year")
        plt.ylabel("Price (USD)")
        plt.title("Bitcoin Closing Price Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_drawdown(self):
        """Plots the drawdown over time with Year on the X-axis."""
        cumulative_returns = (1 + self.returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.dates[1:], drawdown, color='red', label='Drawdown')
        plt.xlabel("Year")
        plt.ylabel("Drawdown")
        plt.title("Bitcoin Drawdown Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_rolling_volatility(self, window=30):
        """Plots rolling volatility with Year on the X-axis."""
        rolling_volatility = self.returns.rolling(window).std() * np.sqrt(252)
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.dates[1:], rolling_volatility, label=f"{window}-day Rolling Volatility", color='blue')
        plt.xlabel("Year")
        plt.ylabel("Volatility")
        plt.title("Bitcoin Rolling Volatility")
        plt.legend()
        plt.grid(True)
        plt.show()
