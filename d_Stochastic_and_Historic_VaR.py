import pandas as pd
import datetime as dt
import numpy as np
import yfinance as yf
from matplotlib import pyplot as plt

# Input parameters
years = 5
days = 365 * years
portfolio_value = 1_000_000  # underscores for clarity

tickers = ["SPY", "BND", "GLD", "QQQ", "VTI"]
weights = np.array([1 / len(tickers)] * len(tickers))


confidence_level = 0.95      # 5% VaR as before
alpha = 1 - confidence_level

day_window = 5

# Dates
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days)

# Data
close = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
log_returns = np.log(close / close.shift(1))
log_returns = log_returns.dropna()

# Splitting into 4 years training and 1 year for testing
# Enables predictions with both methods and a comparison
split_index = int(len(log_returns) * 4 / 5)
train_returns = log_returns.iloc[:split_index]
test_returns = log_returns.iloc[split_index:]

# Historical VaR (from test period)
weighted_returns = log_returns.dot(weights)
day_window_returns = weighted_returns.rolling(day_window).sum().dropna()
split_index = int(len(day_window_returns.dropna()) * 4 / 5)
dollar_returns = day_window_returns * portfolio_value
historical_var = -np.percentile(day_window_returns.iloc[split_index:].values, alpha * 100) * portfolio_value

# Monte Carlo Simulation
np.random.seed(42) # Random Seed Ensures Reproduable Results 
mu = train_returns.mean()
cov = train_returns.cov()
num_simulations = 10_000
num_days = day_window

simulated_paths = np.random.multivariate_normal(mu, cov, size=(num_simulations, num_days))
simulated_weighted_returns = simulated_paths @ weights
simulated_total_returns = simulated_weighted_returns.sum(axis=1)
monte_carlo_var = -np.percentile(simulated_total_returns, alpha * 100) * portfolio_value

# Results
print(f"Historical VaR (95%) over 5 days: ${historical_var:,.2f}")
print(f"Monte Carlo VaR (95%) over 5 days: ${monte_carlo_var:,.2f}")

# Plot Monte Carlo simulated distribution
plt.hist(simulated_total_returns * portfolio_value, bins=100, density=True, alpha=0.7, label='Monte Carlo Simulated')
plt.axvline(-monte_carlo_var, color='red', linestyle='dashed', label='Monte Carlo VaR')
plt.xlabel("5-Day Portfolio Returns ($)")
plt.ylabel("Frequency")
plt.title("Monte Carlo Simulated 5-Day Portfolio Returns")
plt.legend()
plt.show()
