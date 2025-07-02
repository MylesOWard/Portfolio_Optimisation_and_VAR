import matplotlib
import pandas as pd
import datetime as dt
import numpy as np
import time
import yfinance as yf
from scipy.stats import norm
from matplotlib import pyplot as plt

# time over which to take data
years = 10
days = 365*years

end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days)

# tickers for five publically traded funds / companies
# the companies must have had their IPO within the last 10 years
tickers = ["SPY", "BND", "GLD", "QQQ", "VTI"]

# gather data on each tick between the specified dates 
# gathers close values for each tick, "Close" is the adjusted close value
# adjusted close value takes into account stcok splits and dividends 
adj_close_df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]

# print statement to verify, too many requests at once will cause a yf block 
print(adj_close_df)

# ln daily returns using the np.log function, change in close value from previous day 
# dropping any values which are undefined with the dropna function
log_returns = np.log(adj_close_df/adj_close_df.shift(1))
log_returns = log_returns.dropna()

print(log_returns)

# total value of portfolio, 1 million here 
portfolio_value = 1000000

# assign weighting of each component in the portfolio (equal in this case)
# weight = 1/len(tickers)
# weights = [weight, weight, weight, weight, weight]
weights = np.array([1/len(tickers)] * len(tickers))
print(weights)

# dot product of returns log and weights 
historic_returns = log_returns.dot(weights)
print(historic_returns.head())


### Now Define the Window of Analysis ###

# 5 day window is standard - one trading week
day_window = 5

# rolling window 
# every 5 day combiation (requires at least 5 days data so first 4 are n/a)
range_returns = historic_returns.rolling(window = day_window).sum()
range_returns = range_returns.dropna()

# check range returns
print("Range Returns")
print(range_returns)

# confidence intervals, 95% confidence intervals 
confidence_level = 0.95
alpha = 1 - confidence_level  # 0.05
VaR = -np.percentile(range_returns.values, alpha * 100) * portfolio_value

# plot guassian 
return_window = day_window
range_returns = historic_returns.rolling(window=return_window).sum()
range_returns = range_returns.dropna()

range_returns_dollar = range_returns * portfolio_value

plt.hist(range_returns_dollar.dropna(), bins=50, density=True)
plt.xlabel(f'{return_window}-Day Portfolio Return (Dollar Value)')
plt.ylabel('Frequency')
plt.title(f'Distribution of Portfolio {return_window}-Day Returns (Dollar Value)')
plt.axvline(-VaR, color='r', linestyle='dashed', linewidth=2, label=f'VaR at {confidence_level:.0%} confidence level')
plt.legend()
plt.show()
