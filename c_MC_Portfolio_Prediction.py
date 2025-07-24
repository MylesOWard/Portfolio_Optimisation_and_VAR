import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt

# import data 
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)["Close"]
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    # this matrix will be posetive definite or semi definite in every case
    # it will always be symmetric allowing for decomposition
    covMatrix = returns.cov()
    return meanReturns, covMatrix

# United States company stocks
# ETFs can also be simulated readily 
stocklist = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM"]
stocks = [stock for stock in stocklist]

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)
print(meanReturns)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

print(weights)

# mc sims
# number of simulations 

sims_number = 100
T = 100 # time in days as before 

meanM = np.full(shape=(T, len(weights)), fill_value = meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape = (T, sims_number), fill_value = 0.0)
portfolio_value = 1000000


# Monte Carlo loop
for m in range(0, sims_number):
    Z = np.random.normal(size =(T, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    daily_returns = meanM + np.inner(L, Z)
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, daily_returns.T)+1)*portfolio_value



