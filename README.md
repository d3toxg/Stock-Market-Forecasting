# Stock-Market-Forecasting

What is a Stock market?
The stock market is a market that enables the seamless exchange of buying and selling of company stocks. Every Stock Exchange has its own Stock Index value. The index is the average value that is calculated by combining several stocks. This helps in representing the entire stock market and predicting the market’s movement over time. The stock market can have a huge impact on people and the country’s economy as a whole. Therefore, predicting the stock trends in an efficient manner can minimize the risk of loss and maximize profit.

![19333414](https://user-images.githubusercontent.com/41402706/210825054-1ec493bf-7f0b-48d4-9460-6b330d064231.jpg)

<h1> Here in this notebook we will forecast the stock price of ARCH CAPITAL GROUP using ARIMA model </h1>

<strong>What is ARIMA?</strong>
Autoregressive Integrated Moving Average (ARIMA) Model converts non-stationary data to stationary data before working on it. It is one of the most popular models to predict linear time series data.

ARIMA model has been used extensively in the field of finance and economics as it is known to be robust, efficient and has a strong potential for short-term share market prediction.



# %% [markdown]
# ### Load all the required libraries
​
# %% [code] {"_kg_hide-output":true,"execution":{"iopub.status.busy":"2023-01-05T17:04:56.282195Z","iopub.execute_input":"2023-01-05T17:04:56.282941Z"}}
!pip install pmdarima
​
# %% [code] {"_kg_hide-output":true}
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
​
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
​
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
​
# %% [code]
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
stock_data = pd.read_csv('../input/price-volume-data-for-all-us-stocks-etfs/Stocks/acgl.us.txt',sep=',', index_col='Date', parse_dates=['Date'], date_parser=dateparse).fillna(0)
​
​
# %% [code]
stock_data
​
# %% [markdown]
# Visualize the per day closing price of the stock.
​
# %% [code]
#plot close price
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Close Prices')
plt.plot(stock_data['Close'])
plt.title('ARCH CAPITAL GROUP closing price')
plt.show()
​
# %% [markdown]
# ### We can also visualize the data in our series through a probability distribution too.
​
# %% [code]
#Distribution of the dataset
df_close.plot(kind='kde')
​
# %% [markdown]
# Also, a given time series is thought to consist of three systematic components including level, trend, seasonality, and one non-systematic component called noise.
# 
# These components are defined as follows:
# 
# 1. **Level**: The average value in the series.
# 
# 2. **Trend**: The increasing or decreasing value in the series.
# 
# 3. **Seasonality**: The repeating short-term cycle in the series.
# 
