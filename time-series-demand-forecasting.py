## TIME SERIES DEMAND FORECASTING
#------------------------------------------------------------------------------------------
# PART 1 - PREPARATION

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels as sms

import warnings
import itertools

from pylab import rcParams
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.style.use('fivethirtyeight')
rcParams['figure.figsize'] = 16, 8

sales = pd.read_csv('https://github.com/dwoo-work/time-series-demand-forecasting/blob/main/src/sales_data_sample_utf8.csv')
sales = sales.drop_duplicates()

sales_clean = sales.copy()
sales_clean.info()

sales_clean['ORDERDATE'] = pd.to_datetime(sales_clean['ORDERDATE'])

sales_clean['date'] = sales_clean['ORDERDATE'].dt.strftime("%Y-%m-%d")
sales_clean['date'] = pd.to_datetime(sales_clean['date'])

sales_clean['month'] = sales_clean.date.dt.month
sales_clean['year'] = sales_clean.date.dt.year
sales_clean['week'] = sales_clean.date.dt.week

sales_clean.PRODUCTLINE.unique()
sales_clean['motorcycles_QUANTITYORDERED'] = sales_clean.loc[sales_clean['PRODUCTLINE'] == 'Motorcycles', 'QUANTITYORDERED']

time_series = sales_clean.groupby(['week', 'month', 'year']).agg(date = ('date', 'first'), motorcycles_total_qty_ordered = ('motorcycles_QUANTITYORDERED', np.sum)).reset_index().sort_values('date')

time_series.info()
time_series['date'] = pd.to_datetime(time_series['date'])
time_series = time_series.set_index('date')

monthly_series = time_series.motorcycles_total_qty_ordered.resample('M').sum()

monthly_series.plot(label = 'actual').set(title = 'Total Qty. of Motorcycles Sold from Feb 2003 - May 2005')
plt.legend(loc = 'upper left')

#------------------------------------------------------------------------------------------
# PART 2 - DISSECT MONTHLY SERIES DATA INTO SEASONALITY, TREND, AND REMAINDER

components = sm.tsa.seasonal_decompose(monthly_series)
components.plot()

seasonality = components.seasonal
trend = components.trend
remainder = components.resid

#------------------------------------------------------------------------------------------
# PART 3 - PERFORM STATIONALITY TEST FOR MONTHLY SERIES DATA

monthly_series.plot(label = 'actual').set(title = 'Total Qty. of Motorcycles Sold (with mean and S.D from Feb 2003 - May 2005)')
monthly_series.rolling(window = 12).mean().plot(label = 'mean')
monthly_series.rolling(window = 12).std().plot(label = 's.d')
plt.legend(loc = 'upper left')

ad_fuller_test = sm.tsa.stattools.adfuller(monthly_series, autolag = 'AIC')
ad_fuller_test

#------------------------------------------------------------------------------------------
# PART 4 - [ARIMA MODEL] IDENTIFY WHICH TIME SERIES MODEL (MA, AR, ARMA, and ARIMA) IS MOST SUITABLE

plot_acf(monthly_series)
plot_pacf(monthly_series, lags = 13)

model_MA = sm.tsa.statespace.SARIMAX(monthly_series, order = (0, 0, 1))
model_AR = sm.tsa.statespace.SARIMAX(monthly_series, order = (1, 0, 0))
model_ARMA = sm.tsa.statespace.SARIMAX(monthly_series, order = (1, 0, 1))
model_ARIMA = sm.tsa.statespace.SARIMAX(monthly_series, order = (1, 1, 1))

result_MA = model_MA.fit()
result_AR = model_AR.fit()
result_ARMA = model_ARMA.fit()
result_ARIMA = model_ARIMA.fit()

result_MA.aic
result_AR.aic
result_ARMA.aic
result_ARIMA.aic

result_ARIMA.plot_diagnostics(figsize = [20, 16])

#------------------------------------------------------------------------------------------
# PART 5 - [ARIMA MODEL] PERFORM GRID SEARCH TO IDENTIFY THE BEST POSSIBLE COMBINATION

p = d = q = P = D = Q = range(0, 3)
S = 12

combinations = list(itertools.product(p, d, q, P, D, Q))
len(combinations)

arima_orders = [(x[0], x[1], x[2]) for x in combinations]
seasonal_orders = [(x[3], x[4], x[5], S) for x in combinations]

results_data = pd.DataFrame(columns = ['p', 'd', 'q', 'P', 'D', 'Q', 'AIC'])

for i in range(len(combinations)):
    try:
        model = sm.tsa.statespace.SARIMAX(monthly_series, order = arima_orders[i], seasonal_order = seasonal_orders[i])
        result= model.fit()
        results_data.loc[i,'p'] = arima_orders[i][0]
        results_data.loc[i,'d'] = arima_orders[i][1]
        results_data.loc[i,'q'] = arima_orders[i][2]
        results_data.loc[i,'P'] = seasonal_orders[i][0]
        results_data.loc[i,'D'] = seasonal_orders[i][1]
        results_data.loc[i,'Q'] = seasonal_orders[i][2]
        results_data.loc[i,'AIC'] = result.aic
    except:
        continue

results_data[results_data.AIC == min(results_data.AIC)]

#------------------------------------------------------------------------------------------
# PART 6 - [ARIMA MODEL] RUN THE AMIRA MODEL TO PERFORM THE FORECASTING

best_model = sm.tsa.statespace.SARIMAX(monthly_series, order = (2, 1, 0), seasonal_order = (0, 2, 0, 12))
results = best_model.fit()

monthly_series
fitting = results.get_prediction(start = '2003-01-31')
fitting_mean = fitting.predicted_mean

forecast = results.get_forecast(steps = 12)
forecast_mean = forecast.predicted_mean

fitting_mean.plot(label = 'fitting').set(title = 'Forecast Total Qty. of Motorcycles Sold from May 2005 - May 2006 (ARIMA Model)')
forecast_mean.plot(label = 'forecast')
monthly_series.plot(label = 'actual')
plt.legend(loc = 'upper left')

mean_absolute_error = abs(monthly_series - fitting_mean).mean()

#------------------------------------------------------------------------------------------
# PART 7 - [EXPONENTIAL SMOOTHING MODEL] RUN EXPONENTIAL SMOOTHING TO PERFORM THE FORECASTING

model_expo1 = sms.tsa.holtwinters.ExponentialSmoothing(monthly_series, trend = 'add', seasonal = 'add', seasonal_periods = 12)
model_expo2 = sms.tsa.holtwinters.ExponentialSmoothing(monthly_series, trend = 'mul', seasonal = 'add', seasonal_periods = 12)
model_expo3 = sms.tsa.holtwinters.ExponentialSmoothing(monthly_series, trend = 'add', seasonal = 'mul', seasonal_periods = 12)
model_expo4 = sms.tsa.holtwinters.ExponentialSmoothing(monthly_series, trend = 'mul', seasonal = 'mul', seasonal_periods = 12)

results_1 = model_expo1.fit()
results_2 = model_expo2.fit()
results_3 = model_expo3.fit()
results_4 = model_expo4.fit()

fit1 = model_expo1.fit().predict(0, len(monthly_series))
fit2 = model_expo2.fit().predict(0, len(monthly_series))
fit3 = model_expo3.fit().predict(0, len(monthly_series))
fit4 = model_expo4.fit().predict(0, len(monthly_series))

mae1 = abs(monthly_series - fit1).mean()
mae2 = abs(monthly_series - fit2).mean()
mae3 = abs(monthly_series - fit3).mean()
mae4 = abs(monthly_series - fit4).mean()

forecast = model_expo1.fit().predict(0, len(monthly_series) + 12)

monthly_series.plot(label = 'actual').set(title = 'Forecast Total Qty. of Motorcycles Sold from May 2005 - May 2006 (Exponential Smoothing Model)')
forecast.plot(label = 'forecast')
plt.legend(loc = 'upper left')