# Time Series Demand Forecasting

This will demonstrate to you how to perform time series demand forecasting using Python.

The purpose of this is to analyse time series data with the use of statistics, and identify the most suitable time series or exponential smoothing model, to perform demand forecasting for the future.

For this script, we will use the past data of motorcycles quantities sold, to perform a demand forecasting for the next 12 months.

In general, there are two commonly used forecasting models:

- Auto Regressive Integrated Moving Average (ARIMA): This takes past observations, to gain information for the next observation.
- Exponential Smoothing: This takes the average of the past forecasts, to gain information for the next observation.

To decide on which model to use, you would have to perform the AD Fuller (ADF) Test to understand the stationality of the time series data. If the P-value is:

- Stationary data (P < 0.05): Perform the ARIMA model.
- Non-stationary data (P >= 0.05): Perform the Exponential Smoothing model.

After running a model, there is a need to ascertain the accuracy of it. This can be done by four different methods:

- Mean Square Error (MSE): This magnifies the difference, between the actual and the forecast.
- Mean Absolute Error (MAE): This identifies the absolute difference, between the actual and the forecast.
- Mean Error (ME): This identifies the biasness of the forecast, to see if there is a pattern of overshooting or undershooting.
- Root Mean Square Error (RMSE): This identifies the standard deviation of the residuals.

## ARIMA model

For this, there are four different variations for the ARIMA model, which are:

- Moving Average (MA):  This incorporates past errors of the series.
- Auto Regressive (AR): This incorporates past values of the series.
- Auto Regressive Moving Average (ARMA): This incorporates both the past errors and values of the series.
- Auto Regressive Integrated Moving Average (ARIMA): This incorporates both the past errors and values of the series, with an extra 'differencing' component that acts as a lag.

To identify the most suitable ARIMA model to use, you would have to identify each of their Akaike Information Criterion (AIC) value. The lowest has the best balance between model's complexity and errors, and hence shall be used.

After fitting either one of the four variations for the ARIMA model, you can enhance the fit by performing a Grid Search:

![ARIMA Formula](https://github.com/dwoo-work/time-series-forecasting/blob/main/img/ARIMA_Formula.png)

For each ARIMA model, it contains 2 groups, and 6 different variables:

- p: Non-seasonal portion, auto regression parameter
- d: Non-seasonal portion, integration parameter
- q: Non-seasonal portion, moving average parameter
- P: Seasonal portion, auto regression parameter
- D: Seasonal portion, integration parameter
- Q: Seasonal portion, moving average parameter

What a grid search does, is essentially finding out every grouping possible, by using a pre-determined range of value for each variable (usually 0 to 2). With that, the AIC value will computed for every grouping possible.

## Exponential Smoothing model

For this, there are three different variations for the Exponential Smoothing model, which are:

- Single Exponential Smoothing: This incorporates smoothing factor for the level.
- Double Exponential Smoothing (Holt's Method): This incorporates smoothing factor for the level, and the trend.
- Triple Exponential Smoothing (Holt Winter's Method): This incorporates smoothing factor for the level, the trend, and the seasonality.

For the smoothing factor for trend, there are two different types, which are:

- Additive: Seasonal variation remains constant throughout the series.
- Multiplicative: Seasonal variation changes throughout the series.

For this, there are four different variations for the Holt Winter's model, which are:

- Additive Trend + Additive Seasonality
- Multiplicative Trend + Additive Seasonality
- Additive Trend + Multiplicative Seasonality
- Multiplicative Trend + Multiplicative Seasonality

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install:

- pandas: to perform data administration by pulling in .csv or .xlsx files.
- numpy: to perform data manipulation by establishing arrays.
- statsmodels: to provide clases and functions for estimation of different statistical models.
- matplotlib.pyplot: to create static, animated, and interactive visualisations.

```bash
pip install pandas
pip install numpy
pip install statsmodels
pip install matplotlib
```

## Sample Dataset

For this, you can download the sales_data_sample_utf8.csv file from the source folder, which is located [here](https://github.com/dwoo-work/time-series-forecasting/tree/main/src).

Ensure that the file is in CSV UTF-8 format, to avoid UnicodeDecodeError later on.

## Code Explanation

### Part 1 - Preparation

Lines 5-12:  
Import the required libraries.
```python   
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels as sms

import warnings
import itertools
```

Lines 14-15:  
Import the specific function from Pylab abd Statsmodels.
```python   
from pylab import rcParams
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
```

Lines 17-18:  
Adjust the style and size of the plots later on.
```python   
plt.style.use('fivethirtyeight')
rcParams['figure.figsize'] = 16, 8
```

Lines 20-21:  
Import and clean the CSV dataframe.
```python   
sales = pd.read_csv('https://github.com/dwoo-work/time-series-demand-forecasting/blob/main/src/sales_data_sample_utf8.csv')
sales = sales.drop_duplicates()
```

Lines 23-24:  
Create a new CSV dataframe (cleaned data).
```python   
sales_clean = sales.copy()
sales_clean.info()
```

Lines 26:  
Change ORDERDATE from object to datetime.
```python   
sales_clean['ORDERDATE'] = pd.to_datetime(sales_clean['ORDERDATE'])
```

Lines 28-29:  
Create a date column within the sales_clean dataframe.
```python   
sales_clean['date'] = sales_clean['ORDERDATE'].dt.strftime("%Y-%m-%d")
sales_clean['date'] = pd.to_datetime(sales_clean['date'])
```

Lines 31-33:  
Create different columns for week, month, and year within the sales_clean dataframe.
```python   
sales_clean['month'] = sales_clean.date.dt.month
sales_clean['year'] = sales_clean.date.dt.year
sales_clean['week'] = sales_clean.date.dt.week
```

Lines 35-36:  
Create a column for motorcycles quantity ordered.
```python   
sales_clean.PRODUCTLINE.unique()
sales_clean['motorcycles_QUANTITYORDERED'] = sales_clean.loc[sales_clean['PRODUCTLINE'] == 'Motorcycles', 'QUANTITYORDERED']
```

Lines 38:  
Create a variable for time series, and plot a line chart using it.
```python   
time_series = sales_clean.groupby(['week', 'month', 'year']).agg(date = ('date', 'first'), motorcycles_total_qty_ordered = ('motorcycles_QUANTITYORDERED', np.sum)).reset_index().sort_values('date')
```

Lines 40-42:  
Index the date in time_series dataframe.
```python   
time_series.info()
time_series['date'] = pd.to_datetime(time_series['date'])
time_series = time_series.set_index('date')
```

Lines 44-47:  
Create a variable for motorcycles total quantity ordered in a monthly series, and create a line chart for it.
```python   
monthly_series = time_series.motorcycles_total_qty_ordered.resample('M').sum()

monthly_series.plot(label = 'actual').set(title = 'Total Qty. of Motorcycles Sold from Feb 2003 - May 2005')
plt.legend(loc = 'upper left')
```

![Plot 1](https://github.com/dwoo-work/time-series-demand-forecasting/blob/main/plots/plot1.png)

### Part 2 - Dissect monthly series data into seasonality, trend, and remainder

Lines 52-57:  
Use the decomposition function to dissect seasonality and trend from the time series data.
```python   
components = sm.tsa.seasonal_decompose(monthly_series)
components.plot()

seasonality = components.seasonal
trend = components.trend
remainder = components.resid
```

![Plot 2](https://github.com/dwoo-work/time-series-demand-forecasting/blob/main/plots/plot2.png)

### Part 3 - Perform stationality test for monthly series data

Lines 62-65:  
Plot the monthly series chart with the actual data, mean, and the standard deviation.
```python   
monthly_series.plot(label = 'actual').set(title = 'Total Qty. of Motorcycles Sold (with mean and S.D from Feb 2003 - May 2005)')
monthly_series.rolling(window = 12).mean().plot(label = 'mean')
monthly_series.rolling(window = 12).std().plot(label = 's.d')
plt.legend(loc = 'upper left')
```

![Plot 3](https://github.com/dwoo-work/time-series-demand-forecasting/blob/main/plots/plot3.png)

Lines 67-68:  
Run the Augmented Dickey-Fuller (ADF) test to confirm stationality. The P-Value is 0.000005783924. Therefore, reject null hypothesis, and confirm that the data is stationary. Therefore, use ARIMA model to compute.
```python   
ad_fuller_test = sm.tsa.stattools.adfuller(monthly_series, autolag = 'AIC')
ad_fuller_test
```

### Part 4 - [ARIMA Model] Identify which time series model (MA, AR, ARMA, and ARIMA) is the most suitable

Lines 73-74:  
Plot the autocorrelation function (ACF) and the partial autocorrelation function (PACF).
```python   
plot_acf(monthly_series)
plot_pacf(monthly_series, lags = 13)
```

![Plot 4](https://github.com/dwoo-work/time-series-demand-forecasting/blob/main/plots/plot4.png)
![Plot 5](https://github.com/dwoo-work/time-series-demand-forecasting/blob/main/plots/plot5.png)

Lines 76-79:  
Prepare all 4 types of ARIMA models (MA, AR, ARMA, and ARIMA).
```python   
model_MA = sm.tsa.statespace.SARIMAX(monthly_series, order = (0, 0, 1))
model_AR = sm.tsa.statespace.SARIMAX(monthly_series, order = (1, 0, 0))
model_ARMA = sm.tsa.statespace.SARIMAX(monthly_series, order = (1, 0, 1))
model_ARIMA = sm.tsa.statespace.SARIMAX(monthly_series, order = (1, 1, 1))
```

Lines 81-84:  
Fit all 4 types of ARIMA models (MA, AR, ARMA, and ARIMA).
```python   
result_MA = model_MA.fit()
result_AR = model_AR.fit()
result_ARMA = model_ARMA.fit()
result_ARIMA = model_ARIMA.fit()
```

Lines 86-89:  
Perform Akaike Information Criterion (AIC) Analysis on all 4 types of ARIMA models (MA, AR, ARMA, and ARIMA). The ARIMA model has the lowest AIC value of 406.908. Therefore, it shall be used for later's analysis.
```python   
result_MA.aic
result_AR.aic
result_ARMA.aic
result_ARIMA.aic
```

Lines 91:  
Run diagnostics for the ARIMA model.
```python   
result_ARIMA.plot_diagnostics(figsize = [20, 16])
```

![Plot 6](https://github.com/dwoo-work/time-series-demand-forecasting/blob/main/plots/plot6.png)

### Part 5 - [ARIMA Model] Perform grid search to identify the best possible combination

Lines 96-97:  
Set a pre-determined range of values for p, d, q, P, D, and Q.
```python   
p = d = q = P = D = Q = range(0, 3)
S = 12
```

Lines 99-100:  
Create a variable to store all the possible combinations.
```python   
combinations = list(itertools.product(p, d, q, P, D, Q))
len(combinations)
```

Lines 102-103:  
Identify all possible non-seasonal and seasonal portion orders.
```python   
arima_orders = [(x[0], x[1], x[2]) for x in combinations]
seasonal_orders = [(x[3], x[4], x[5], S) for x in combinations]
```

Lines 105:  
Save the output of the models in a dataframe.
```python   
results_data = pd.DataFrame(columns = ['p', 'd', 'q', 'P', 'D', 'Q', 'AIC'])
```

Lines 107-119:  
Create a function to automatically compute all the combination's AIC, and create an error handling mechanism. Running this will take about 1-2 minutes, so please wait for a while.
```python   
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
```

Lines 121:  
Identify the combinations with the lowest AIC. For this dataset, the one with the lowest AIC is combination #180. (p, d, q = 2, 1, 0), (P, D, Q = 0, 2, 0), (AIC = 6.0)
```python   
results_data[results_data.AIC == min(results_data.AIC)]
```

### Part 6 - [ARIMA Model] Run the ARIMA model to perform the forecasting

Lines 127:  
Use the best combination (no. 573) value, to create and fit the best forecasting model.
```python   
best_model = sm.tsa.statespace.SARIMAX(monthly_series, order = (2, 1, 0), seasonal_order = (0, 2, 0, 12))
results = best_model.fit()
```

Lines 129-131:  
Define fitting model's timeframe (from when the monthly series begin), and identify its fitting value.
```python   
monthly_series
fitting = results.get_prediction(start = '2003-01-31')
fitting_mean = fitting.predicted_mean
```

Lines 133-134:  
Define forecast model's extended prediction (12 months), and identify its forecast value.
```python   
forecast = results.get_forecast(steps = 12)
forecast_mean = forecast.predicted_mean
```

Lines 136-139:  
Create a plot with fitting line, forecast line, and actual line.
```python   
fitting_mean.plot(label = 'fitting').set(title = 'Forecast Total Qty. of Motorcycles Sold from May 2005 - May 2006 (ARIMA Model)')
forecast_mean.plot(label = 'forecast')
monthly_series.plot(label = 'actual')
plt.legend(loc = 'upper left')
```

![Plot 7](https://github.com/dwoo-work/time-series-demand-forecasting/blob/main/plots/plot7.png)

Lines 141:  
Measure the accuracy of the model using the Mean Absolute Error.
```python   
mean_absolute_error = abs(monthly_series - fitting_mean).mean()
```

### Part 7 - [Exponential Smoothing Model] Run Exponential Smoothing to perform the forecasting

Lines 146-149:  
Prepare all 4 types of Holt Winter's Exponential Smoothing models. For motorcycles quantity ordered, can only run for the model with both trend and seasonal being additive.
```python   
model_expo1 = sms.tsa.holtwinters.ExponentialSmoothing(monthly_series, trend = 'add', seasonal = 'add', seasonal_periods = 12)
model_expo2 = sms.tsa.holtwinters.ExponentialSmoothing(monthly_series, trend = 'mul', seasonal = 'add', seasonal_periods = 12)
model_expo3 = sms.tsa.holtwinters.ExponentialSmoothing(monthly_series, trend = 'add', seasonal = 'mul', seasonal_periods = 12)
model_expo4 = sms.tsa.holtwinters.ExponentialSmoothing(monthly_series, trend = 'mul', seasonal = 'mul', seasonal_periods = 12)
```

Lines 151-159:  
Fit all 4 types of Holt Winter's Exponential Smoothing models. Since only model 1 can be performed, only can fit for this.
```python   
results_1 = model_expo1.fit()
results_2 = model_expo2.fit()
results_3 = model_expo3.fit()
results_4 = model_expo4.fit()

fit1 = model_expo1.fit().predict(0, len(monthly_series))
fit2 = model_expo2.fit().predict(0, len(monthly_series))
fit3 = model_expo3.fit().predict(0, len(monthly_series))
fit4 = model_expo4.fit().predict(0, len(monthly_series))
```

Lines 161-164:  
Measure the accuracy of the 4 exponential smoothing models. The lowest MAE is the best. mae1: 121.06331375972876, the rest are N.A.
```python   
mae1 = abs(monthly_series - fit1).mean()
mae2 = abs(monthly_series - fit2).mean()
mae3 = abs(monthly_series - fit3).mean()
mae4 = abs(monthly_series - fit4).mean()
```

Lines 166:  
Use the exponential smoothing model with the lowest MAE (model 1) to perform the forecasting.
```python   
forecast = model_expo1.fit().predict(0, len(monthly_series) + 12)
```

Lines 168-170:  
Create a plot with forecast line, and actual line.
```python   
monthly_series.plot(label = 'actual').set(title = 'Forecast Total Qty. of Motorcycles Sold from May 2005 - May 2006 (Exponential Smoothing Model)')
forecast.plot(label = 'forecast')
plt.legend(loc = 'upper left')
```

![Plot 8](https://github.com/dwoo-work/time-series-demand-forecasting/blob/main/plots/plot8.png)

## Credit

Sales Data Sample (https://www.kaggle.com/datasets/kyanyoga/sample-sales-data)

## License

[MIT](https://choosealicense.com/licenses/mit/)
