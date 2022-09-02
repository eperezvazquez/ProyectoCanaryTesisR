import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import base64
import seaborn as sns
import matplotlib.pyplot as plt

st.title('📈Forecasting with prophet')

"""
About this data set 
The Standard and Poor's 500 or S&P 500 is the most famous financial benchmark in the world.

This stock market index tracks the performance of 500 large companies listed on stock exchanges in the United States. As of December 31, 2020, more than $5.4 trillion was invested in assets tied to the performance of this index.

Because the index includes multiple classes of stock of some constituent companies—for example, Alphabet's Class A (GOOGL) and Class C (GOOG)—there are actually 505 stocks in the gauge.

The S&P 500 is a stock market index that tracks the largest 500 publicly traded U.S. companies. Investors have long used the S&P 500 as a benchmark for their investments as it tends to signal overall market health. The S&P 500 is a “free-floating index” meaning that it only takes into consideration the health and price of shares that are publicly traded; it does not consider government-owned or privately-owned shares. The index is a popular choice for long-term investors who wish to watch growth over the coming decades.
"""

"""
### Step 1: Import Data
"""
data1 = pd.read_csv('assets/sp500_stocks.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data = data1.dropna()
data = data[data['Volume']>0]
data= data.reset_index()
data= data[data['Symbol'].isin(("AVGO","MPWR","LRCX"))]
data= data[['Date','Open']]
data= data.groupby('Date').sum('Open')
data= data.reset_index()
data = data.rename(columns={'Date': 'ds','Open': 'y'})
data['ds'] = pd.to_datetime(data['ds'],errors='coerce')
data2 = data.rename(columns={'ds': 'Date','y': 'Open_value'})

st.write(data2)
max_date = data['ds'].max()


sns.set_style('darkgrid')
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (20, 9)
plt.rcParams['figure.facecolor'] = '#00000000'
plt.rcParams['lines.linewidth'] = 2
broad = data1.query("Symbol == 'AVGO'")
mono = data1.query("Symbol == 'MPWR'")
inc = data1.query("Symbol == 'LRCX'")
broad['Close'].plot(label = "Broadcom Inc.")
mono['Close'].plot(label = 'Monolithic Power Systems')

inc['Close'].plot(label = 'Inc. MPWR Lam Research Corporation LRCX')

plt.title('Stock Prices Semiconduction')
plt.legend()
plt.show()


"""
### Step 2: Select Forecast Horizon
Keep in mind that forecasts become less accurate with larger forecast horizons.
"""

periods_input = st.number_input('How many periods would you like to forecast into the future?',
min_value = 1, max_value = 365)



"""
### Step 3: Visualize Forecast Data
The below visual shows future predicted values. "yhat" is the predicted value, and the upper and lower limits are (by default) 80% confidence intervals.
"""
m = Prophet(interval_width=0.95, weekly_seasonality=False, changepoint_prior_scale=0.9)
m.add_seasonality(name='yearly', period=365, fourier_order=8)
m.add_country_holidays(country_name='US')
m.fit(data)

future = m.make_future_dataframe(periods=periods_input)

forecast = m.predict(future)   
#forecast = m.predict(future)
fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

fcst_filtered =  fcst[fcst['ds'] > max_date] 
fcst_filtered = fcst_filtered.rename(columns={'ds': 'Date','yhat': 'Mean_predict', 'yhat_lower': 'Low_prediction', 'yhat_upper': 'High_prediction'})
st.write(fcst_filtered)
    
"""
The next visual shows the actual (black dots) and predicted (blue line) values over time.
"""

fig1 = m.plot(forecast)
st.write(fig1)

"""
* In 2014 they did a business with Intel for 650 million dollars. 
* An increase of Broadcom Inc. is denoted since in 2016 it merges with another company Avago from there the peak is observed. 
* And in August 2018 an agreement with CA tencoloy for 19 billion dollars, from there lies the peak of the rise.
* La pandemia en 2020 generó movimientos fuera de lo que se podía predecir. La demanda umentó e hizo que subiera el valor de estas empresas
* Por último, la guerra entre China y Taiwán afectó negativamente el valor ya que este último es el principal proveedor asiático.
"""

"""
The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.
"""
fig2 = m.plot_components(forecast)
st.write(fig2)


"""
### Step 4: Download the Forecast Data
The below link allows you to download the newly created forecast to your computer for further analysis and use.
"""

csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
st.markdown(href, unsafe_allow_html=True)