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
import altair as alt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


st.title('📈Presupuesto')

"""
El objetivo de este streamlit ese poder generar un modelo de presupuesto que permite evaluar si un proyecto termina en, fuera o debajo de presupuesto
"""

"""
### Step 1: Importing Data
"""

"""
Importan datos de SIGES
Importan datos de wekan
Se une en ambas BD
Debajo es una prueba con datos de Kaggle para saber si funciona
"""
data1 = pd.read_csv('src\sp500_stocks.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data = data1.dropna()
data = data[data['Volume']>0]
data= data.reset_index()
data3= data[data['Symbol'].isin(("AVGO","MPWR","LRCX"))]
data3 = data3.reset_index()
data= data3[['Date','Open']]
data= data.groupby('Date').sum('Open')
data1= data.reset_index()
data = data1.rename(columns={'Date': 'ds','Open': 'y'})
data['ds'] = pd.to_datetime(data['ds'],errors='coerce')
data2 = data.rename(columns={'ds': 'Date','y': 'Open_value'})

st.write(data2)
max_date = data['ds'].max()

#####
def get_data():
    source = data3
    source = source[source.Date.gt("2004-01-01")]
    return source

source = get_data()

def get_chart(data3):
    hover = alt.selection_single(
        fields=["Date"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data3, title="Evolution of stock prices")
        .mark_line()
        .encode(
            x="Date",
            y="Open",
            color="Symbol",
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data3)
        .mark_rule()
        .encode(
            x="yearmonthdate(Date)",
            y="Open",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("Date", title="Date"),
                alt.Tooltip("Open", title="Price (USD)"),
            ],
        )
        .add_selection(hover)
    )
    return (lines + points + tooltips).interactive()

chart = get_chart(source)

# Add annotations
ANNOTATIONS = [
    ("Sep 01, 2014", "Broadcom Inc. (AVGO) did a business with Intel for 650 million dollars."),
    ("Jan 01, 2016", "An increase of Broadcom Inc. (AVGO) is denoted since in 2016 it merges with another company Avago from there the peak is observed."),
    ("Aug 01, 2018", "An agreement with CA tencoloy for 19 billion dollars, from there lies the peak of the rise."),
    ("Aug 01, 2022", "The conflict between China and Taiwan has negatively affected the value of the latter, being the main Asian supplier."),
]

annotations_df = pd.DataFrame(ANNOTATIONS, columns=["Date", "event"])
annotations_df.Date = pd.to_datetime(annotations_df.Date)
annotations_df["y"] = 10

annotation_layer = (
    alt.Chart(annotations_df)
    .mark_text(size=20, text="⬇", dx=-8, dy=-10, align="left")
    .encode(
        x="Date:T",
        y=alt.Y("y:Q"),
        tooltip=["event"],
    )
    .interactive()
)

st.altair_chart(
    (chart + annotation_layer).interactive(),
    use_container_width=True
)
#####

#####
def get_data2():
    source2 = data1
    source2 = source2[source2.Date.gt("2004-01-01")]
    return source2

source2 = get_data()

def get_chart2(data1):
    hover = alt.selection_single(
        fields=["Date"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data1, title="Evolution of stock sumarized")
        .mark_line()
        .encode(
            x="Date",
            y="Open",
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data1)
        .mark_rule()
        .encode(
            x="yearmonthdate(Date)",
            y="Open",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("Date", title="Date"),
                alt.Tooltip("Open", title="Price (USD)"),
            ],
        )
        .add_selection(hover)
    )
    return (lines + points + tooltips).interactive()

chart2 = get_chart2(source2)

# Add annotations

st.altair_chart(
    (chart2).interactive(),
    use_container_width=True
)

###

"""
### Step 2: Select Forecast Horizon
Keep in mind that forecasts become less accurate with larger forecast horizons.
"""

periods_input = st.number_input('How many periods would you like to forecast into the future?',
min_value = 1, max_value = 365)



"""
### Step 3: Visualize Forecast Data
The below visual shows future predicted values. "Mean_predict" is the predicted value, and the upper and lower limits are (by default) 80% confidence intervals.
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
The next few visuals show a high level trend of values, day of week trends, and yearly trends (if dataset covers multiple years).
"""

fig2 = m.plot_components(forecast)
st.write(fig2)  

"""
The next visual shows the reals (black dots) and predicted (blue line) values over time. The blue shaded area represents upper and lower confidence intervals.
"""

fig1 = m.plot(forecast)
st.write(fig1)



metric_df = forecast.set_index('ds')[['yhat']].join(data.set_index('ds').y).reset_index()
metric_df.dropna(inplace=True)

"""
The r-squared is close to 1 so we can conclude that the fit is good. 
An r-squared value over 0.9 is amazing (and probably too good to be true, which tells me this data is most likely overfit).
The value obtained is:
"""
st.write(r2_score(metric_df.y, metric_df.yhat))
"""
MSE:
"""
st.write(mean_squared_error(metric_df.y, metric_df.yhat))
"""
That's a large MSE value... and confirms my suspicion that this data is overfit and won't likely hold up well into the future. Remember... for MSE, closer to zero is better.
And finally, MAE result:
"""
st.write(mean_absolute_error(metric_df.y, metric_df.yhat))



"""
In summary, both in the application of Time Series at the level of the General Index with all the companies and at the level of the 3 selected companies that are AVGO Broadcom, MPWR Monolithic Power Systems Inc, LRCX Lam Research Corporation Lam Research, in both cases we find that time series seasonal manages to better predict the model with a confidence interval of 95% and weekly_seasonality=False and changepoint_prior_scale=0.9.
For the next cases, we will make predictions for each companies, also using a multivariable model. For example, models MLP Multivariant (https://www.youtube.com/watch?v=87c9D_41GWg)
"""

"""
### Step 4: Download the Forecast Data
The below link allows you to download the newly created forecast to your computer for further analysis and use.
"""

csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
st.markdown(href, unsafe_allow_html=True)