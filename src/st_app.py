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


st.title('📈Machine Learning y analítica aplicada al portafolio de proyectos - prueba')

"""
El objetivo general es poder generar un cuadro de mando integral, del área de Planificación estratégica en particular de los portafolios de AGESIC, donde permita tener indicadores a la hora de la toma de decisiones. 

"""

"""
### Step 1: Importing Data
"""

"""

"""

###

"""
### Step 2: Select Forecast Horizon
Keep in mind that forecasts become less accurate with larger forecast horizons.
"""



"""
### Step 3: Visualize Forecast Data
The below visual shows future predicted values. "Mean_predict" is the predicted value, and the upper and lower limits are (by default) 80% confidence intervals.
"""



"""
### Step 4: Download the Forecast Data
The below link allows you to download the newly created forecast to your computer for further analysis and use.
"""

#csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
#b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
##href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
##st.markdown(href, unsafe_allow_html=True)