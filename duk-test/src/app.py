#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px


# In[2]:


#Import libraries
import pandas as pd
import requests
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
from datetime import date
from datetime import datetime
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import seaborn as sns
import pytz
import csv
from pathlib import Path
import statsmodels as sm                 
from statsmodels.tools.eval_measures import rmse
import warnings
from datetime import datetime
from datetime import timedelta
import xgboost as xgb
import pytz
from datetime import datetime
import csv
from datetime import timedelta
import plotly.express as px
warnings.filterwarnings("ignore")


# In[3]:


TEMPLATE = 'plotly_white'


# In[4]:


df_latest_36 = pd.read_csv('df_latest_36.csv')
df_latest_36['Datetime'] = pd.to_datetime(df_latest_36['Datetime'])
df_latest_36.set_index('Datetime',inplace=True)
annual_peak_2023 = 18119


# In[5]:


duk_annual_peak = pd.read_csv('duk_annual_peak.csv')
duk_annual_peak['Datetime'] = pd.to_datetime(duk_annual_peak['Datetime'])
# duk_annual_peak.set_index('Datetime',inplace = True)
duk_annual_peak


# In[6]:


# Use Plotly to show final results

import plotly.graph_objects as go
from plotly.subplots import make_subplots



fig = make_subplots(specs=[[{"secondary_y": True}]])



fig.add_trace(go.Scatter(x=df_latest_36.index, y=df_latest_36.demand, name='demand',yaxis="y3",
                         line = dict(color='royalblue', width=4)))

fig.add_trace(go.Scatter(x=df_latest_36.index, y=df_latest_36.prediction, name='prediction',yaxis="y3",
                         line = dict(color='lime', width=4, dash='dot')))

fig.add_trace(go.Scatter(x=df_latest_36.index, y=df_latest_36.prediction_EIA,name='prediction_EIA',yaxis="y3",
                         line=dict(color='mediumpurple', width=4,dash='dot')))


fig.add_trace(go.Scatter(x=df_latest_36.index, y=df_latest_36.actual_temp,name='actual_temp',yaxis="y1",
                         line=dict(color='rosybrown', width=4)))


fig.add_trace(go.Scatter(x=df_latest_36.index, y=df_latest_36.forecasted_temp,name='forecasted_temp',yaxis='y1',
                         line=dict(color='red', width=4)))


fig.add_hline(y = 70, line_dash="dot",line_color='black',line_width=0,
              annotation_text="2023 annual demand peak", 
              annotation_position="bottom right")

fig.add_shape(type="line",
    x0=df_latest_36.index[0], y0=annual_peak_2023, x1=df_latest_36.index[-1], y1=annual_peak_2023,
    line=dict(color="black",width=3,dash="dot",),yref="y3"
)


fig.update_layout(

    yaxis3=dict(
        title="DUK demand (megawatthours)",
        overlaying="y",
        side="left",
        range=[8000,20000],
    ),
    yaxis1=dict(
        title="Temperature (°F)",
        side="right",
        range=[0,84]
    ),
    title=str(date.today()) + '-' + str(pd.Timestamp.now().hour) + 'h' ' DUK electricity demand prediction',
    template=TEMPLATE
     

)

fig.show()


# In[7]:


description = 'Duke Energy Carolinas is a subsidiary of Duke Energy, one of the largest electric power holding companies in the United States.Duke Energy Carolinas serves approximately 2.6 million customers in North Carolina and South Carolina.The company provides electric service to residential, commercial, and industrial customers,as well as wholesale customers such as municipalities and electric cooperatives.'


# In[8]:


model_description='The DUK forecasting model was trained on historical load and weather data    from 2015/7-2023/2. Weather readings were from VisualCrossing.'


# In[9]:


app = dash.Dash(
    external_stylesheets=[dbc.themes.LUX],
    suppress_callback_exceptions=True
)
server = app.server


# In[10]:


fig1 = px.bar(duk_annual_peak, x='DUK_MW', y=duk_annual_peak.index,
              hover_data = ['Datetime'],orientation = 'h'
             )
fig1.update_layout(template=TEMPLATE)
fig1.show()


# In[11]:


duk_annual_peak


# In[12]:


t0 = str(date.today())


# In[13]:


# get the latest updated historic weather data of Mecklenburg County
base = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/history?&aggregateHours=1&'
time = 'startDateTime='+t0+'&endDateTime='+t0+'T24:00:00&'
unit = 'unitGroup=us&contentType=csv&dayStartTime=0:0:00&dayEndTime=0:0:00&'
location = 'location=MecklenburgCounty,NC,US&'
api = 'key=D5FAK4DB3LVUJGWCTFZRXA2T6'
url = base + time + unit + location + api
dfM = pd.read_csv(url)
dfM = dfM[['Date time','Address','Temperature']]
dfM = dfM.set_index('Date time')
dfM


# In[14]:


url = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/spartanburg,sc?unitGroup=us&include=days&key=D5FAK4DB3LVUJGWCTFZRXA2T6&contentType=csv'


# In[15]:


dfM = pd.read_csv(url)
dfM


# In[16]:


dfM = dfM[['datetime','name','tempmax','tempmin']]
dfM = dfM.set_index('datetime')
dfM


# In[17]:


city_list = ['GuilfordCounty,NC', 'GreenvilleCounty,SC', 'ForsythCounty,NC', 'SpartanburgCounty,SC', 'DurhamCounty,NC', 'YorkCounty,SC', 'IndianTrail,NC', 'GastonCounty,NC',
              'CabarrusCounty,NC', 'AndersonCounty,SC']

for i in city_list:
    url = url = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'+i+'?unitGroup=us&include=days&key=D5FAK4DB3LVUJGWCTFZRXA2T6&contentType=csv'
    df0 = pd.read_csv(url)
    df0 = df0[['datetime','name','tempmax','tempmin']]
    df0 = df0.set_index('datetime')
    dfM = pd.concat([dfM,df0],axis=1)
    dfM.dropna(inplace = True)


# In[18]:


dfM


# In[19]:


Tem15d = dfM[['tempmax','tempmin']]
Tem15d['temp_max']= Tem15d.tempmax.mean(axis=1)
Tem15d['temp_min']= Tem15d.tempmin.mean(axis=1)
Tem15d = Tem15d[['temp_max','temp_min']]
Tem15d


# In[20]:


fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(go.Scatter(x=Tem15d.index, y=Tem15d.temp_max,name='temp_max',yaxis="y1",
                         line=dict(color='red', width=4)))


fig2.add_trace(go.Scatter(x=Tem15d.index, y=Tem15d.temp_min,name='temp_min',yaxis='y1',
                         line=dict(color='rosybrown', width=4)))

fig2.update_layout(template=TEMPLATE)

fig2.show()


# In[21]:


app.layout = html.Div([
        html.Div(id='duk-content'),
        html.Br(),
#         dbc.Row([
#             dbc.Col(
#                 html.Div(l.BUTTON_LAYOUT), width=4),
#             dbc.Col(width=7),
#         ], justify='center'),
#     html.Br(),
#         html.Br(),
        dbc.Row([
            dbc.Col(html.H1('Duke Energy Carolinas (DUK)'), width=9),
            dbc.Col(width=2),
        ], justify='center'),
    dbc.Row([
            dbc.Col(
            html.Div(children=description), width=9),
            dbc.Col(width=2)
        ], justify='center'),
    html.Br(),
        dbc.Row([
            dbc.Col(
                html.H3('Model Performance'), width=9
            ),
            dbc.Col(width=2),
        ], justify='center'),
#         dbc.Row([
#             dbc.Col(
#                 html.Div(
#                     children='Mean Absolute Error (MAE)'
#                 ), width=9
#             ),
#             dbc.Col(width=2),
#         ], justify='center'),
#      html.Br(),
#         dbc.Row([
#             dbc.Col(
#                     dcc.Dropdown(
#                         id='duk-dropdown',
#                         options=[
#                             {'label': 'Actual', 'value': 'Actual'},
#                             {'label': 'Predicted', 'value': 'Predicted'}
#                         ],
#                         value=['Actual', 'Predicted'],
#                         multi=True,
#                     ), width=6
#             ),
#             dbc.Col(width=5),
#         ], justify='center'),
    dcc.Graph(id='duk-graph',
             figure=fig),
        html.Br(),
        html.Br(),
        dbc.Row([
            dbc.Col(html.H3('Training Data'), width=9),
            dbc.Col(width=2)
        ], justify='center'),
        dbc.Row([
            dbc.Col(
                    html.Div(children=model_description), width=9
            ),
            dbc.Col(width=2)
        ], justify='center'),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.Div([
                        dcc.Graph(
                            figure=fig1
                        ),
                    ]), width=4),
                dbc.Col(
                    html.Div([
                        dcc.Graph(
                            figure=fig2
                        ),]), width=4),
])                
])
        


# In[22]:


if __name__ == '__main__':
    app.run_server(debug = True,use_reloader=False)

