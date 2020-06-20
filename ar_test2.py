# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 17:10:34 2020

@author: Nielsen
"""
from statsmodels.tsa.ar_model import AR
from random import random
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

dateparse = lambda dates: pd.datetime.strptime(dates,'%Y-%m')
data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col = 'Month', date_parser=dateparse)

model = AR(data['#Passengers'])

model_fit = model.fit()

y_pred = model_fit.predict(len(data),len(data))

data.plot(color='red', marker="o")
y_pred.plot(color='blue', marker="x")