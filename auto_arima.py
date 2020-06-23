# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:02:29 2020

@author: Nielsen
"""

## pip install pmdarima

import pandas as pd
from matplotlib.pylab import rcParams
from pmdarima.arima import auto_arima
rcParams['figure.figsize'] = 15, 6

df = pd.read_csv('all-stocks-5yr.csv')

companies = df.Name.unique()

z = df.loc[df['Name'] == 'ZTS']
z = z.dropna()

dataset = z[['close']]

TRAIN_SIZE = 0.90
train_size = int(len(dataset) * TRAIN_SIZE)
test_size = len(dataset) - 

train = dataset.head(train_size)
test  = dataset.tail(test_size)

stepwise_model = auto_arima(train, start_p=1, max_p=6, m=20, suppress_warnings=True, stepwise=True)

stepwise_model.fit(train)

future_forecast = stepwise_model.predict(n_periods=test_size)

future_forecast = pd.DataFrame(future_forecast, index = test.index, columns=['prediction'])

result = pd.concat([future_forecast,test], axis=1)
result.plot()