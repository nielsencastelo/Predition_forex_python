# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:24:57 2019

@author: Nielsen
"""

# AR example
from statsmodels.tsa.ar_model import AR
from random import random
import matplotlib.pyplot as plt
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = AR(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))

plt.plot(data)