import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
rcParams['figure.figsize'] = 15, 6

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)

fit1 = ExponentialSmoothing(data, seasonal_periods = 12, trend='additive', seasonal='additive').fit(use_boxcox=True)

fit1.fittedvalues.plot(style='--', color='red')
fit1.forecast(24).plot(style='--', marker="o", color='blue', legend=True)
