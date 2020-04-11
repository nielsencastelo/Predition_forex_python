# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:15:31 2019

@author: Nielsen
"""
from MetaTrader5 import *
import pandas as pd
import matplotlib.pyplot as plt
# Initializing MT5 connection 
MT5Initialize()
MT5WaitForTerminal()

print(MT5TerminalInfo())
print(MT5Version())

# Copying data to pandas data frame
stockdata = pd.DataFrame()
rates = MT5CopyRatesFromPos("EURUSD", MT5_TIMEFRAME_M5, 0, 10)
# Deinitializing MT5 connection
MT5Shutdown()

stockdata['Open'] = [y.open for y in rates]
stockdata['Close'] = [y.close for y in rates]
stockdata['High'] = [y.high for y in rates]
stockdata['Low'] = [y.low for y in rates]
stockdata['Date'] = [y.time for y in rates]

#plt.plot(stockdata['Date'], stockdata['Close'])