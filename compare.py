from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle
import pandas as pd

train = read_csv("train.csv",header=0,index_col=0)
test = read_csv("test.csv",header=0,index_col=0)
predictions = read_csv("predictions.csv",header=0,index_col=0)

plt.plot(train)
plt.plot(test)
plt.plot(predictions)
plt.title('Forecast vs Actuals')
plt.show()
#plt.legend(loc='upper left', fontsize=8)