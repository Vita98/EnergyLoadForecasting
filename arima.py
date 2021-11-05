from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib
import pandas as pd



numbersOfRowToRead = 11520
trainSize = 10080
testSize = 1440

originFileName = "ukdale_def4.csv"
seriesName = "Gas_Boiler"






def parser(x):
	return datetime.strptime(x, '%y-%m-%d %H:%M:%S')

series = read_csv(originFileName,header=0,index_col=0,nrows=numbersOfRowToRead)
print(series[seriesName].head())


X = series[seriesName]
train, test = X[0:trainSize], X[trainSize:trainSize+testSize]
history = [x for x in train]
predictions = list()


model = ARIMA(history, order=(5,0,1))
model_fit = model.fit(start_params=[0,0,0,0,0,0,0,1])

maxLen = len(test)



# walk-forward validation
for t in range(len(test)):

	perc = (100 / maxLen) * t
	print("\nPerc: " + str(perc))

	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)

	model_fit = model_fit.append([test[t]])
	print('predicted=%f, expected=%f' % (yhat, obs))


fc_series = pd.Series(predictions,index=test.index)



# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
pyplot.plot(train, color='blue')
pyplot.plot(test, color='blue')
pyplot.plot(fc_series, color='red')

ax = pyplot.gca()
ax.axes.xaxis.set_visible(False)
pyplot.show()
