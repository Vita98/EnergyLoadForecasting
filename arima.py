from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib
import pandas as pd



def parser(x):
	return datetime.strptime(x, '%y-%m-%d %H:%M:%S')


series = read_csv('Dataset/ukdale_def4.csv',header=0,index_col=0,nrows=11520)
print(series['Gas_Boiler'].head())


X = series['Gas_Boiler']
size = int(len(X) * 0.87)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()


model = ARIMA(history, order=(5,0,1))
model_fit = model.fit(start_params=[0,0,0,0,0,0,0,1])

maxLen = len(test)



# walk-forward validation
for t in range(len(test)):

	perc = (100 / maxLen) * t
	print(perc)
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






