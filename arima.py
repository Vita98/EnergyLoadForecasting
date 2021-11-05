from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import os

# 1440 = 1 day, 10080 = 7 days, 43200 = 30 days
numbersOfRowToRead = 11520
trainSize = 10080
testSize = 1440

originFileName = "ukdale_def4.csv"
seriesName = "Gas_Boiler"



def save_series_to_csv(series, fileName):
	path = "result/" + originFileName[:-4]
	try:
		os.mkdir(path)
	except OSError:
		print("Creation of the directory %s failed" % path)

	path = "result/" + originFileName[:-4] + "/" + seriesName
	try:
		os.mkdir(path)
	except OSError:
		print("Creation of the directory %s failed" % path)

	day = trainSize / 1440
	file = open(path + "/" + str(int(day)) + "Days" + fileName, "w")
	file.write(series.to_csv())
	file.close()



def parser(x):
	return datetime.strptime(x, '%y-%m-%d %H:%M:%S')



series = read_csv("Dataset/" + originFileName,header=0,index_col=0,nrows=numbersOfRowToRead)
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
	print("\nPerc: %.2f" %perc)

	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)

	model_fit = model_fit.append([test[t]])
	print('predicted=%f, expected=%f' % (yhat, obs))

#add time index to predictions
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

#saving date
save_series_to_csv(train, "train.csv")
save_series_to_csv(test, "test.csv")
save_series_to_csv(fc_series, "predictions.csv")

#show graph
pyplot.show()
