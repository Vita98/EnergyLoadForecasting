from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
# from statsmodels.tsa.statespace.sarimax.SARIMAX import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import os
import enum
import numpy as np
import csv
import multiprocessing


class TrainignTimeType(enum.IntEnum):
    ONE_WEEK = 10080
    ONE_MONTH = 43200


class TestingTimeType(enum.IntEnum):
    ONE_DAY = 1440


# Save the time series given as parameter
def save_series_to_csv(series, fileName, seriesName):
    path = "results/ARIMA/" + "ukdale_def3"

    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)

    path = "results/ARIMA/" + "ukdale_def3" + "/" + seriesName

    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)

    day = trainSize / 1440
    file = open(path + "/" + str(int(day)) + "days_" + fileName, "w")
    file.write(series.to_csv(header=False))
    file.close()


#
def save_accuracy_to_csv(values, fileName, seriesName):
    path = "results/ARIMA/" + "ukdale_def3"

    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)

    day = trainSize / 1440
    # file = open(path + "/" + str(int(day)) + "days_" + fileName + "accuracy", "w")

    with open(path + "/" + fileName, mode="a+") as csv_file:
        lines = csv_file.readlines()

        fieldnames = ['mape', 'corr', 'rmse', 'minmax', 'seriesName', 'days']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if os.stat(path + "/" + fileName).st_size == 0:
            writer.writerow({'mape': values.get("mape"), 'corr': values.get("corr"), 'rmse': values.get("rmse"),
                             'minmax': values.get("minmax"), 'seriesName': seriesName, 'days': str(int(day))})
        else:
            writer.writerow({'mape': values.get("mape"), 'corr': values.get("corr"), 'rmse': values.get("rmse"),
                             'minmax': values.get("minmax"), 'seriesName': seriesName, 'days': str(int(day))})

    csv_file.close()


# Save the plot from pyplot
def save_plot(seriesName):
    path = "results/ARIMA/" + "ukdale_def3"

    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)

    path = "results/ARIMA/" + "ukdale_def3" + "/" + seriesName
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)

    day = trainSize / 1440
    finalPath = path + "/" + str(int(day)) + "days_plot.png"
    pyplot.savefig(finalPath, dpi=100)


# Parser for the read_csv
def parser(x):
    return datetime.strptime(x, '%y-%m-%d %H:%M:%S')


# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
    corr = np.corrcoef(forecast, actual)[0, 1]  # corr
    rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE
    mins = np.amin(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    minmax = 1 - np.mean(mins / maxs)  # minmax

    return ({'mape': mape,
             'corr': corr, 'rmse': rmse, 'minmax': minmax})


'''
	PUT HERE THE CONFIGURATION VALUES
										'''
trainSize = TrainignTimeType.ONE_WEEK
testSize = TestingTimeType.ONE_DAY
shiftRow = 26423

originFileName = "ukdale_def3.csv"
seriesName = "Gas_Boiler"


def main(seriesName):
    # main function

    trainSize = TrainignTimeType.ONE_WEEK
    testSize = TestingTimeType.ONE_DAY
    # Splitting the dataset into training and testing
    X = series[seriesName]
    train, test = X[0:trainSize], X[trainSize:trainSize + testSize]
    history = [x for x in train]
    predictions = list()



    maxLen = len(test)
 
	# Creating the ARIMA model
	# (5,2,1) start_params=[0,0,0,0,0,0,1,5]
	# (5,1,1) start_params=[0,0,0,0,0,0,1,3]
    print("\nTraining the model...\n")
    model = ARIMA(history, order=(5, 0, 1))
    model_fit = model.fit(start_params=[0, 0, 0, 0, 0, 0, 0, 1])
    maxLen = len(test)

    print("Testing...")
	# walk-forward validation
    for t in range(len(test)):
	    perc = (100 / maxLen) * t
	    print("\nPerc: %.2f%%" % perc, end="\r")
		#print("\033[A                             \033[A")

	    output = model_fit.forecast()
	    yhat = output[0]
	    predictions.append(yhat)
	    obs = test[t]
	    history.append(obs)

	    model_fit = model_fit.append([test[t]])


# print('predicted=%f, expected=%f' % (yhat, obs))

    """
	mod = sm.tsa.statespace.SARIMAX(df,
									order=(1, 0, 1),
									seasonal_order=(0, 0, 1, 12),
									enforce_stationarity=False,
									enforce_invertibility=False)
	"""


    print("Testing...")

    fc_series = pd.Series(predictions, index=test.index)
    fc_series[fc_series < 0] = 0

    # evaluate forecasts
    values = forecast_accuracy(fc_series.values, test.values)
    print(values)

    pyplot.figure(figsize=(12, 5), dpi=100)
    pyplot.plot(train, color='blue')
    pyplot.plot(test, color='blue')
    pyplot.plot(fc_series, color='red')
    day = trainSize / 1440
    pyplot.title(seriesName + " " + str(int(day)) + " days trained")
    ax = pyplot.gca()
    ax.axes.xaxis.set_visible(False)

    # saving date
    save_series_to_csv(train, "train.csv", seriesName)
    save_series_to_csv(test, "test.csv", seriesName)
    save_series_to_csv(fc_series, "predictions.csv", seriesName)
    save_accuracy_to_csv(values, "accuracy.csv", seriesName)
    save_plot(seriesName)
    # pyplot.show()

    print("\nAll done!\n")


if __name__ == '__main__':

    numbersOfRowToRead = int(trainSize) + int(testSize) + shiftRow

    # Reading the series from the dataset file
    series = read_csv("Dataset/" + originFileName, header=0, index_col=0, nrows=numbersOfRowToRead,
                      skiprows=range(1, shiftRow))
    # seriesNames = list(series.columns.values)

    # seriesNames = ['Speakers']
    appls = ["Kettle", "Electric_Heater", "Laptop", "Projector"]
    proc = []
    for appliance in appls:
        p = multiprocessing.Process(target=main, args=[appliance])
        p.start()
        proc.append(p)

    for procces in proc:
        procces.join()
