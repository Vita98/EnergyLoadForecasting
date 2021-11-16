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


class TrainignTimeType(enum.IntEnum):
    ONE_WEEK = 10080
    ONE_MONTH = 43200


class TestingTimeType(enum.IntEnum):
    ONE_DAY = 1440


# Save the time series given as parameter
def save_series_to_csv(series, fileName, seriesName, algorithmType):
    try:
        if algorithmType in "arima sarima sarimax arimastd":
            path = "results/" + algorithmType.upper() + "/" + originFileName[:-4] + "/total"

            if not os.path.isdir(path):
                try:
                    os.mkdir(path)
                except OSError:
                    print("Creation of the directory %s failed" % path)

            day = trainSize / 1440
            file = open(path + "/" + str(int(day)) + "days_" + fileName, "w")
            file.write(series.to_csv(header=False))
            file.close()

    except ValueError:
        print("unsupported algorithm")
        return


#
def save_accuracy_to_csv(values, fileName, seriesName, algorithmType):
    try:
        if algorithmType in "arima sarima sarimax arimastd":
            path = "results/" + algorithmType.upper() + "/" + originFileName[:-4] + "/total"

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

    except ValueError:
        print("unsupported algorithm")
        return


# Save the plot from pyplot
def save_plot(seriesName, algorithmType):
    try:
        if algorithmType in "arima sarima sarimax arimastd":
            path = "results/" + algorithmType.upper() + "/" + originFileName[:-4] + "/total"

            if not os.path.isdir(path):
                try:
                    os.mkdir(path)
                except OSError:
                    print("Creation of the directory %s failed" % path)

            day = trainSize / 1440
            finalPath = path + "/" + str(int(day)) + "days_plot_total.png"
            pyplot.savefig(finalPath, dpi=100)

    except ValueError:
        print("unsupported algorithm")
        return


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


def main(algorithmType):
    X = series["total"]
    train, test = X[0:trainSize], X[trainSize:trainSize + testSize]
    history = [x for x in train]
    predictions = list()

    if algorithmType == "arima":

        # ARIMA
        model = ARIMA(history, order=(5, 0, 1))
        model_fit = model.fit(start_params=[0, 0, 0, 0, 0, 0, 0, 1])
        maxLen = len(test)

        print("Testing...")
        # walk-forward validation
        for t in range(len(test)):
            perc = (100 / maxLen) * t
            print("\nPerc: %.2f%%" % perc, end="\r")

            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)

            model_fit = model_fit.append([test[t]])

    # print('predicted=%f, expected=%f' % (yhat, obs))
    elif algorithmType == "arimastd":

        print("\nTraining the model...\n")
        model = ARIMA(train, order=(5,0,1))
        model_fit = model.fit()
        

        yhat = model_fit.predict(start=0, end=len(test))
        #print(yhat)
        predictions = list()

        for value in yhat[1:]:
            predictions.append(value)



    elif algorithmType == "sarima":
        # creating SARIMA model
        my_order = (1, 0, 1)
        my_seasonal_order = (0, 0, 1, 12)
        # define model

        model = sm.tsa.statespace.SARIMAX(train, order=my_order, seasonal_order=my_seasonal_order,
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)
        model_fit = model.fit()

        yhat = model_fit.predict(start=0, end=len(test))

        for value in yhat[1:]:
            predictions.append(value)


    elif algorithmType == "sarimax":

        # SARIMAX
        # exo
        exo = series.drop(labels="total", axis=1)
        exoTrain, exoTest = exo[0:trainSize], exo[trainSize:trainSize + testSize]

        print("\nTraining the model...\n")

        # creating SARIMAX model
        my_order = (1, 0, 1)
        my_seasonal_order = (0, 0, 1, 12)

        # define model
        model = sm.tsa.statespace.SARIMAX(train, order=my_order, seasonal_order=my_seasonal_order,
                                          enforce_stationarity=False,
                                          enforce_invertibility=False, exog=exoTrain)

        model_fit = model.fit()

        print(model_fit.summary())

        yhat = model_fit.forecast(steps=len(test), exog=exoTest)

        for value in yhat:
            predictions.append(value)


    else:
        raise ValueError("Unsupported algorithm, only support arima, sarima and sarimax")

    # plot forecasts against actual outcomes

    print("Testing...")

    fc_series = pd.Series(predictions, index=test.index)
    fc_series[fc_series < 0] = 0

    # evaluate forecasts SARIMA

    values = forecast_accuracy(fc_series.values, test.values)
    print(values)

    pyplot.figure(figsize=(12, 5), dpi=100)
    pyplot.plot(train, color='blue')
    pyplot.plot(test, color='blue')
    pyplot.plot(fc_series, color='red')
    day = trainSize / 1440
    pyplot.title("Total" + str(int(day)) + " days trained")
    ax = pyplot.gca()
    ax.axes.xaxis.set_visible(False)

    # saving date
    seriesName = "total"

    save_series_to_csv(train, "train.csv", seriesName, algorithmType=algorithmType)
    save_series_to_csv(test, "test.csv", seriesName, algorithmType=algorithmType)
    save_series_to_csv(fc_series, "predictions.csv", seriesName, algorithmType=algorithmType)
    save_accuracy_to_csv(values, "accuracy.csv", seriesName, algorithmType=algorithmType)
    save_plot("Total", algorithmType=algorithmType)
    # pyplot.show()

    print("\nAll done!\n")


'''
	PUT HERE THE CONFIGURATION VALUES
										'''
trainSize = TrainignTimeType.ONE_MONTH
testSize = TestingTimeType.ONE_DAY
  # 561

# originFileName = "ukdale_def2.csv"
# seriesName = "Cooker"

if __name__ == '__main__':

    houses = ["ukdale_def1.csv", "ukdale_def2.csv", "ukdale_def3.csv", "ukdale_def4.csv", "ukdale_def5.csv"]
    #houses = ["ukdale_def4.csv"]
    #algorithms = ["arima", "sarima", "sarimax"]
    algorithms = ["arimastd"]

    # Reading the series from the dataset file

    for alg in algorithms:
        for house in houses:
            if house == "ukdale_def1.csv":
                shiftRow = 200000
            elif  house == "ukdale_def2.csv":
                shiftRow = 150000
            elif house == "ukdale_def3.csv":
                shiftRow = 26423
            else:
                shiftRow = 1

            numbersOfRowToRead = int(trainSize) + int(testSize) + shiftRow

            series = read_csv("Dataset/" + house, header=0, index_col=0, nrows=numbersOfRowToRead,
                              skiprows=range(1, shiftRow))
            originFileName = house
            seriesNames = list(series.columns.values)
            series["total"] = series.sum(axis=1)
            main(algorithmType=alg)
