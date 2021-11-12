from pandas import read_csv
import os
import numpy as np
import csv




basePath = "results/ARIMA/"
datasateBaseName = "ukdale_def"








def save_accuracy_to_csv(values,path,day):

	if not os.path.isdir(path):
		try:
			os.mkdir(path)
		except OSError:
			print("Creation of the directory %s failed" % path)


	with open(path + "/" + "accuracy.csv", mode="a+") as csv_file:
		lines = csv_file.readlines()

		fieldnames = ['mape', 'corr', 'rmse', 'minmax', 'days']
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)


		if os.stat(path + "/" + "accuracy.csv").st_size == 0:
			writer.writerow({'mape':values.get("mape"),'corr':values.get("corr"),'rmse':values.get("rmse"),'minmax':values.get("minmax"), 'days':str(int(day))})
		else:
			writer.writerow({'mape':values.get("mape"),'corr':values.get("corr"),'rmse':values.get("rmse"),'minmax':values.get("minmax"), 'days':str(int(day))})

# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax

    return({'mape':mape, 
            'corr':corr, 'rmse':rmse,'minmax':minmax})


def calc_loss(path):
	#print(path + "/7days_predictions.csv")
	if os.path.exists(path + "/7days_predictions.csv") and os.path.exists(path + "/7days_test.csv"):
		predictions = read_csv(path + "/7days_predictions.csv",index_col=0,header=None)
		test = read_csv(path + "/7days_test.csv",index_col=0,header=None)

		acc = forecast_accuracy(predictions.values,test.values)
		save_accuracy_to_csv(acc,path,7)

	if os.path.exists(path + "/30days_predictions.csv") and os.path.exists(path + "/30days_test.csv"):
		predictions = read_csv(path + "/30days_predictions.csv",index_col=0,header=None)
		test = read_csv(path + "/30days_test.csv",index_col=0,header=None)

		acc = forecast_accuracy(predictions.values,test.values)
		save_accuracy_to_csv(acc,path,30)


def main():

	#Iterating for all the 
	for i in range(1,6):
		datasetName = datasateBaseName + str(i)

		#reading the header from the dataset
		series = read_csv("Dataset/" + datasetName + ".csv",header=0,index_col=0,nrows=1)
		seriesNames = list(series.columns.values)

		for serie in seriesNames:
			seriesPath = basePath + datasetName + "/" + serie
			print(seriesPath)
			calc_loss(seriesPath)


if __name__ == "__main__":
	main()
