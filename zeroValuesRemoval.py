from pandas import read_csv
import os
import numpy as np
import csv
from matplotlib import pyplot



basePath = "results/ARIMA/"
datasateBaseName = "ukdale_def"

destinationBasePath = "results/ARIMA2/"










def save_accuracy_to_csv(values,path,day,seriesName):

	if not os.path.isdir(path):
		try:
			os.mkdir(path)
		except OSError:
			print("Creation of the directory %s failed" % path)


	with open(path + "/" + "accuracy.csv", mode="a+") as csv_file:
		lines = csv_file.readlines()

		fieldnames = ['mape', 'corr', 'rmse', 'minmax', 'seriesName', 'days']
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)


		if os.stat(path + "/" + "accuracy.csv").st_size == 0:
			writer.writerow({'mape':values.get("mape"),'corr':values.get("corr"),'rmse':values.get("rmse"),'minmax':values.get("minmax"), 'seriesName':seriesName, 'days':str(int(day))})
		else:
			writer.writerow({'mape':values.get("mape"),'corr':values.get("corr"),'rmse':values.get("rmse"),'minmax':values.get("minmax"), 'seriesName':seriesName, 'days':str(int(day))})

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

def save_series_to_csv(path,series, fileName,day):

	if not os.path.isdir(path):
		try:
			os.mkdir(path)
		except OSError:
			print("Creation of the directory %s failed" % path)

	file = open(path + "/" + str(int(day)) + "days_" + fileName, "w+")
	file.write(series.to_csv(header=False))
	file.close()



def read_series(path):
	if os.path.exists(path + "/7days_predictions.csv") and os.path.exists(path + "/7days_test.csv") and os.path.exists(path + "/7days_train.csv"):
		predictions1 = read_csv(path + "/7days_predictions.csv",index_col=0,header=None)
		test1 = read_csv(path + "/7days_test.csv",index_col=0,header=None)
		train1 = read_csv(path + "/7days_train.csv",index_col=0,header=None)

	if os.path.exists(path + "/30days_predictions.csv") and os.path.exists(path + "/30days_test.csv") and os.path.exists(path + "/30days_train.csv"):
		predictions2 = read_csv(path + "/30days_predictions.csv",index_col=0,header=None)
		test2 = read_csv(path + "/30days_test.csv",index_col=0,header=None)
		train2 = read_csv(path + "/30days_train.csv",index_col=0,header=None)

	return (predictions1,test1,train1,predictions2,test2,train2)

#Save the plot from pyplot
def save_plot(path,seriesName,day):

	if not os.path.isdir(path):
		try:
			os.mkdir(path)
		except OSError:
			print("Creation of the directory %s failed" % path)

	finalPath = path + "/" + str(int(day)) + "days_plot.png"
	pyplot.savefig(finalPath, dpi=100)

def newPlot(train,test,predictions,seriesName,path,day):
	pyplot.figure(figsize=(12,5), dpi=100)
	pyplot.plot(train, color='blue',label="Train")
	pyplot.plot(test, color='blue',label="Test")
	pyplot.plot(predictions, color='red',label="Prediction")
	pyplot.title(seriesName + " " + str(int(day)) + " days trained")
	ax = pyplot.gca()
	ax.axes.xaxis.set_visible(False)
	ax.legend(loc="upper left")

	save_plot(path,seriesName,day)


def main():
	#Iterating for all the 
	for i in range(1,2):
		datasetName = datasateBaseName + str(i)

		#reading the header from the dataset
		series = read_csv("Dataset/" + datasetName + ".csv",header=0,index_col=0,nrows=1)
		seriesNames = list(series.columns.values)
		outPath = basePath + datasetName

		for serie in seriesNames:
			seriesPath = outPath + "/" + serie
			print(seriesPath)

			#reading the csv
			predictions7,test7,train7,predictions30,test30,train30 = read_series(seriesPath)

			#removing all the negative values
			predictions7[predictions7 < 0] = 0
			test7[test7 < 0] = 0
			predictions30[predictions30 < 0] = 0
			test30[test30 < 0] = 0

			newOutPath = destinationBasePath + datasetName
			newSeriesPath = newOutPath + "/" + serie

			#saving the new results
			save_series_to_csv(newSeriesPath,predictions7,"predictions.csv",7)
			save_series_to_csv(newSeriesPath,test7,"test.csv",7)
			save_series_to_csv(newSeriesPath,train7,"train.csv",7)
			save_series_to_csv(newSeriesPath,predictions30,"predictions.csv",30)
			save_series_to_csv(newSeriesPath,test30,"test.csv",30)
			save_series_to_csv(newSeriesPath,train30,"train.csv",30)

			#calculating the new accuracy
			acc7 = forecast_accuracy(predictions7.values,test7.values)
			acc30 = forecast_accuracy(predictions30.values,test30.values)

			save_accuracy_to_csv(acc7,newOutPath,7,serie)
			save_accuracy_to_csv(acc30,newOutPath,30,serie)

			#plot the new plot
			newPlot(train7,test7,predictions7,serie,newSeriesPath,7)
			newPlot(train30,test30,predictions30,serie,newSeriesPath,30)

			print(newSeriesPath + " done! \n")




if __name__ == "__main__":
	main()