from pandas import read_csv
import os
import numpy as np
import csv
from matplotlib import pyplot
import time




basePath = "results/ARIMASTD/"
datasateBaseName = "ukdale_def"





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
	pyplot.close()

def newPlot(train,test,predictions,seriesName,path,day):
	pyplot.figure(figsize=(12,5), dpi=100)
	pyplot.plot(train, color='blue',label="Train")
	pyplot.plot(test, color='orange',label="Test")
	pyplot.plot(predictions, color='red',label="Prediction")
	pyplot.title(seriesName + " " + str(int(day)) + " days trained")
	ax = pyplot.gca()
	ax.axes.xaxis.set_visible(False)
	ax.legend(loc="upper left")

	save_plot(path,seriesName,day)

def main():
	#Iterating for all the 
	for i in range(1,6):

		datasetName = datasateBaseName + str(i)
		#datasetName = datasateBaseName

		#reading the header from the dataset
		series = read_csv("Dataset/" + datasetName + ".csv",header=0,index_col=0,nrows=1)
		#seriesNames = list(series.columns.values)
		seriesNames = ["total"]
		outPath = basePath + datasetName

		for serie in seriesNames:
			seriesPath = outPath + "/" + serie
			print(seriesPath)

			#reading the csv
			predictions7,test7,train7,predictions30,test30,train30 = read_series(seriesPath)

			#plot the new plot
			newPlot(train7,test7,predictions7,serie,seriesPath,7)
			end = newPlot(train30,test30,predictions30,serie,seriesPath,30)
			print(seriesPath + " done! \n")

if __name__ == "__main__":
	main()