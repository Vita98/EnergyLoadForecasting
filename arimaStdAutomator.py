from multiprocessing import process
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import os
import enum
import numpy as np
import csv
import concurrent.futures
import multiprocessing

class TrainignTimeType(enum.IntEnum):
	ONE_WEEK = 10080
	ONE_MONTH = 43200

class TestingTimeType(enum.IntEnum):
	ONE_DAY = 1440


	#Save the time series given as parameter 
def save_series_to_csv(series, fileName, seriesName):
	path = "results/ARIMAstd/" + originFileName[:-4]

	if not os.path.isdir(path):
		try:
			os.mkdir(path)
		except OSError:
			print("Creation of the directory %s failed" % path)

	path = "results/ARIMAstd/" + originFileName[:-4] + "/" + seriesName

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
	path = "results/ARIMAstd/" + originFileName[:-4]

	if not os.path.isdir(path):
		try:
			os.mkdir(path)
		except OSError:
			print("Creation of the directory %s failed" % path)

	day = trainSize / 1440
	#file = open(path + "/" + str(int(day)) + "days_" + fileName + "accuracy", "w")

	with open(path + "/" + fileName, mode="a+") as csv_file:
		lines = csv_file.readlines()

		fieldnames = ['mape', 'corr', 'rmse', 'minmax', 'seriesName', 'days']
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)


		if os.stat(path + "/" + fileName).st_size == 0:
			writer.writerow({'mape':values.get("mape"),'corr':values.get("corr"),'rmse':values.get("rmse"),'minmax':values.get("minmax"), 'seriesName':seriesName, 'days':str(int(day))})
		else:
			writer.writerow({'mape':values.get("mape"),'corr':values.get("corr"),'rmse':values.get("rmse"),'minmax':values.get("minmax"), 'seriesName':seriesName, 'days':str(int(day))})

		
				
		
	csv_file.close()

#Save the plot from pyplot
def save_plot(seriesName):
	path = "results/ARIMAstd/" + originFileName[:-4]

	if not os.path.isdir(path):
		try:
			os.mkdir(path)
		except OSError:
			print("Creation of the directory %s failed" % path)

	path = "results/ARIMAstd/" + originFileName[:-4] + "/" + seriesName
	if not os.path.isdir(path):
		try:
			os.mkdir(path)
		except OSError:
			print("Creation of the directory %s failed" % path)

	day = trainSize / 1440
	finalPath = path + "/" + str(int(day)) + "days_plot.png"
	pyplot.savefig(finalPath, dpi=100)

#Parser for the read_csv
def parser(x):
	return datetime.strptime(x, '%y-%m-%d %H:%M:%S')

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



def main(seriesName):
	#main function

	#Splitting the dataset into training and testing 
	X = series[seriesName]
	train, test = X[0:trainSize], X[trainSize:trainSize+testSize]
	#history = [x for x in train]
	predictions = list()

	maxLen = len(test)

	#Creating the ARIMA model
	#(5,2,1) start_params=[0,0,0,0,0,0,1,5]
	#(5,1,1) start_params=[0,0,0,0,0,0,1,3]
	print("\nTraining the model...\n")
	model = ARIMA(train, order=(5,0,1))
	model_fit = model.fit()

	print("Testing...")
	
	predictions = list()
	for i in range(0,len(test)):
		perc = (100 / maxLen) * i
		print("\nPerc: %.2f%%" %perc,end="\r")
		print ("\033[A                             \033[A")
		yhat = model_fit.predict(start=i, end=i+1)
		predictions.append(yhat[0])
		model_fit = model_fit.append([yhat[0]])


	#print(model_fit.summary())

	''' 
	# plot forecasts against actual outcomes
	yhat = model_fit.forecast(steps=len(test), exog=exoTest)
	predictions = list()
	'''

	


	fc_series = pd.Series(predictions,index=test.index)
	fc_series[fc_series < 0] = 0

	# evaluate forecasts
	values = forecast_accuracy(fc_series.values, test.values)
	print(values)


	pyplot.figure(figsize=(12,5), dpi=100)
	pyplot.plot(train, color='blue',label="Train")
	pyplot.plot(test, color='orange',label="Test")
	pyplot.plot(fc_series, color='red',label="Prediction")
	day = trainSize / 1440
	pyplot.title(seriesName + " " + str(int(day)) + " days trained")
	ax = pyplot.gca()
	ax.axes.xaxis.set_visible(False)
	ax.legend(loc="upper left")


	#saving date
	save_series_to_csv(train, "train.csv",seriesName)
	save_series_to_csv(test, "test.csv",seriesName)
	save_series_to_csv(fc_series, "predictions.csv",seriesName)
	save_accuracy_to_csv(values, "accuracy.csv", seriesName)
	save_plot(seriesName)
	#pyplot.show()

	print("\nAll done!\n")



'''
	PUT HERE THE CONFIGURATION VALUES
										'''
trainSize = TrainignTimeType.ONE_WEEK
testSize = TestingTimeType.ONE_DAY
shiftRow = 1

#originFileName = "ukdale_def1.csv"
#seriesName = "Tv_Dvd_Lamp"


if __name__ == '__main__':
    #houses = ["ukdale_def1.csv", "ukdale_def2.csv", "ukdale_def3.csv", "ukdale_def4.csv", "ukdale_def5.csv"]
	houses = ["ukdale_def4.csv"]

	appliances = {
		"ukdale_def1.csv" : ["Boiler","Solar Termal Pump","Laptop","Washing Machine","Dishwasher","TV","Kitchen Lights","Htpc","Kettle","Toaster","Fridge","Microwave","LCD_Office","HIFI_Office","BreadMaker","Amp_Living","ADSL_Router","Living_lamp1","Soldering_Iron","GigE_USBHUB","Hoover","Kitchen Lamp1","Bedroom_Lamp","Lightning_Circuit","Living_lamp2","iPad_Charger","Subwoofer_L","Living_LampTV","DAB_radio","Kitchen Lamp2","Kitchen Phone_Stereo","UtilityRM_Lamp","Samsung_Charger","Bedroom_d_Lamp","Coffee_Machine","Kitchen_Radio","Bedroom_Chargers","Hair_Dryer","Straighteners","Iron","Gas_Oven","Datalogger_PC","Childs_table_lamp","Childs_ds_lamp","BabyMonitor","Battery_Charger","Office_Lamp1","Office_Lamp2","Office_Lamp3","Office_PC","Office_Fan","Led_Printer"],

		"ukdale_def2.csv" : ["Laptop","Monitor","Speakers","Server","Router","Server HDD","Kettle","Rice Cooker","Running Machine","Laptop 2","Washing Machine","Dishwasher","Fridge","Microwave","Toaster","PlayStation","Modem","Cooker"],

		"ukdale_def3.csv" : ["Kettle","Electric_Heater","Laptop","Projector"],

		"ukdale_def4.csv" : ["Tv_Dvd_Lamp"],#,"Kettle_Radio","Gas_Boiler","Freezer","Washing_Machine_Microwave_breadmaker"],

		"ukdale_def5.csv" : ["Stereo_Speakers_Bedroom","Pc Desktop","Hair Dryer","Primary TV","TV Bed","Treadmill","Network_attached_storage","Core2_server","24_inch_lcd TV","PS4","Steam_iron","Nespresso_Pixie","Atom_Pc","Toaster","Home_Theatre_Amp","Sky_Hd_Box","Kettle","Fridge_Freezer","Oven","Electric_Hob","Dishwasher","Microwave","Washer_dryer","Vacuum_Cleaner"]
		}

	# Reading the series from the dataset file

	for house in houses:
		if house == "ukdale_def1.csv":
			shiftRow = 200000
		elif  house == "ukdale_def2.csv":
			shiftRow = 150000
		elif house == "ukdale_def3.csv":
			shiftRow = 26423
		else:
			shiftRow = 1

		originFileName = house
		numbersOfRowToRead = int(trainSize) + int(testSize) + int(shiftRow)
		
		series = read_csv("Dataset/" + house, header=0, index_col=0,nrows=numbersOfRowToRead, skiprows=range(1, shiftRow))
		originFileName = house

		
		proc = []
		for appliance in appliances.get(house):
			p = multiprocessing.Process(target=main, args=[appliance])
			p.start()
			proc.append(p)

		for procces in proc:
			process.join()
		


