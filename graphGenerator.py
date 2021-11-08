from pandas import read_csv
from pandas import datetime
import enum
from matplotlib import pyplot
import os

class TrainignTimeType(enum.IntEnum):
	ONE_WEEK = 7
	ONE_MONTH = 30




'''
	PUT HERE THE CONFIGURATION VALUES
										'''
trainSize = TrainignTimeType.ONE_MONTH

originFileName = "ukdale_def5.csv"
seriesName = "Dishwasher"





#Parser for the read_csv
def parser(x):
	return datetime.strptime(x, '%y-%m-%d %H:%M:%S')

#Defining the path
finalPath = "result/" + originFileName.split(".")[0] + "/" + seriesName + "/"  

predictionPath = finalPath + str(int(trainSize)) + "days_predictions.csv"
testPath = finalPath + str(int(trainSize)) + "days_test.csv"
trainPath = finalPath + str(int(trainSize)) + "days_train.csv"

print("Reconstructing graph for home: \"" + originFileName + "\"")
print("Appliance: \""+seriesName+"\"\n")

pred = None
test = None
train = None

#Loading the series
if os.path.exists(predictionPath):
	pred = read_csv(str(predictionPath),header=0,index_col=0)
	print("Prediction file found!")
else:
	print("Prediction file not found!")

if os.path.exists(testPath):
	test = read_csv(str(testPath),header=0,index_col=0)
	print("Test file found!")
else:
	print("Test file not found!")

if os.path.exists(trainPath):
	train = read_csv(str(trainPath),header=0,index_col=0)
	print("Train file found!")
else:
	print("Train file not found!")


#Showing the graph with matplotlib
print("\nShowing the graph...\n")
pyplot.figure(figsize=(12,5), dpi=100)
if train is not None:
	pyplot.plot(train, color='blue')
if test is not None:
	pyplot.plot(test, color='blue')
if pred is not None:
	pyplot.plot(pred, color='red')
day = trainSize / 1440
pyplot.title(seriesName + " " + str(int(trainSize)) + " days trained")
ax = pyplot.gca()
ax.axes.xaxis.set_visible(False)
pyplot.show()
