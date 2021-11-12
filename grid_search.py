# grid search sarima hyperparameters
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from datetime import datetime
import numpy as np
import pandas as pd
 
# one-step sarima forecast
def sarima_forecast(history, config):
	order, sorder = config
	# define model
	model = SARIMAX(history, order=order, seasonal_order=sorder)
	# fit model
	model_fit = model.fit(disp=False)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]
 
# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))
 
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]
 
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	train, test = train_test_split(data, n_test)
	history = [x for x in train]
	order, sorder = cfg
	# define model
	model = SARIMAX(history, order=order, seasonal_order=sorder)
	model_fit = model.fit(disp=False)
	# plot forecasts against actual outcomes
	yhat = model_fit.predict(start=0, end=len(test))
	# print(yhat)
	predictions = list()

	for value in yhat[1:]:
		predictions.append(value)

	return measure_rmse(test, predictions)
	''' 
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = sarima_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions) '''
	#return error
 
# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
	result = None
	# convert config to a key
	key = str(cfg)
	# show all warnings and fail on exception if debugging
	if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# check for an interesting result
	if result is not None:
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)
 
# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores
 
# create a set of sarima configs to try
def sarima_configs(seasonal=[150,180,210,240,270,300,360,720]):
	#
	#seasonal = [12]
	models = list()

	'''
	# define config lists
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]'''
	m_params = seasonal

	for m in m_params:
		cfg = [(1, 0, 1), (0, 0, 1, m)]
		models.append(cfg)
	'''
	# create config instances
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)'''
	return models

#Parser for the read_csv
def parser(x):
	return datetime.strptime(x, '%y-%m-%d %H:%M:%S')





import enum
class TrainignTimeType(enum.IntEnum):
	ONE_WEEK = 10080
	ONE_MONTH = 43200

class TestingTimeType(enum.IntEnum):
	ONE_DAY = 1440







'''
	PUT HERE THE CONFIGURATION VALUES
										'''
trainSize = TrainignTimeType.ONE_WEEK
testSize = TestingTimeType.ONE_DAY
shiftRow = 1

originFileName = "ukdale_def4.csv"
seriesName = "Tv_Dvd_Lamp"






 
if __name__ == '__main__':
	# define dataset

	numbersOfRowToRead = int(trainSize) + int(testSize) + shiftRow

	#Reading the series from the dataset file
	data = read_csv("Dataset/" + originFileName,header=0,index_col=0,nrows=numbersOfRowToRead,skiprows=range(1,shiftRow))

	data = data[seriesName]

	

	# data split
	n_test = int(testSize)
	# model configs
	cfg_list = sarima_configs()
	# grid search
	scores = grid_search(data, cfg_list, n_test)
	print('done')
	# list top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)