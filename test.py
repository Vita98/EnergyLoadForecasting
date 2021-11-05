from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
 
def parser(x):
	ts = int(x)
	return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
 
fields = ['Date', 'Power']
series = read_csv('ukdale/house_5/channel_1.dat',sep='\s+', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
print(series.head())
#series.plot()
#pyplot.show()

print(series.keys()[0])
print(series.values)
results = adfuller(series)
print(results[0])
print(results[1])