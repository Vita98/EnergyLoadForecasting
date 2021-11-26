# Time series forecasting with ARIMA and variants
Energy load forcasting project based on ARIMA statistical model and variants, developed for University exam

## General info
This project use [UKDALE](https://jack-kelly.com/data/ "UKDALE website") dataset. You can visualize our predictions in 
[results](https://github.com/Vita98/MMSAProject/tree/main/results "our results") folder, there is 5 folders each on that rappresent one house and containts data in csv
format and graphs. In particolar there are results with 7 days traing and 1 day prediction, 30 days traing and 1 day prediction for each household appliance.

In alternative you can run the code to get results, but in some case you might get different results because we have tuning ARIMA parameters for some household appliance 
to get better results.

## How to use
First of all clone this repository `git clone https://github.com/Vita98/EnergyLoadForecasting.git` to make sure there are no missing folders
### Arima
For arima you can use [arima.py](https://github.com/Vita98/EnergyLoadForecasting/blob/main/arima.py), [arimaNotebook.ipynb](https://github.com/Vita98/EnergyLoadForecasting/blob/main/arimaNotebook.ipynb) or [arimaAutomator.py](https://github.com/Vita98/EnergyLoadForecasting/blob/main/arimaAutomator.py). You can modify the configuration in the appropriate block in the code and in main block
### Sarima
For sarima you can use [sarima.py](https://github.com/Vita98/EnergyLoadForecasting/blob/main/sarima.py), [sarimaAutomator.py](https://github.com/Vita98/EnergyLoadForecasting/blob/main/sarimaAutomator.py) or [sarimaNotebook.ipynb](https://github.com/Vita98/EnergyLoadForecasting/blob/main/sarimaNotebook.ipynb). You can modify the configuration in the appropriate block in the code and in main block

### Sarimax
For sarimax you can use [sarimaxAutomator.py](https://github.com/Vita98/EnergyLoadForecasting/blob/main/sarimaxAutomator.py) or [sarimaxMulti.py](https://github.com/Vita98/EnergyLoadForecasting/blob/main/sarimaxMulti.py). You can modify the configuration in the appropriate block in the code and in main block

## Authors 
* Alessio Tartarelli aka [a-tartarelli](https://github.com/a-tartarelli "a-tartarelli's profile")
* Vitandrea Sorino aka [Vita98](https://github.com/Vita98 "Vita98's profile")
