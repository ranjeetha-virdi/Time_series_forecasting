## Multiple Time Series Forecasting using DeepAR for prediction of electricity consuption for different housholds.


### About DeepAR Algorithm:

DeepAR is multiple time series forecasting algorithm that uses available as open source package gluonts as well as AWS DeepAR ML Algorithm.
It is a supervised machine learning algorithm for time series forecasting that uses recurrent neural netwrok to produce both point in time as well as 
probabilistic forecast.
Point in time prediction means we build a model to say this is our forecast and probabilistic forecast is basically defining a confidence interval like 
for example how confident are we with the forecast say 95% confident or 80% confident, it gives a confidence band.
The best thing about DeepAR is that it can work to create a single global model when we have multiple time series. 
It creates a single global model on a large number of related time series and it has a builtin special treatment where time series magnitude varies across
each time series differently.

### About Libraries and Dataset:
We will first install mxnet to do deep learning for DeepAR and GluonTS a Python library for quick prototyping of Deep Learning models for Time Series applications.
We can install GluonTS both GPU or CPU version. Here we are using CPU version.
The data set is downloaded from UC Irvine Machine Learning Repo https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014. It is time series 
data from 2011 to 2014. The data has energy usage for 370 customers and each customer is a seperate time series.
The first column is date and the data is recorded every 15 mins. MT_001,MT_002...MT_370 are representation of individual household and readings for them.

#### Step 1: Importing libraries:  
     - Pandas to get the data into DataFrame 
     - Matplotlib to vizualize the dataset
     - deepAREstimator package of GluonTS for modelling
     - Trainer object to set the hyperparameters
#### Step 2: Load the data into a Pandas DataFrame and visualize the data for 10 households for only two weeks. The vizulisation is a shown below. Here we can see that each household 
has a different consumption pattern.

![data_head](https://github.com/ranjeetha-virdi/Time_series_forecasting/assets/81987445/7dab47b2-958b-4227-905e-3c198814a632)


![time_series_power](https://github.com/ranjeetha-virdi/Time_series_forecasting/assets/81987445/42eb4f54-e7ee-4040-9062-202450ecd6b1)

#### step 3: We will transpose the data so that MT_001 becomes the new index.


