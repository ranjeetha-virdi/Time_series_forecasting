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

