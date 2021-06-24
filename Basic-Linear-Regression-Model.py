#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn import linear_model 
from sklearn.metrics import mean_squared_error, r2_score

#Reading data from the csv file

data = pd.read_csv("lrdata.csv")
data.head()

data.rename(columns = {'32.502345269453031':'A', '31.70700584656992':'B'}, inplace = True)


#Converting Pandas DataFrame to Numpy Array

data_x = data.A.to_numpy()
data_y = data.B.to_numpy()

#Spliting data into training and test sets

train_data_x = np.transpose(np.atleast_2d(data_x[:-30]))
test_data_x = np.transpose(np.atleast_2d(data_x[-30:]))

train_data_y = np.transpose(np.atleast_2d(data_y[:-30]))
test_data_y = np.transpose(np.atleast_2d(data_y[-30:]))


#Creating a linear regression object

regr = linear_model.LinearRegression()

#Training the model 

regr.fit(train_data_x, train_data_y)

#Making Predictions on the test dataset

predictions = regr.predict(test_data_x)

#Estimated coefficients for the linear regression problem

print('Coefficients: \n', regr.coef_)


#Calculating the mean squared error

print('Mean squared error: %.2f'
      % mean_squared_error(test_data_y, predictions))

#Coefficient of determination / $R^{2}$ error
# 
# The best possible score is 1, which is obtained when the predicted values are the same as the actual values.
# A constant model that always predicts the expected value of y, disregarding the input features, would get a score of 0.0. It can be negative in case the model is arbitrarily worse.

print('Coefficient of determination: %.2f'
      % r2_score(test_data_y, predictions))


#Visualizing the fit

plt.scatter(test_data_x, test_data_y)
plt.plot(test_data_x, predictions, color='red')
plt.xlabel("A")
plt.ylabel("B")
plt.show()

