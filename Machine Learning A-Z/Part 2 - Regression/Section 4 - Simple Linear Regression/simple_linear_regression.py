# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 15:36:54 2018
Simple Linear Regression
@author: apple
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the data set
dataset = pd.read_csv('Salary_Data.csv')

# create matrix of features & dependent variable vector
X = dataset.iloc[:, :-1].values    # years of experience
y = dataset.iloc[:, 1].values      # salary

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_predicted = regressor.predict(X_test)

# Visualizing the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs. Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test set results
plt.scatter(X_test, y_test, color = 'orange')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs. Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_predicted)
r2score = r2_score(y_test, y_predicted)

coef = regressor.coef_
inter = regressor.intercept_