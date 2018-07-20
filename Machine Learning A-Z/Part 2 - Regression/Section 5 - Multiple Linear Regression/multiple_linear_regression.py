# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:13:38 2018
Multiple Linear Regression
@author: apple
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the data set
dataset = pd.read_csv('50_Startups.csv')

# create matrix of features & dependent variable vector
X = dataset.iloc[:, :-1].values    #rows[0:x], col[0:y-1]
y = dataset.iloc[:, 4].values      #rows[0:x], col[y]

# Encoding categorical data - optional
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
# Python library for linear regression is taking care of the DVT
# No need to execute this code. Just for reference because for some libraries,
# taking away 1 variable manually is necessary
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
#import statsmodels.regression.linear_model as sm
# Add column of 1 to the X feature. This corresponds to the X0 feature.
# This is being done in LinearRegression library but not in statsmodels
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

# matrix containing optimal variables
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# BE step 1: select a significance level (SL = 0.05)
# BE step 2
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# BE step 3
regressor_OLS.summary()

# BE step4
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# BE step4
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# BE step4
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# BE step4
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
