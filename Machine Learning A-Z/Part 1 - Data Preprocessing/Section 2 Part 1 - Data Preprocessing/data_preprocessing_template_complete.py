# -*- coding: utf-8 -*-

"""
Created on Fri Jun 29 15:36:06 2018

Data Preprocessing

@author: apple
"""
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the data set
dataset = pd.read_csv('Data.csv')

# create matrix of features & dependent variable vector
X = dataset.iloc[:, :-1].values    #rows[0:x], col[0:y-1]
y = dataset.iloc[:, 3].values      #rows[0:x], col[y]

# Taking care of missing data - optional
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data - optional
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_country = LabelEncoder()
X[:, 0] = labelencoder_country.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_purchased = LabelEncoder()
y = labelencoder_purchased.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling - optional
from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
X_train = X_sc.fit_transform(X_train)
X_test = X_sc.transform(X_test)