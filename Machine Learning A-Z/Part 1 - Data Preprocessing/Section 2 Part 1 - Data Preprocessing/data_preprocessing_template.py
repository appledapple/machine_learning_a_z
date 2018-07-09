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

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling - optional
'''from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
X_train = X_sc.fit_transform(X_train)
X_test = X_sc.transform(X_test)'''