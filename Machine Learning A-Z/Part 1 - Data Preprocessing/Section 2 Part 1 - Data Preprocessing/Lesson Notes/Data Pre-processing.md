## Data Pre-Processing

### ★ Import Libraries
#### What is a library?
A library is a tool used to do a specific job.

####3 Essential libraries:
1. **Numpy:** contains mathematical tools.
2. **Matplotlib - Pyplot(sub-library):** plots charts.
3. **Pandas:** best library to import & manage data sets.

#### Import libraries in python:
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

### ★ Import Data Sets
* Set a working directory before importing data sets.
* Use the library, Pandas to import data sets

```
dataset = pd.read_csv('Data.csv')
```

### ★ Create Matrix of Features & Dependent Variable Feature
```
X = dataset.iloc[:, :-1].values    #rows[0:x], col[0:y-1]
y = dataset.iloc[:, 3].values      #rows[0:x], col[y]
```

### ★ Take Care of Missing Data
* Data can have missing values for a number of reasons (ie. not recorded observations, data corruption, etc.)
* Handling missing data is important as many ML algorithms do not support data with missing values.
* Use the Imputer from the Scikit-learn library
* Select the word Imputer + `Ctrl + i` to view the documentation

```
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
```

### ★ Encode Categorical Variables into numbers
* Categorical Data: variables that contain label values rather than numerical values.
* Most ML algorithms cannot operate on label data directly & require all input & output variables to be numeric.
* Two steps to convert Categorical Data to Numerical Data:
	1. Integer Encoding
		- each unique category value is assigned an integer value
		Example:
        ```
        Red = 0
        Blue = 1
        Green = 2
        ```
	2. One-Hot Encoding
		- When there's no ordinal relationship, integer encoding is not enough.
		- integer encoded variable is removed and a new binary variable is added for each unique integer value
		- the binary variables are often called dummy variables
		Example:
        ```
        red		blue		green
         1		 0			  0
         0		 1			  0
         0		 0			  1
        ```
* Use the LabelEncoder & OneHotEncoder modules from the Scikit-learn library

```
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_country = LabelEncoder()
X[:, 0] = labelencoder_country.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_purchased = LabelEncoder()
y = labelencoder_purchased.fit_transform(y)
```

### ★ Split the dataset into Training set & Test set
* Training set: dataset of examples used for learning, to fit the parameters of a classifier/algorithm.
* Test set: set of examples used only to assess the performance of a classifier.
* It is recommended to split the dataset to a 8:2 ratio (training:test).

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

### ★ Feature Scaling
* method used to standardize the range of independent variables or features of data.
* Some methods used:
	1. Standardization

		$$
		 x_{s} = \frac{x - mean(x)} {standard deviation(x)}
		$$

	2. Normalization

		$$
        x_{n} = \frac{x - min(x)} {max(x) - min(x)}
        $$

```
from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
X_train = X_sc.fit_transform(X_train)
X_test = X_sc.transform(X_test)
```
* Difference between fit and fit_transform:
	- **fit** calculates the mean & variance from the values in X_train
	- **transform** transforms all of the features by subtracting the mean from the the values in X_train & dividing it by the variance.
	- You  fit the scaler using only the training data so that there will be no bias in your model with information from the test data.
	- you only want to transform the test data by using the parameters(mean & variance) computed on the training data.

