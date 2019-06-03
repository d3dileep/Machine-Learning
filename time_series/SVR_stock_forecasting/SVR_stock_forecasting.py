import numpy as np
import pandas as pd
import datetime
from sklearn.svm import SVR
from sklearn import preprocessing, cross_validation, svm
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, Imputer
import matplotlib.pyplot as plt
import seaborn
import math as m
import pickle

query = "SELECT date, volume, daily_high, daily_low, closing_price, opening_price FROM `pecten_dataset.stockprice_historical_collection` WHERE constituent_name = 'ADIDAS AG' ORDER BY date ASC"

# get data from bq
data = pd.read_gbq(query, project_id="igenie-project", index_col=None, col_order=None, reauth=False, verbose=None, private_key="key directory", dialect='standard')

data = data[['opening_price', 'daily_high', 'daily_low', 'closing_price', 'volume']]
print(data.head())

#reset index as numbers spanning the lenght of the dataframe

data.index = pd.RangeIndex(len(data.index))
data.index = range(len(data.index))

# Drop rows with missing values

data=data.dropna()
data.Volume = data.volume.astype(float)
print(data.dtypes)
print(data.index[-1])

# pick a forecast column
forecast_col = 'closing_price'

# Chosing 30 days as number of forecast days
forecast_out = int(30)
print('length =',len(data), "and forecast_out =", forecast_out)

# Creating label by shifting 'Close' according to 'forecast_out'
data['label'] = data[forecast_col].shift(-forecast_out)
print(data.head(2))
print('\n')
# If we look at the tail, it consists of n(=forecast_out) rows with NAN in Label column
print(data.tail(2))

# Define features Matrix X by excluding the label column which we just created
X = np.array(data.drop(['label'], 1))

scaler = StandardScaler()
X = scaler.fit_transform(X)

print(X[1,:])

# X contains last 'n= forecast_out' rows for which we don't have label data
# Put those rows in different Matrix X_forecast_out by X_forecast_out = X[end-forecast_out:end]

X_forecast_out = X[-forecast_out:]
X = X[:-forecast_out]
print ("Length of X_forecast_out:", len(X_forecast_out), "& Length of X :", len(X))

# Similarly Define Label vector y for the data we have prediction for
# A good test is to make sure length of X and y are identical
y = np.array(data['label'])
y = y[:-forecast_out]
print('Length of y: ',len(y))

# Cross validation (split into test and train data)
# test_size = 0.3 ==> 30% data is test data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3)

print('length of X_train and x_test: ', len(X_train), len(X_test))

# Train
clf = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, gamma='auto', kernel='rbf', max_iter=-1, shrinking=True, epsilon=0.1, tol=0.001, verbose=False)
model = clf.fit(X_train,y_train)
# Test
accuracy = model.score(X_test, y_test)
print("Accuracy of model: ", accuracy)

# predict all values in test data
a = model.predict(X_test)
#print(a)

# predict values in original data
b = model.predict(X)
#print(b)
