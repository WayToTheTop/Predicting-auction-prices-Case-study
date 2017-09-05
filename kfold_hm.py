import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation, linear_model
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

def load data(address):
    address = 'data/Train.csv'
    df = pd.read_csv('address')

    X = df['ProductGroup']
    Y = df['SalePrice']
    colNames = df.columns

def ourLRMSE(y, yhat):
    """Compute Root-mean-squared-error"""
    log_diff = np.log(yhat+1) - np.log(y+1)
    return np.sqrt(np.mean(log_diff**2))

def ourKfold(model, xdata, ydata, k = 5):
    #take the data and run through k-fold cross validation
    #record list of RMSE (one for each fold) and return RMSE list
    test_error = []
    kfold = KFold(n_splits = k, shuffle = True)
    for train_index, test_index in kfold.split(X):
        cvx_train, cvx_test = xdata[train_index], xdata[test_index]
        cvy_train, cvy_test = ydata[train_index], ydata[test_index]

        #call linear model
        model.fit(cvx_train, cvy_train)
        cvtest_predicted = model.predict(cvx_test)

        test_error.append(ourLRMSE(cvy_test, cvtest_predicted))
    return np.mean(test_error)
