import pandas as pd
from sklearn.model_selection import KFold
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn import cross_validation, linear_model
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np


def run(train_address, test_address):
    """
    address - string filepath
    """

    df = pd.read_csv(train_address)
    df_test = pd.read_csv(test_address)
    X = get_dummies(df, 'ProductGroup')
    Y = df['SalePrice'].values
    X_test = get_dummies(df_test, 'ProductGroup')

    model = ridge_model()
    error = ourKfold(model, X, Y)
    train_on_data(df_test, model, X, Y, X_test)
    return error


def ourLRMSE(y, yhat):
    """Compute Root-mean-squared-error"""
    log_diff = np.log(yhat+1) - np.log(y+1)
    return np.sqrt(np.mean(log_diff**2))

def ourKfold(model, xdata, ydata, k = 5):
    #take the data and run through k-fold cross validation
    #record list of RMSE (one for each fold) and return RMSE list
    test_error = []
    kfold = KFold(n_splits = k, shuffle = True)
    for train_index, test_index in kfold.split(xdata):
        cvx_train, cvx_test = xdata[train_index], xdata[test_index]
        cvy_train, cvy_test = ydata[train_index], ydata[test_index]

        #call linear model
        model.fit(cvx_train, cvy_train)
        cvtest_predicted = model.predict(cvx_test)

        test_error.append(ourLRMSE(cvy_test, cvtest_predicted))
    return np.mean(test_error)


def get_dummies(df, colname):
    """
    colname - string
    """
    dummies = pd.get_dummies(df[colname]).values
    return dummies

def ridge_model(alpha=0.5):
    ridge = Ridge(alpha=alpha)
    return ridge

def train_on_data(df_test, model, X_train, Y_train, X_test):
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    df_test['SalePrice'] = map(int, prediction)
    output = df_test[['SalesID', 'SalePrice']].set_index('SalesID')
    output.to_csv('data/BELJ_predictions.csv', sep=',')


if __name__ == '__main__':
    print (run('data/Train.csv','data/test.csv'))
