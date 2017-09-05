import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn import cross_validation, linear_model
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np


def run(address):
    """
    address - string filepath
    """

    df = pd.read_csv(address)
    X = get_dummies('ProductGroup')
    Y = df['SalePrice'].values

    model = ridge_model()

    #kfold(ridge)
    #rmlse(y_predicted, y_test)

def get_dummies(colname):
    """
    colname - string
    """
    dummies = pd.get_dummies(df[colname]).values
    return dummies

def ridge_model(alpha=0.5):
    ridge = Ridge(alpha=alpha)
    return ridge

def rmlse(y_predicted, y_test):
    log_diff = np.log(y_predicted+1) - np.log(y_test+1)
    return np.sqrt(np.mean(log_diff**2))
