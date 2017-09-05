import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation, linear_model
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

def load data(address):
    df = pd.read_csv('address')

    X = df['ProductGroup']
    Y = df['SalePrice']
    colNames = df.columns

def rmse(theta, thetahat):
    """Compute Root-mean-squared-error"""
    pas
