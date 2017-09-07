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

    df = pd.read_csv(train_address,
                parse_dates=['saledate'], infer_datetime_format=True)
    df_test = pd.read_csv(test_address,
                parse_dates=['saledate'], infer_datetime_format=True)
    auct_list = df.auctioneerID.unique()
    X, y = data_preproccessing(df, auct_list)
    X_test = data_preproccessing(df_test, auct_list, is_test=True)
    model = ridge_model()
    error = ourKfold(model, X, y)
    train_n_predict(df_test, model, X, y, X_test)
    return error


def ourLRMSE(y, yhat):
    """Compute Root-mean-squared-error"""
    yhat[yhat < 0] = 4750
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
    dummies = pd.get_dummies(df[colname])
    return dummies

def data_preproccessing(df, auct_list, is_test=False):
    condition = df.YearMade > 1900
    mode_year = df.YearMade[condition].mode()
    df.loc[~condition, 'YearMade'] = mode_year.values
    df['age'] = df.saledate.dt.year - df.YearMade
    df.auctioneerID.fillna(100., inplace=True)
    for col in auct_list:
        df[str(col)] = (df.auctioneerID == col)
    #Create 5 dummy varibales from ProductGroup
    X = get_dummies(df, 'ProductGroup')
    X.drop(labels=X.columns[-1],axis=1,inplace=True)
    X['age'] = df.age
    for col in auct_list:
        X[str(col)] = df[str(col)]
    if is_test:
        return X.values
    y = df['SalePrice']
    return X.values, y.values

def ridge_model(alpha=0.1):
    ridge = Ridge(alpha=alpha)
    return ridge

def train_n_predict(df_test, model, X_train, Y_train, X_test):
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    df_test['SalePrice'] = map(int, prediction)
    output = df_test[['SalesID', 'SalePrice']].set_index('SalesID')
    output.to_csv('data/BELJ_predictions.csv', sep=',')


if __name__ == '__main__':
    print (run('data/Train.csv','data/test.csv'))
    #X,y = (run('data/Train.csv','data/test.csv'))
