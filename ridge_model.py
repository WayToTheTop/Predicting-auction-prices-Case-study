import pandas as pd
from sklearn.model_selection import KFold
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn import cross_validation, linear_model
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

def loss(y, yhat):
    """
    Compute Root-mean-squared-log-error
    INPUT:
    y, yhat   : (numpy ndarrays) true and predicted targets
    OTPUT:
    (float)     : Root Mean Squared Log Error
    """
    # If predicted value is less then zero, we predict mininaml value for
    #training data (4750)
    yhat[yhat < 0] = 4750
    log_diff = np.log(yhat+1) - np.log(y+1)
    return np.sqrt(np.mean(log_diff**2))

def ourKfold(model, xdata, ydata, k = 5):
    '''
    Cross validate model wiht k volds. Returns mean error over all folds.
    INPUT:
    model      : (sklearn model)
    xdata      : (numpy ndarray)
    ydata      : (numpy ndarray)
    k          : (int) number of folds. Default is 5
    OUTPUT
    (float)    : (float) mean error of the model
    '''
    test_error = []
    kfold = KFold(n_splits = k, shuffle = True)

    #record list of RMSE (one for each fold) and return RMSE list
    for train_index, test_index in kfold.split(xdata):
        cvx_train, cvx_test = xdata[train_index], xdata[test_index]
        cvy_train, cvy_test = ydata[train_index], ydata[test_index]

        #call linear model
        model.fit(cvx_train, cvy_train)
        cvtest_predicted = model.predict(cvx_test)

        test_error.append(loss(cvy_test, cvtest_predicted))
    return np.mean(test_error)


def get_dummies(df, colname):
    """
    Dummifies column with colname in dataframe df
    INPUT:
    df:       (pandas DataFrame)
    colname:  (str) name of the colomn in df to be dummified
    OUTPUT:
    dummies:  (pandas DataFrame)
    """
    dummies = pd.get_dummies(df[colname])
    return dummies

def data_preproccessing(df, auct_list, is_test=False):
    '''
    Preprocess df for modeling.
    INPUT:
    df:        pandas DataFrame
    auct_list: (list) list of unique actioneer IDs
    is_test:   (bool) True if preporocessing test data, default is False.
    OUTPUT:
    X.values:  (numpy ndarray) predictors
    y.values:  (numpy ndarray) target values (only if is_test=True)
    '''
    #If year made is less then 1900 repalce it with mode
    condition = df.YearMade > 1900
    mode_year = df.YearMade[condition].mode()
    df.loc[~condition, 'YearMade'] = mode_year.values

    #Create 'age' feature
    df['age'] = df.saledate.dt.year - df.YearMade

    #Dummify AuctioneerID
    df.auctioneerID.fillna(100., inplace=True)
    for col in auct_list:
        df[str(col)] = (df.auctioneerID == col)

    #Create 5 dummy varibales from ProductGroup
    X = get_dummies(df, 'ProductGroup')

    #Create design matrix and target vector
    X.drop(labels=X.columns[-1],axis=1,inplace=True)
    X['age'] = df.age
    for col in auct_list:
        X[str(col)] = df[str(col)]
    if is_test:
        return X.values
    y = df['SalePrice']
    return X.values, y.values

def ridge_model(alpha=0.1):
    '''
    Creates Ridge Regression model with regularizing parameter alpha.
    INPUT:
    alpha:     (float) regularazing parameter
    OUTPUT:
    ridge:     (sklearn model)
    '''
    ridge = Ridge(alpha=alpha)
    return ridge

def train_n_predict(df_test, model, X_train, Y_train, X_test):
    '''
    Trains model on train data, makes predictions on test data, saves
    predictions to file predictions.csv
    INPUT:
    df_test:    (pandas DataFrame) with test data
    model:      (sklearn model) model for predictions
    X_train:    (numpy ndarray) predictors data for training
    Y_train:    (numpy ndarray) target data
    X_test:     (numpy ndarray) predictors data for testing
    '''
    model.fit(X_train, Y_train)
    prediction (list)= model.predict(X_test)
    df_test['SalePrice'] = map(int, prediction)
    output = df_test[['SalesID', 'SalePrice']].set_index('SalesID')
    output.to_csv('predictions.csv', sep=',')
