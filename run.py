import pandas as pd
from ridge_model import data_preproccessing, ridge_model, ourKfold  \
                                                                train_n_predict

def run(train_address, test_address):
    """
    Runs Linera Regression Ridge model, saves predictions to predictions.csv
    Returns crossvalidated RMSLE error.
    INPUT:
    train_address, test_address: (str) location of train and test data
    OUTPUT:
    (float)    : RMSLE error
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

if __name__ == '__main__':
    print (run('data/Train.csv','data/test.csv'))
