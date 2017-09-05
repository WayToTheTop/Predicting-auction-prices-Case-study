import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix

df = pd.read_csv('Train.csv')
df.describe().T
df.info()

'''
401,125 records total
Many fields only partially filled out (usage band). Many appear not to apply to all kinds of equipment
'''

df['SalePrice'].hist()
scatter_matrix(df)

dfnew = df[['SalesID', 'SalePrice', 'MachineID', 'ModelID', 'datasource', 'auctioneerID', 'YearMade', 'UsageBand','saledate', 'fiModelDesc', 'fiBaseModel', 'ProductSize','state', 'ProductGroup']]

dfsmall =

dfnew[dfnew.ProductGroup== 'TEX'].hist()
dfnew[dfnew.ProductGroup== 'TTT'].hist()
dfnew[dfnew.ProductGroup== 'BL'].hist()
dfnew[dfnew.ProductGroup== 'WL'].hist()
dfnew[dfnew.ProductGroup== 'SSL'].hist()
dfnew[dfnew.ProductGroup== 'MG'].hist()
plt.show()

df.groupby('auctioneerID')['SalePrice'].mean()
