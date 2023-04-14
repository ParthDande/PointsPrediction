# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model 
from sklearn.model_selection import train_test_split
df2=pd.read_csv("nba_data_processed.csv")
reg= linear_model.LinearRegression()
df2=df2.dropna()
df = df2[['FGA','PTS']]
X = np.array(df['FGA']).reshape(-1, 1)
y = np.array(df['PTS']).reshape(-1, 1)
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
regr =linear_model.LinearRegression()
regr.fit(X,y)
print(regr.predict(X))
plt.plot(X,regr.predict(X),color='black')
plt.scatter(X,y , marker = "+")
plt.show()