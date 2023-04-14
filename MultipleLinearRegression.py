# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:19:05 2023

@author: Parth
"""
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model 
#from sklearn.model_selection import train_test_split
df=pd.read_csv("nba_data_processed.csv")
df2 = df.replace(r'^\s*$', np.nan, regex=True).dropna()#replaced all the null data or missing data with NaN and the used dropna function 
#to drop all the rows with Nan Values so we get a cleaner data
df= df2[['FGA','FG%','3PA','3P%','FTA','FT%','PTS']]#creatd a new dataframe with the select features as columns
reg =linear_model.LinearRegression()
reg.fit(df[['FGA','FG%','3PA','3P%','FTA','FT%']],df['PTS'])
plt.scatter(df['FGA'],df['PTS'],marker="+")
plt.scatter(df[['FGA']],reg.predict(df[['FGA','FG%','3PA','3P%','FTA','FT%']]),color='RED')
plt.show()

#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)





 