#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:00:57 2017

@author: aripiralasrinivas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression


## drop NA values

#data_df.drop(['store_nbr'],inplace=True, axis=1)
#data_df.drop(['Year'],inplace=True, axis=1)
data_df.dropna(axis=0,subset=['onpromotion'], inplace=True)

#### convert date to Year and Week and Month

data_df['year'] = pd.DatetimeIndex(data_df['date']).year
data_df['month'] = pd.DatetimeIndex(data_df['date']).month
data_df['week'] = pd.DatetimeIndex(data_df['date']).week
data_df['day'] = pd.DatetimeIndex(data_df['date']).dayofweek

data_df.drop(['date'],inplace=True, axis=1)



#####

data_dummies_df = pd.get_dummies(data_df, columns=['item_nbr', 'onpromotion', 'family','class','perishable','year','month','week','day'])

train_X, test_X, train_y, test_y = train_test_split(data_dummies_df.loc[:,data_dummies_df.columns != 'unit_sales'], data_df["unit_sales"], 
                                                    train_size=0.75, 
                                                    random_state=123
                                                    )

############# step 1: Create a pipe with all the operations your dataframe goes through 
############# step 2: create a parameter grid to be passed to each of the operations within pipe
############# step 3: create a gridSearch and do cross validation
############# step 4: finally train your model using train_X and train_y and score the performance on test_X & test_y
pipe_Impute_LM = make_pipeline(Imputer(),LinearRegression())

pipe_param_grid = {'imputer__strategy':['mean','knn']                      
        }

grid = GridSearchCV(pipe_Impute_LM, param_grid = pipe_param_grid, cv=3, verbose=3)
grid.fit(train_X, train_y).score(test_X,test_y)
print(grid.best_params_)
