#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:07:42 2017

@author: aripiralasrinivas
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



FILE_PATH = '/Users/aripiralasrinivas/Documents/Data Science/Kaggle Competitions/Grocery Sales Forecasting/Data/'

holiday_df = pd.read_csv(FILE_PATH+'holidays_events.csv')
items_df = pd.read_csv(FILE_PATH+'items.csv')
oil_df = pd.read_csv(FILE_PATH+'oil.csv')
stores_df = pd.read_csv(FILE_PATH+'stores.csv')
test_df = pd.read_csv(FILE_PATH+'test.csv')
train_df = pd.read_csv(FILE_PATH+'train.csv')
transactions_df = pd.read_csv(FILE_PATH+'transactions.csv')


###


### convert date object to datetime object

train_df['date']=pd.to_datetime(train_df['date'])
test_df['date']=pd.to_datetime(test_df['date'])


### select a data set for store 54

train_store54_df = train_df.loc[train_df.store_nbr==54,:]

########### create one big dataframe with holidays, oil info, items info into train dataframe ####

oil_df.date = pd.to_datetime(oil_df.date)
## oil_df.reset_index(inplace=True)
## train_df.reset_index(inplace=True)

train_oil_df = pd.merge(train_store54_df,oil_df, how='left', on='date')


### bring in transactions 
transactions_df.date = pd.to_datetime(transactions_df.date)

train_oil_trans_df = pd.merge(train_oil_df,transactions_df, how='left', on=['date','store_nbr'])

### bring item information

train_oil_trans_item_df = pd.merge(train_oil_trans_df,items_df, how='left', on='item_nbr')

## convert items and store numbers to categorical values

train_oil_trans_item_df.drop(['id'], inplace=True, axis=1)
train_oil_trans_item_df.store_nbr = train_oil_trans_item_df.store_nbr.astype('category')
train_oil_trans_item_df.item_nbr = train_oil_trans_item_df.item_nbr.astype('category')

train_oil_trans_item_df['class'] = train_oil_trans_item_df['class'].astype('category')

train_oil_trans_item_df.perishable = train_oil_trans_item_df.perishable.astype('category')


### 

