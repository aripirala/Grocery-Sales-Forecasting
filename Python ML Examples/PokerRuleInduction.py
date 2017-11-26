# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:52:55 2017

@author: aripiralas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier



def hand_key(df):
    keys = list()
    for i in range(5):
        keys.append(df[i*2]*100 + df[i*2+1])
    
    keys.sort()
    
    keys_str = " ".join(str(x) for x in keys)
    return keys_str



poker_df = pd.read_csv("train.csv")

train_X, test_X, train_y, test_y = train_test_split(poker_df.iloc[:,:10], poker_df["hand"], 
                                                    train_size=0.75, 
                                                    random_state=123,
                                                    stratify=poker_df["hand"])

### estimate what is the score for using simple KNN classifier


clf = KNeighborsClassifier(random_state = 123)
clf.fit(train_X, train_y).score(test_X,test_y) 

rf = RandomForestClassifier(random_state = 123)
rf.fit(train_X, train_y).score(test_X,test_y) 

NB_B = BernoulliNB()
NB_B.fit(train_X, train_y).score(test_X,test_y) 

lr = LogisticRegression(random_state = 123)
lr.fit(train_X, train_y).score(test_X,test_y) 

## make a copy of train dataframe
train_X_copy = train_X.copy()
test_X_copy = test_X.copy()

## create a key column to account for any permutation of the hands


train_X_copy["key"] = train_X_copy.apply(hand_key,axis=1)
test_X_copy["key"] = test_X_copy.apply(hand_key, axis=1)

train_X_copy = train_X_copy["key"]
test_X_copy = test_X_copy["key"]

## convert in to categorical type of variables
train_X_copy["key"] = train_X_copy["key"].astype('category')
test_X_copy["key"] = test_X_copy["key"].astype('category')
train_y = train_y.astype('category')
test_y = test_y.astype('category')



vect = CountVectorizer(ngram_range=(1,5))
vect.fit(train_X_copy)

X_train = vect.transform(train_X_copy)
X_test = vect.transform(test_X_copy)


from sklearn.linear_model import LogisticRegression
clf =  LogisticRegression()
clf.fit(X_train,train_y).score(X_test,test_y)

knn = KNeighborsClassifier()
knn.fit(X_train,train_y).score(X_test,test_y)

NB_B = BernoulliNB()
NB_B.fit(X_train,train_y).score(X_test,test_y)

rf = RandomForestClassifier(random_state = 123, n_estimators=100)
rf.fit(X_train,train_y).score(X_test,test_y)


##### apply TFIDF transformation as well 



vect2 = TfidfVectorizer()

X_train2 = vect2.fit_transform(train_X_copy)
X_test2 = vect2.transform(test_X_copy)


######

clf =  LogisticRegression()
clf.fit(X_train2,train_y).score(X_test2,test_y)

knn = KNeighborsClassifier()
knn.fit(X_train2,train_y).score(X_test2,test_y)

NB_B = BernoulliNB()
NB_B.fit(X_train2,train_y).score(X_test2,test_y)

rf = RandomForestClassifier(random_state = 123, n_estimators=200)
rf.fit(X_train2,train_y).score(X_test2,test_y)

################################


pipe_CountVect_TfidTran_SGDC = make_pipeline(CountVectorizer(),TfidfTransformer(),SGDClassifier(random_state=123))

pipe_param_grid = {'countvectorizer__ngram_range':[(1,1),(1,2),(1,3),(1,5)],
        'tfidftransformer__use_idf':[True, False],
        'tfidftransformer__norm':['l1', 'l2'],
        'sgdclassifier__alpha':[.00001,0.000001],
        'sgdclassifier__penalty':['l2','elasticnet']                      
        }

grid = GridSearchCV(pipe_CountVect_TfidTran_SGDC, param_grid = pipe_param_grid, cv=5)
grid.fit(train_X_copy, train_y).score(test_X_copy,test_y)
print(grid.best_params_)


########################################

pipe2_CountVect_TfidTran_RandFor = make_pipeline(CountVectorizer(),TfidfTransformer(),RandomForestClassifier(random_state=123, n_estimators=100))

pipe2_param_grid = {'countvectorizer__ngram_range':[(1,1),(1,3),(1,5)],
        #'tfidftransformer__use_idf':[True, False],
        'tfidftransformer__norm':['l1', 'l2'],
        }

grid = GridSearchCV(pipe2_CountVect_TfidTran_RandFor, param_grid = pipe2_param_grid, cv=3, verbose=3)
grid.fit(train_X_copy, train_y).score(test_X_copy,test_y)
print(grid.best_params_)
print(grid.best_score_)
