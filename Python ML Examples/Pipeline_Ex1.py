# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:49:31 2017

@author: aripiralas
"""

### using pipeline to do transformations and grid search and run the model
## MinMaxScaler -> PCA transformation -> Gridsearch CV on Logistic Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import load_digits
digits = load_digits()


digits_df = pd.DataFrame(digits.data)
digits_df["digit"] = digits.target

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(digits_df.iloc[:,:64], digits_df["digit"], 
                                                    train_size=0.75, 
                                                    random_state=123,
                                                    stratify=digits_df["digit"])


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier


pipe1_MinMax_PCA_LogR = make_pipeline(MinMaxScaler(),PCA(),LogisticRegression(random_state=123))

pipe2_StdSca_PCA_LogR = make_pipeline(StandardScaler(), PCA(), LogisticRegression(random_state=123))

pipe3_MinMax_PCA_RndForClass = make_pipeline(MinMaxScaler(), PCA(), RandomForestClassifier(n_estimators = 100, random_state = 123))

pipe1_2_param_grid = {'logisticregression__C':[.1,1,10,100]                    
        }

pipe3_param_grid = { 'randomforestclassifier__max_features':[5,10,30,64]
        }
## Logistic Regression Classifier combined with MinMax Transformation
grid1 = GridSearchCV(pipe1_MinMax_PCA_LogR, param_grid=pipe1_2_param_grid, cv=5)
grid1.fit(train_X,train_y)
print(grid1.best_params_)
grid1.score(test_X,test_y)

pred_y = grid1.predict(test_X)
# Compute confusion matrix
cnf_matrix = confusion_matrix(test_y,pred_y)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=grid.classes_,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=grid.classes_, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

## Logistic Regression Classifier combined with StandardScaler Transformation
grid2 = GridSearchCV(pipe2_StdSca_PCA_LogR, param_grid=pipe1_2_param_grid, cv=5)
grid2.fit(train_X,train_y)
print(grid2.best_params_)
grid2.score(test_X,test_y)


### RandomForest Classifier combined with MinMax Transformation

grid3 = GridSearchCV(pipe3_MinMax_PCA_RndForClass, param_grid=pipe3_param_grid, cv=5)
grid3.fit(train_X,train_y)
print(grid3.best_params_)
grid3.score(test_X,test_y)

 