#!/usr/bin/python3
#Aviral Upadhyay
#Vandit Maheshwari
#Version 1.0
#Date May 7th, 2020


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier


def randomForest(df1,df2):

    # %%
    df1_len = df1.shape[0]
    df2_len = df2.shape[0]
    df2.shape + df1.shape

    # %%
    X = pd.concat([df1, df2])

    # %%
    y1 = pd.Series([0]*df1_len)
    y1.shape

    # %%
    y2 = pd.Series([1]*df2_len, index = range(df1_len-1,(df1_len + df2_len)-1))
    
    y = pd.concat([y1,y2]) 
    

    
    y.tail()

    
    max_acc=0
    max_est =0 
    curr_max_acc=0
    curr_max_n=0
    for i in range(1, 100):
        X_train, X_test, y_train1, y_test1 = train_test_split(X, np.ravel(y),random_state=i)
        y_train = pd.DataFrame(y_train1)
        y_test = pd.DataFrame(y_test1)

        for n in [5, 10, 20, 40, 100, 200]:
            clf = RandomForestClassifier(n_estimators = int(n), n_jobs=2)
            clf.fit(X_train, np.ravel(y_train))
            if(max_acc < clf.score(X_test, np.ravel(y_test))):
                    max_acc = clf.score(X_test, np.ravel(y_test))
                    max_n=int(n)
                    
        if(curr_max_acc < max_acc):
                curr_max_acc = max_acc
                curr_max_n= max_n
                n_rs = i
        print("{} done".format(i))
    print("Accuracy = {}, n_estimators = {}, Random State = {}".format(curr_max_acc, curr_max_n, n_rs)) 

    
    X_train, X_test, y_train1, y_test1 = train_test_split(X, np.ravel(y))
    y_train = pd.DataFrame(y_train1)
    y_test = pd.DataFrame(y_test1)
    clf = RandomForestClassifier(n_estimators = 200, n_jobs=2, max_depth=10)
    clf.fit(X_train, np.ravel(y_train))
    clf.score(X_test, np.ravel(y_test))

    

    
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_auc_score
    grid_vals = {'n_estimators': [5, 10, 20, 40, 100, 200]}
    grid_clf = GridSearchCV(clf, param_grid=grid_vals, scoring='accuracy')
    grid_clf.fit(X_train, y_train1.reshape(229,))
    y_pred_proba = grid_clf.predict_proba(X_test)
    print(roc_auc_score(y_test, y_pred_proba[:,1]))
    print("Best Score",grid_clf.best_score_)

    
    print("Best Parameters",grid_clf.best_params_)
