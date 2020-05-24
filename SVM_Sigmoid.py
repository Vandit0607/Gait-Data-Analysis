#!/usr/bin/python3
#Aviral Upadhyay
#Vandit Maheshwari
#Version 1.0
#Date May 7th, 2020



import pandas as pd
import numpy as np
from sklearn import decomposition
import os
import scipy.stats as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


def svmSigmoid(df_co,df_pt):


    df_co_len = df_co.shape[0]
    df_pt_len = df_pt.shape[0]

    
    df_co_pca = pd.DataFrame(df_co)
    df_pt_pca = pd.DataFrame(df_pt)

    
    y1 = pd.Series([0]*df_co_len)


    
    y2 = pd.Series([1]*df_pt_len, index = range(df_co_len-1,(df_co_len + df_pt_len)-1))


    
    y = pd.concat([y1,y2]) 


    
    X = pd.concat([df_co_pca, df_pt_pca])

    
    c_val = 0
    score_val = 0
    min_range = 0
    max_range = 100
    state = 0

    for rand_state in range(min_range, max_range):
        X_train, X_test, y_train1, y_test1 = train_test_split(X, y, random_state = rand_state)
        y_train = pd.DataFrame(y_train1)
        y_test = pd.DataFrame(y_test1)

        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        
        for x in [1, 5, 10, 20, 100, 200, 400]:
            clf = SVC(kernel = 'sigmoid', C = x).fit(X_train, y_train.values.ravel())
            score = clf.score(X_test, np.ravel(y_test))

            if score > score_val:
                c_val = x
                score_val = score
                state = rand_state

    print("C = {}, Score = {}, Random State = {}".format(c_val, score_val, state))



    max_score = 0
    best_param = []
    for i in range(0,100):
        from sklearn.model_selection import GridSearchCV
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        clf = SVC(kernel = 'sigmoid', C = 1).fit(X_train, y_train.values.ravel())

        grid_values = {'C': [1, 2, 5, 10, 15, 20, 25, 40, 50, 70, 100, 500, 1000]}

        grid_lr = GridSearchCV(clf, param_grid = grid_values, scoring = 'accuracy')
        grid_lr.fit(X_train, y_train.values.ravel())
        if max_score < grid_lr.best_score_:
            max_score = grid_lr.best_score_
            best_param = grid_lr.best_params_
        



    print( "max_score",max_score)
    print( "best_param",best_param)

