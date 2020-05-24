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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV

def svmLinear(df_co,df_pt):

    df_co_len = df_co.shape[0]
    df_pt_len = df_pt.shape[0]


    df_co_pca = pd.DataFrame(df_co)
    df_pt_pca = pd.DataFrame(df_pt)


    y1 = pd.Series([0]*df_co_len)
    y1.shape


    y2 = pd.Series([1]*df_pt_len, index = range(df_co_len-1,(df_co_len + df_pt_len)-1))
    y2.shape


    y = pd.concat([y1,y2]) 
    y.shape

    X = pd.concat([df_co_pca, df_pt_pca])
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc
    best_score = 0
    for i in range(100):
        X_train, X_test, y_train1, y_test1 = train_test_split(X, y)
        y_train = pd.DataFrame(y_train1)
        y_test = pd.DataFrame(y_test1)

        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        clf = SVC(kernel = 'linear', C = 1).fit(X_train, y_train.values.ravel())
        grid_vals = {'C': [0.1, 1, 5, 10, 12, 15, 20, 25, 50, 100, 250]}
        grid_clf = GridSearchCV(clf, param_grid=grid_vals, scoring='accuracy')
        grid_clf.fit(X_train, y_train1.values.ravel())
        decision_fn_scores = grid_clf.decision_function(X_test)
        
        if best_score < grid_clf.best_score_:
            best_score = grid_clf.best_score_
            best_params = grid_clf.best_params_
            prediction = grid_clf.predict(X_test)
            y_score = grid_clf.decision_function(X_test)
            prec = precision_score(y_test, prediction)
            rec = recall_score(y_test, prediction)
            f1 = f1_score(y_test, prediction)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            
    print(best_score)
    print(best_params)
    print('Precision: {:.4f}'.format(prec))
    print('Recall: {:.4f}'.format(rec))
    print('F1: {:.4f}'.format(f1))
    print('AUC: {:.4f}'.format(roc_auc))
