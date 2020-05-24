#!/usr/bin/python3
#Aviral Upadhyay
#Vandit Maheshwari
#Version 1.0
#Date May 7th, 2020


import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import decomposition
import os
import scipy.stats as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

def kNN(df_co,df_pt):

    df_co_len = df_co.shape[0]
    df_pt_len = df_pt.shape[0]


    df_co_pca = pd.DataFrame(df_co)
    df_pt_pca = pd.DataFrame(df_pt)


    y1 = pd.Series([0]*df_co_len)
   


    y2 = pd.Series([1]*df_pt_len, index = range(df_co_len-1,(df_co_len + df_pt_len)-1))
    


    y = pd.concat([y1,y2]) 
    


    X = pd.concat([df_co_pca, df_pt_pca])





    score_val = 0
    min_range = 0
    max_range = 100
    state = 0
    neigh = 0

    for rand_state in range(min_range, max_range):
        X_train, X_test, y_train1, y_test1 = train_test_split(X, y, random_state = rand_state)
        y_train = pd.DataFrame(y_train1)
        y_test = pd.DataFrame(y_test1)

        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        
        for neighbors in ([1, 3, 5, 10, 15, 20]):
            knn = KNeighborsClassifier(n_neighbors = neighbors)
            knn.fit(X_train, y_train.values.ravel())
            score = knn.score(X_test, y_test.values.ravel())

            if score > score_val:
                neigh = neighbors
                score_val = score
                state = rand_state

        print('{} out of {} done'.format(rand_state - min_range, max_range - min_range))
    print("Neighbors = {}, Score = {}, Random State = {}".format(neigh, score_val, state))



    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)




        
    for neighbors in ([1, 3, 5, 10, 15, 20]):

        knn = KNeighborsClassifier(n_neighbors = neighbors)
        knn.fit(X_train, y_train.values.ravel())
        print("Neighbors = {}".format(neighbors))
        print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
        print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))
        print()



    X_train, X_test, y_train1, y_test1 = train_test_split(X, y)
    knn = KNeighborsClassifier(n_neighbors = 1)


    max_score = 0
    best_param = []
    for i in range(0,100):
        from sklearn.model_selection import GridSearchCV

        grid_values = {'n_neighbors': [1, 2, 5, 10, 20, 50, 70, 100]}

        grid_lr = GridSearchCV(knn, param_grid = grid_values, scoring = 'accuracy')
        grid_lr.fit(X_train, y_train1.values.ravel())
        if max_score < grid_lr.best_score_:
            max_score = grid_lr.best_score_
            best_param = grid_lr.best_params_
    print( "max_score",max_score)
    print( "best_param",best_param)

    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve
    best_score = 0
    for i in range(100):
        X_train, X_test, y_train1, y_test1 = train_test_split(X, y)
        y_train = pd.DataFrame(y_train1)
        y_test = pd.DataFrame(y_test1)

        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train.values.ravel())

        grid_values = {'n_neighbors': [1, 2, 5, 10, 20, 50, 70, 100]}
        grid_clf = GridSearchCV(knn, param_grid=grid_values, scoring='roc_auc')
        grid_clf.fit(X_train, y_train1.values.ravel())
        decision_fn_scores = grid_clf.predict(X_test)
        
        if best_score < grid_clf.best_score_:
            best_score = grid_clf.best_score_
            prediction = grid_clf.predict(X_test)
            
            
            prec = precision_score(y_test, prediction)
            rec = recall_score(y_test, prediction)
            f1 = f1_score(y_test, prediction)
            
    print(best_score)



    print(grid_clf.best_params_)


    print("Precision: {}".format(prec))
    print("Recall: {}".format(rec))
    print("F1: {}".format(f1))


