#!/usr/bin/python3
#Aviral Upadhyay
#Vandit Maheshwari
#Version 1.0
#Date May 9th, 2020


import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import decomposition
import os 
import scipy.stats as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc



def svmPoly(df_co,df_pt):
    
    df_co_pca= df_co
    df_pt_pca= df_pt


    df_co_len = df_co_pca.shape[0]
    df_pt_len = df_pt_pca.shape[0]


    df_co_pca = pd.DataFrame(df_co_pca)
    df_pt_pca = pd.DataFrame(df_pt_pca)


    X = pd.concat([df_co_pca, df_pt_pca])


    y1 = pd.Series([0]*df_co_len)


    y2 = pd.Series([1]*df_pt_len, index = range(df_co_len-1,(df_co_len + df_pt_len)-1))


    y = pd.concat([y1,y2])
    

    c_val = 0
    score_val = 0
    min_range = 0
    max_range = 100
    state = 0
    gamma = 0
    for rand_state in range(min_range, max_range):
        X_train, X_test, y_train1, y_test1 = train_test_split(X, y, random_state = rand_state)
        y_train = pd.DataFrame(y_train1)
        y_test = pd.DataFrame(y_test1)

        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        
        for this_gamma in ([0.1, 1, 5]):
        
            for this_C in ([0.1, 1, 15, 250]):

                clf = SVC(kernel = 'poly', gamma = this_gamma, C = this_C).fit(X_train, y_train.values.ravel())
                score = clf.score(X_test, np.ravel(y_test))

            if score > score_val:
                gamma = this_gamma
                c_val = this_C
                score_val = score
                state = rand_state

        print('{} out of {} done'.format(rand_state - min_range, max_range - min_range))
    print("C = {}, gamma = {}, Score = {}, Random State = {}".format(c_val, gamma, score_val, state))


    for this_gamma in ([0.01, 1, 5]):
        
        for this_C in ([0.1, 1, 15, 250]):
             
            clf = SVC(kernel = 'poly', gamma = this_gamma, C = this_C).fit(X_train, y_train.values.ravel())
            print('gamma = {:.2f}, C = {:.2f}, accuracy = {}'.format(this_gamma, this_C, clf.score(X_test, np.ravel(y_test))))

    gamma = 5
    C = 1
    for d in ([1,2,3,5]):
        clf = SVC(kernel = 'poly', gamma = gamma, C = C, degree = d).fit(X_train, y_train.values.ravel())
        print('gamma = {:.2f}, C = {:.2f}, accuracy = {}'.format(gamma, C, clf.score(X_test, np.ravel(y_test))))

    # %%
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve
    from sklearn.model_selection import GridSearchCV
    best_score = 0
    for i in range(100):
        X_train, X_test, y_train1, y_test1 = train_test_split(X, y)
        y_train = pd.DataFrame(y_train1)
        y_test = pd.DataFrame(y_test1)

        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        clf = SVC(kernel = 'poly', gamma = 1, C = 1).fit(X_train, y_train.values.ravel())
        grid_vals = {'C': [0.1, 1, 5, 10, 12, 15, 20, 25, 50, 100, 250], 'gamma': [0.01, 1, 2, 3, 5], 'degree': [2, 3, 4]}
        grid_clf = GridSearchCV(clf, param_grid=grid_vals, scoring='accuracy')
        grid_clf.fit(X_train, y_train1.values.ravel())
        decision_fn_scores = grid_clf.decision_function(X_test)
        
        if best_score < grid_clf.best_score_:
            best_score = grid_clf.best_score_
            best_params = grid_clf.best_params_
            prediction = grid_clf.predict(X_test)
            y_score_svm = grid_clf.decision_function(X_test)
            fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
            roc_auc_svm = auc(fpr_svm, tpr_svm)
            prec = precision_score(y_test, prediction)
            rec = recall_score(y_test, prediction)
            f1 = f1_score(y_test, prediction)

    print(best_score)
    print(best_params)




    print('Precision: {:.4f}'.format(prec))
    print('Recall: {:.4f}'.format(rec))
    print('F1: {:.4f}'.format(f1))
    print('AUC: {:.4f}'.format(roc_auc_svm))