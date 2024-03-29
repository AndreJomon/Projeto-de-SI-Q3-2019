# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 19:20:03 2019

@author: Felipe
"""

import numpy as np
import os
from pandas import read_csv
import time
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import normalize
import argparse
import csv
    

def sum_confusion_matrix(conf_matrix, conf_matrix_sum):

    for i in range(0,2):
        for j in range(0,2):
            conf_matrix_sum[i][j] = conf_matrix_sum[i][j] + conf_matrix[i][j]

    return conf_matrix_sum

def create_csv(path, name_csvfile, cl):

    if not os.path.exists(os.path.join(path, 'csvfiles')):
        os.makedirs(os.path.join(path, 'csvfiles'))
        
    if 'knn' in cl:
        param = 'Neighbors'
    
    elif 'svm' in cl:
        param = 'C value'
        
    elif 'dt' in cl:
        param = 'Features'
    
    elif 'mlp' in cl:
        param = 'Layers'

    if not os.path.exists(os.path.join(path, 'csvfiles', name_csvfile + '.csv')):
        with open(os.path.join(path, 'csvfiles', name_csvfile + '.csv'), 'w', newline='') as csvf:
            writer = csv.writer(csvf, delimiter=',')
            writer.writerow(['sep=,'])
            writer.writerow(['Name', 'Classifier', param, 'Precision', 'Recall', 'Fscore', 'Consusion Matrix'])
            
def appendrow_csv(path, name_csvfile, cl, precision, recall, fscore, parameter, conf_matrix_sum):
    

    if os.path.exists(os.path.join(path, 'csvfiles', name_csvfile + '.csv')):
        with open(os.path.join(path, 'csvfiles', name_csvfile + '.csv'), 'a', newline='') as csvf:

            writer = csv.writer(csvf, delimiter=',')
            writer.writerow([name_csvfile, cl, str(parameter), round(precision, 3), round(recall, 3), round(fscore, 3), conf_matrix_sum[0]])
            writer.writerow(['', '', '', '', '', '' , conf_matrix_sum[1]])
            writer.writerow([''])
            
def create_clasifiers(cl):
    
    classifiers = []
    parameters = []
    names = []

    if 'knn' in cl:
        
        neighbors = [1, 5, 10, 15, 20]
        
        for k in neighbors:
            
            name = cl + '_' + str(k) + 'neighbors'
            classifiers.append(KNeighborsClassifier(n_neighbors=k))
            parameters.append(k)
            names.append(name)
    
    elif 'svm' in cl:
        
        c = [0.1, 1, 5, 10]
        
        for Cval in c:
            
            name = cl + '_' + str(Cval) + 'Cval'
            classifiers.append(LinearSVC(random_state=42, C=Cval, max_iter=10000))
            parameters.append(Cval)
            names.append(name)
    
    
    elif 'dt' in cl:
        
        max_f = ['sqrt', 'log2', 'auto', None]
        
        for maxf in max_f:
            
            name = cl + '_' + str(maxf) + 'feat'
            classifiers.append(DecisionTreeClassifier(random_state=42, max_features=maxf))
            parameters.append(maxf)
            names.append(name)
            
    elif 'mlp' in cl:
        
        layers = [5, 10, 30, 50]
        
        for lay in layers:
            
            name = cl + '_' + str(lay) + 'layers'
            classifiers.append(MLPClassifier(random_state=42, hidden_layer_sizes=(lay,), max_iter=1000))
            parameters.append(lay)
            names.append(name)
    
    return classifiers, names, parameters

if (__name__=='__main__'):
    
    
    start = time.time()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-F', '--folder', default='.', type=str, help='input folder')
    parser.add_argument('-o', '--out_dir', default='.', type=str, help='output folder')
    parser.add_argument('--clf', choices=['knn', 'svm', 'dt', 'mlp'],  default='knn', type=str, help='classifier')

    args = parser.parse_args()
    
    cl = args.clf
    
    path = os.getcwd()
    
    dataset = read_csv(os.path.join(path, 'data.csv')).replace(['B', 'M'], [0, 1]).values

    #print(dataset.shape)
    
    X = dataset[:, 2:]
    y = np.asarray(dataset[:, 1], dtype=np.uint8)
    
    X = normalize(X, 'max', axis=0)
    
    classifiers, names, parameters = create_clasifiers(cl)
    
    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)  
    
        
    for i in range(len(classifiers)):
        
        print(names[i])
        clf = classifiers[i]
        y_test_all = np.asarray([], dtype=np.uint8)
        y_pred_all = np.asarray([], dtype=np.uint8)
        conf_matrix_sum = np.zeros((2,2), dtype=np.uint16)
        create_csv(path, names[i], cl)
        
        for train_index, test_index in kf.split(X, y):
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred)
            y_test_all = np.concatenate((y_test_all, y_test))
            y_pred_all = np.concatenate((y_pred_all, y_pred))
            conf_matrix_sum = sum_confusion_matrix(conf_matrix, conf_matrix_sum)
            
        
        print(conf_matrix_sum)
        precision, recall, fscore, _ = precision_recall_fscore_support(y_test_all, y_pred_all, average='macro')
        print('Precision: ', precision)
        print('Recall: ', recall)
        print('Fscore: ', fscore)
        print('\n=======================================================\n')
    
        appendrow_csv(path, names[i], cl, precision, recall, fscore, parameters[i], conf_matrix_sum)
            
    
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    t = str(int(hours)) + ':' + str(int(minutes)) + ':' + str(round(seconds, 2))
    print('Elapsed time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))