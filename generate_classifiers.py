# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 20:39:11 2019

@author: Felipe
"""

import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier


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
        
        max_f = ["sqrt", "log2", "auto", None]
        
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

    with open('classifiers_list_' + cl + '.pickle', 'wb') as handle:

        pickle.dump([classifiers, names, parameters], handle)