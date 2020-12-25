import numpy as np
import pandas as  pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from sklearn.multiclass import OneVsRestClassifier


from grid_search import *
from helpers import *


def get_best_model_virus():
    """
    Obtain best classifier for following tasks:
    1. Detect virus type and presence
    _________
    Use recall on train dataset
    Note class imbalance in data, skewed towards not detected
    """
    df = pd.read_csv('csv_outputs/train.csv') #csv with selected features- make sure vlidation and test sets went through preprocessing as well as training
    X_train = df.drop(['Spreader', 'notdetected', 'flue', 'covid', 'measles', 'cmv', 'atRisk'], axis=1)
    y_train = df[['notdetected', 'flue', 'covid', 'measles', 'cmv']]

    knn = customKNeighborsClassifier(3,'uniform')
    log = customKLogisticRegression( penalty='l2', C=1 , solver='lbfgs') # class weights inverse-proportional to relative size in dataset
    svm = customSVM(C=1.0, kernel='rbf', degree=3)
    models = [ knn, log, svm ]
    hyper = [ { 'estimator__n_neighbors': [3,5,7] , 'estimator__weights': ['uniform', 'distance' ]}, {  'estimator__C' : [ 0.1, 0.5, 1, 2, 4, 10], 'estimator__penalty' : [ 'l2', 'none' ] }, \
                                     {  'estimator__C' : [ 0.1, 0.5, 1, 2, 4, 10] , 'estimator__kernel' : ['linear', 'poly', 'rbf', 'sigmoid']} ]
    best = []
    for model,hyper in zip(models,hyper):
        best.append( get_best_ova( model, hyper, 'accuracy', X_train, y_train ) )
    return best


def get_best_model_risk():
    """
    Obtain best classifier for following tasks:
    2. Decide whether the patient is “at risk”
    """
    df = pd.read_csv('csv_outputs/train.csv') #csv with selected features- make sure vlidation and test sets went through preprocessing as well as training
    X_train = df.drop(['Spreader', 'notdetected', 'flue', 'covid', 'measles', 'cmv', 'atRisk'], axis=1)
    y_train = df.atRisk

    knn = customKNeighborsClassifier(3,'uniform')
    log = customKLogisticRegression( penalty='l2', C=1, solver='newton-cg' ) # class weights inverse-proportional to relative size in dataset
    svm = customSVM(C=1.0, kernel='rbf', degree=3)
    models = [ knn, log, svm ]
    hyper = [ { 'n_neighbors': [3,5,7] , 'weights': ['uniform', 'distance' ]}, {  'C' : [ 0.1, 1,  4], 'penalty' : [ 'l2', 'none' ] }, \
                                     {  'C' : [ 0.1,  1, 10] , 'kernel' : [ 'poly',  'sigmoid']} ]

    best = []
    for model,hyper in zip(models,hyper):
        best.append( get_best( model, hyper, 'recall', X_train, y_train ) )
    return best


def get_best_model_spreader():
    """
    Obtain best classifier for following tasks:
    3. Decide whether the patient is a potentially “super-spreader” 
    """
    df = pd.read_csv('csv_outputs/train.csv') #csv with selected features- make sure vlidation and test sets went through preprocessing as well as training
    X_train = df.drop(['atRisk', 'notdetected', 'flue', 'covid', 'measles', 'cmv', 'Spreader'], axis=1)
    y_train = df.Spreader

    knn = customKNeighborsClassifier(3,'uniform')
    log = customKLogisticRegression( penalty='l2', C=1, solver='liblinear' ) # class weights inverse-proportional to relative size in dataset
    svm = customSVM(C=1.0, kernel='rbf', degree=3)
    models = [ knn, log, svm ]
    hyper = [ { 'n_neighbors': [3,5,7] , 'weights': ['uniform', 'distance' ]}, {  'C' : [ 0.1, 1,  4], 'penalty' : [ 'l2'] }, \
                                     {  'C' : [ 0.1,  1, 10] , 'kernel' : [ 'poly',  'sigmoid']} ]

    best = []
    for model,hyper in zip(models,hyper):
        best.append( get_best( model, hyper, 'accuracy', X_train, y_train ) )
    return best

def get_best( clf, hyper, score,  X_train, y_train ):
    """
    Use Grid Search to obtain best parameters
    _____________
    gets classifier, dict with param grid, data
     returns optimal params
    """
    best = customGridSearch( clf, param_grid=hyper, score=score ) #uses cross validation
    best.fit(X_train,y_train)
    return best.best_estimator_

def get_best_ova( clf, hyper, score,  X_train, y_train):
    """
    Use Grid Search to obtain best parameters for multiclass classification
    using One vs. All
    _____________
    gets classifier, dict with param grid, data
    returns optimal params
    """
    ovr = OneVsRestClassifier(clf).fit(X_train,y_train)
    best = customGridSearch( ovr, param_grid=hyper, score=score )
    best.fit( X_train, y_train )
    return best.best_estimator_
    

def get_best_score_virus( clf, score, X_val, y_val ):
    """
     Use recall on validation dataset
    __________
    recieves classifier objects with their best params 
    and outputs best clasifier with params 
    """
 
    return [ score( c.predict(X_val), y_val ) for c in clf ]

def get_best_score_risk():
    """
    Use recall on validation dataset
    __________
    recieves classifier objects with their best params 
    and outputs best clasifier with params 
    """
    df = pd.read_csv( 'csv_outputs/valid.csv' )
    X_val = df.drop(['Spreader', 'notdetected', 'flue', 'covid', 'measles', 'cmv', 'atRisk'], axis=1)
    y_val = df.atRisk

    clf = get_best_model_risk() #get best estimators

    return np.amax(np.array([ recall_score( c.predict(X_val), y_val ) for c in clf ]))

def get_best_score_spreader():
    """
    Use accuracy on validation dataset
    __________
    recieves classifier objects with their best params 
    and outputs best clasifier with params 
    """
    df = pd.read_csv( 'csv_outputs/valid.csv' )
    X_val = df.drop(['Spreader', 'notdetected', 'flue', 'covid', 'measles', 'cmv', 'atRisk'], axis=1)
    y_val = df.Spreader

    clf = get_best_model_spreader() #get best estimators

    return np.amax(np.array([ recall_score( c.predict(X_val), y_val ) for c in clf ]))