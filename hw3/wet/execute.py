import numpy as np
import pandas as  pd

from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


from grid_search import *
from helpers import *


def get_best_model_virus(self, parameter_list):
    """
    Obtain best classifier for following tasks:
    1. Detect virus type and presence
    _________
    Use recall on train dataset
    Note class imbalance in data, skewed towards not detected
    """
    df = pd.read_csv('csv_outputs/train.csv') #csv with selected features- make sure vlidation and test sets went through preprocessing as well as training
    X_train = df.drop(['atRisk', 'Spreader'], axis=1)
    y_train = df[['notdetected', 'flue', 'covid', 'measles', 'cmv']]

    pass
def get_best_model_risk(self, parameter_list):
    """
    Obtain best classifier for following tasks:
    2. Decide whether the patient is “at risk”
    """
    df = pd.read_csv('csv_outputs/train.csv') #csv with selected features- make sure vlidation and test sets went through preprocessing as well as training
    X_train = df.drop(['Spreader', 'notdetected', 'flue', 'covid', 'measles', 'cmv'], axis=1)
    y_train = df.atRisk
    pass
def get_best_model_spreader():
    """
    Obtain best classifier for following tasks:
    3. Decide whether the patient is a potentially “super-spreader” 
    """
    df = pd.read_csv('csv_outputs/train.csv') #csv with selected features- make sure vlidation and test sets went through preprocessing as well as training
    X_train = df.drop(['atRisk', 'notdetected', 'flue', 'covid', 'measles', 'cmv'], axis=1)
    y_train = df.Spreader

    knn = customKNeighborsClassifier(3,'uniform')
    #log = customKLogisticRegression( penalty='l2', C=1 ) # class weights inverse-proportional to relative size in dataset
    #svm = customSVM(C=1.0, kernel='rbf', degree=3)
    models = [ knn ]
    hyper = [ { 'n_neighbors': [3,5,7] , 'weights': ['uniform', 'distance' ]} ] 
               # {  'C' : [ 0.5, 1, 2, 4] },\
                 #{ 'kernel' :  ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 'C' : [ 0.5, 1, 2, 4] , 'degree' : [2,3,4,5] } ]
    best_params = []
    for model,hyper in zip(models,hyper):
        best_params.append( get_best_params( model, hyper, 'accuracy', X_train, y_train ) )
    return best_params

def get_best_params( clf, hyper, score,  X_train, y_train ):
    """
    Use Grid Search to obtain best parameters
    _____________
    gets classifier, dict with param grid, data
     and retruns optimal params

    """
    best = customGridSearch( clf, hyper, score ) #uses cross validation
    best.fit(X_train,y_train)
    return best.best_params_
    

def get_best_score_virus( clf, params, loss ):
    """
     Use recall on validation dataset
    __________
    recieves classifier objects with their best params 
    and outputs best clasifier with params 
    """

    pass
def get_best_score_risk( clf, params, loss ):
    """
    Use recall on validation dataset
    __________
    recieves classifier objects with their best params 
    and outputs best clasifier with params 
    """

    pass
def get_best_score_spreader( clf, params, loss ):
    """
    Use accuracy on validation dataset
    __________
    recieves classifier objects with their best params 
    and outputs best clasifier with params 
    """

    pass