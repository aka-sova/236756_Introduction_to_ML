from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np


class customGridSearch(GridSearchCV):
    """return dataframe of params of best estimator"""

    def __init__(self,estimator,param_grid):
        super().__init__(estimator, param_grid, scoring=None, n_jobs=None, iid='deprecated',\
                         refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs',\
                         error_score=np.nan, return_train_score=False)

    def best(self): #returns dataframe rather than dict
        best_ = self.best_estimator_
        return pd.DataFrame(best_)


class customParameterGrid(ParameterGrid):
    """ return list of dictionaries """

    def __init__(self):
        super().__init__()

    def params_(self):
        params  = list( self )
        return params

class customSVM(SVC):
    """ svm from sklearn """

    def __init__(self, c , kernel, degree):
        super().__init__(  C=c, kernel=kernel, degree=degree, gamma='scale', coef0=0.0, shrinking=True,\
         probability=False, tol=0.001, cache_size=200, class_weight='balanced', verbose=False, max_iter=-1,#change class_weight?\ 
          decision_function_shape='ovr', break_ties=False, random_state=33)
        
class customKNeighborsClassifier(KNeighborsClassifier):
    """ configure constant params"""
    def __init__(self, n_neighbors, weights):
        super().__init__(n_neighbors=n_neighbors , weights=weights)

class customKLogisticRegression(LogisticRegression):
    """ configure constant params"""
    def __init__(self, n_neighbors, weights):
        super().__init__(penalty='l2',  dual=False, tol=0.0001, C=1.0,\
         fit_intercept=True, intercept_scaling=1, class_weight=None, \
          random_state=None, solver='lbfgs', max_iter=100, \
          multi_class='ovr', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None) #multi_class ovr is binary classification







        