from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid

import pandas as pd



class customGridSearch(GridSearchCV):
    """return dataframe of params of best estimator"""

    def __init__(self):
        super().__init__()

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

