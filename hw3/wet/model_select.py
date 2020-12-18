
import pandas as pd                 # data analysis and manipulation tool
import numpy as np                  # Numerical computing tools
import seaborn as sns               # visualization library
import matplotlib.pyplot as plt     # another visualization library

from helpers import *
from grid_search import *



def main():

    virus_df = pd.read_csv('csv_outputs/train.csv') #csv with selected features- make sure vlidation and test sets went through preprocessing as well as training
    cf = [ 'Spreader', 'atRisk', 'notdetected', 'flue', 'covid', 'measles', 'cmv' ] # binary classification on each 
    models = [  'logistic_regression',  'SVM', 'kNN', 'perceptron', 'decision_tree' ] #models for classification
    hyper = [ ('SVM', { 'kernel' :  ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 'C' : [ 0.5, 1, 2, 4] , 'degree' : [2,3,4,5] }), 
                ('kNN', { 'n_neighbors': [3,5,7] , 'weights': ['uniform', 'distance']})] #hyperparameters for models
                
    
    print("It works!")
                                                                                                         




if __name__ == "__main__":
    main()