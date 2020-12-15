
import pandas as pd                 # data analysis and manipulation tool
import numpy as np                  # Numerical computing tools
import seaborn as sns               # visualization library
import matplotlib.pyplot as plt     # another visualization library

from helpers import *
from grid_search import *



def main():

    virus_df = pd.read_csv("csv_outputs") #csv with selected features- make sure vlidation and test sets went through preprocessing as well as training
    cf = [ 'Spreader', 'atRisk', 'notdetected', 'flue', 'covid', 'measles', 'cmv' ] # binary classification on each 
    models = [  'logistic_regression',  'SVM', 'kNN', 'perceptron', 'decision_tree' ] #models for classification
    hyper = [  ] #hyperparameters for models




if __name__ == "__main__":
    main()