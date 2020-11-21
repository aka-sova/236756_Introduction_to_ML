

import sklearn
from sklearn.pipeline import Pipeline

# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC

import pandas as pd
import numpy as np

from utils import customPipeline, CustomFeatureHandler




class BMI_handler(CustomFeatureHandler):
    """Apply transformation on the BMI parameters

    Any custom transform function has to include the "transform" function
    which takes and returns dataframe
    """
    def __init__(self, max_threshold: int):
        self.max_threshold = max_threshold

    def transform(self, df_dut : pd.DataFrame()):
        return df_dut