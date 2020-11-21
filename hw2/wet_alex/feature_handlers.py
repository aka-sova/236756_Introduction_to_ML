

import sklearn
from sklearn.pipeline import Pipeline

# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC

import pandas as pd
import numpy as np

from utils import customPipeline, CustomFeatureHandler

"""
Put all the operators on the features here
Implement in a pipeline as in the 'data_preparation.py' file
"""


class BMI_handler(CustomFeatureHandler):
    """Apply transformation on the BMI parameters"""

    def __init__(self, max_threshold: int):
        self.max_threshold = max_threshold

    def transform(self, df_dut : pd.DataFrame()):

        mean_train_BMI = np.mean(df_dut.BMI)
        outlier_bmi_mask = df_dut.BMI > self.max_threshold

        df_dut.loc[outlier_bmi_mask, "BMI"] = mean_train_BMI
        return df_dut


class PCR_results_handler(CustomFeatureHandler):
    """From the analysis, the PCR results 3, 12, and 16 can be removed"""
    def __init__(self):
        super().__init__()

    def transform(self, df_dut : pd.DataFrame()):

        pcr_results_to_remove = [3, 12, 16]
        fields_to_drop = ["pcrResult" + str(i) for i in pcr_results_to_remove]

        for field_to_drop in fields_to_drop:
            df_dut = df_dut.drop(field_to_drop, axis=1)

        return df_dut