


import pandas as pd                 # data analysis and manipulation tool
import numpy as np                  # Numerical computing tools
import seaborn as sns               # visualization library
import matplotlib.pyplot as plt     # another visualization library

import typing

from utils import *
from feature_handlers import *




def main():

    # 1. load the csv input
    virus_df = pd.read_csv("virus_hw2.csv")
    split_list = [0.75, 0.15, 0.10]

    df = split_the_data(virus_df, split_list)


    # push the dataframe through the pipeline
    # pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
    data_processing_pipe = customPipeline(steps = [('BMI_handler', BMI_handler(max_threshold=50)),
                                                   ('BMI_handler2', BMI_handler(max_threshold=50))])

    df = data_processing_pipe.apply_transforms(df)

    print(df.head(10))


    







if __name__ == "__main__":
    main()

