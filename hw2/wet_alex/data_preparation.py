


import pandas as pd                 # data analysis and manipulation tool
import numpy as np                  # Numerical computing tools
import seaborn as sns               # visualization library
import matplotlib.pyplot as plt     # another visualization library


from utils import *
from feature_handlers import *
from dataset_operations import *




def main():

    # 1. load the csv input
    virus_df = pd.read_csv("virus_hw2.csv")
    split_list = [0.75, 0.15, 0.10]

    df = split_the_data(virus_df, split_list)

    # PCR results learning
    pcr_scaler, pcr_pca = learn_pcr_transform(df["train"], n_components=5)


    # push the dataframe through the pipeline
    data_processing_pipe = customPipeline(steps = [('Drop_Irrelevant', Drop_Irrelevant()),
                                                   ('PCR_results_handler', PCR_results_handler(pcr_scaler, pcr_pca)),
                                                   ('SexHandler', SexHandler()),
                                                   ('BMI_handler', BMI_handler(max_threshold=50)),
                                                   ('BloodTypeHandler', BloodTypeHandler()),
                                                   ('SelfDeclaration_to_Categories', SelfDeclaration_to_Categories()),
                                                   ('Modify_Results_Code', Modify_Results_Code()),
                                                   ('DropNA', DropNA()),
                                                   ])

    # data_processing_pipe = customPipeline(steps = [('BMI_handler', BMI_handler(max_threshold=50)),
    #                                                ('PCR_results_handler', PCR_results_handler())])

    # apply all the transforms one by one
    df["train"] = data_processing_pipe.apply_transforms(df["train"])
    df["valid"] = data_processing_pipe.apply_transforms(df["valid"])
    df["test"] = data_processing_pipe.apply_transforms(df["test"])


    print(df["train"].head(10))
    

    save_csv_files(df, "csv_outputs")



def main2():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
    dataset = pd.read_csv(url, names=names)


    X = dataset.drop('Class', 1)
    y = dataset['Class']

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

if __name__ == "__main__":
    main()
