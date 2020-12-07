import pandas as pd                 # data analysis and manipulation tool
import numpy as np                  # Numerical computing tools
import seaborn as sns               # visualization library
import matplotlib.pyplot as plt     # another visualization library


from sklearn import decomposition
from sklearn.preprocessing import StandardScaler



def learn_pcr_transform(df : pd.DataFrame, n_components : int):

    sc = StandardScaler()
    pca = decomposition.PCA(n_components=n_components)

    # make dataframe only with pcr results
    pcr_results_to_remove = range(17)
    fields_to_remain = ["pcrResult" + str(i) for i in pcr_results_to_remove]

    for column in df.columns:
        if column not in fields_to_remain:
            df = df.drop(column, axis=1)

    # drop rows where at least 1 data is NA
    df = df.dropna(axis=0)

    # learn scaler on the train dataset
    df_scaled = sc.fit_transform(df)

    # find the PCA components
    pca.fit_transform(df_scaled)


    return sc, pca

def get_pca_transform(df : pd.DataFrame, n_components : int):

    sc = StandardScaler()
    pca = decomposition.PCA(n_components=n_components)

    df = sc.fit_transform(df)
    pca.fit_transform(df)


def get_pca_columns(df : pd.DataFrame):

    pca = decomposition.PCA(n_components=5)
    pca.fit_transform(df)  # plug in scaled values ( with outliers )
    V = pca.components_

    cov = np.zeros((5, 4))
    for i in range(5):
        sort = np.sort(np.absolute(V[i]))
        for j in range(4):
            cov[i][j] = sort[15 - j]
    cov_idx = np.zeros((5, 4))

    for i in range(5):
        where = [(idx + 1) for idx, item in enumerate(V[i]) if
                 np.absolute(item) >= cov[i][3]]  # indices of 3 maximal coefficients
    for j in range(4):
        cov_idx[i][j] = where[j]

    keep = [int(cov_idx[i][j]) for i in range(3) for j in range(4)]
    keep_cols = ['pcrResult{}'.format(itr) for itr in keep]

    return keep_cols