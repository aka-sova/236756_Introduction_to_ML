
import os
import random
import pandas as pd # data analysis and manipulation tool
import numpy as np # Numerical computing tools
import seaborn as sns  # visualization library
import matplotlib.pyplot as plt  # another visualization library

from scipy import stats
from copy import copy
from tabulate import tabulate
from sklearn.model_selection import train_test_split


from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import davies_bouldin_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import silhouette_samples

from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn import metrics

from app_functions import *

def main ():

    # ----------------------------------
    # DEFINITIONS

    imputation_method = "KNN"   #  use ["KNN", "MEAN"]
    clustering_method = "GMM" # use ["KMEANS", "SPECTRAL", "GMM"]
    features_selection_method = "SFS"  # use ["MULTICLASS", "MUTUAL_INFO", "PCA_IMPORTANCE", "SFS"]

    # additional settings
    silhouette_metrics_list = ['l1', 'l2', 'cosine']
    multiclass_models = [LogisticRegression(max_iter=10000), DecisionTreeClassifier(), ExtraTreesClassifier()]
    sfs_model = KNeighborsClassifier(n_neighbors=3)  # use KNeighborsClassifier(n_neighbors=3) / DecisionTreeClassifier(random_state=0)


    # ----------------------------------
    # APPLICATION

    print(f"IMPUTATION METHOD: {imputation_method}")
    print(f"CLUSTERING METHOD: {clustering_method}")
    print(f"FEATURES SELECTION METHOD: {features_selection_method}")

    random.seed(42)
    out_dirname = "outputs"
    os.makedirs(out_dirname,exist_ok=True)

    # load the data
    df = pd.read_csv('protein.csv', sep=',', header=0)

    # imputation
    df = impute_data(imputation_method, df, out_dirname)

    # do the clustering
    out_dirname = os.path.join(out_dirname, clustering_method)
    labels = do_clustering(clustering_method, silhouette_metrics_list, df, out_dirname)

    # save the clustering result
    out_dirname = os.path.join(out_dirname, features_selection_method)
    best_features = do_feature_selection(features_selection_method= features_selection_method,
                                         multiclass_models = multiclass_models,
                                         sfs_model = sfs_model,
                                         labels = labels,
                                         df = df,
                                         out_dir = out_dirname)


    find_most_prevalent_mutations(labels = labels, num_mutations=3, out_dir = out_dirname)

    print_selected_proteins(best_features)
    print_labeled_data(labels)


    print("Finito")



if __name__ == "__main__":
    main()