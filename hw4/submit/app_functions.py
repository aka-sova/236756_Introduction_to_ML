

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




def draw_scatter(dims : str, reduced_data, labels, output_name : str):
    # dims : "2D", "3D"

    colors = ('b', 'g', 'r', 'c', 'm')

    if dims == "2D":

        fig, ax = plt.subplots()
        for idx in range(5):
            color = colors[idx]

            class_idx = [index for index, val in enumerate(labels) if val == idx]

            x = reduced_data[class_idx, 0]
            y = reduced_data[class_idx, 1]

            ax.scatter(x, y, c=color, s=10, label=f"mut_{idx}",
                       alpha=0.8, edgecolors='none')

        ax.legend()
        ax.grid(True)

    elif dims == "3D":

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for idx in range(5):
            color = colors[idx]

            class_idx = [index for index, val in enumerate(labels) if val == idx]

            x = reduced_data[class_idx, 0]
            y = reduced_data[class_idx, 1]
            z = reduced_data[class_idx, 2]

            ax.scatter(x, y, z, c=color, label=f"mut_{idx}", s=3)

        ax.legend()
        ax.grid(True)


    plt.savefig(output_name)


def draw_silhouette_graph(X, y, output_name : str):

    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    # Get silhouette samples
    silhouette_vals = silhouette_samples(X, y)
    # avg silhouette score for each cluster
    avg_cluster = []

    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(y)):
        cluster_silhouette_vals = silhouette_vals[y == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)
        avg_cluster.append(np.mean(cluster_silhouette_vals))

    # Get the average silhouette score and plot it
    avg_score = np.mean(silhouette_vals)
    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
    ax1.set_yticks([])
    ax1.set_xlim([-0.1, 1])
    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster labels')
    ax1.set_title('Silhouette plot for the various clusters', y=1.02);

    plt.savefig(output_name)

def impute_data(imputation_method : str, df, out_dirname):


    if imputation_method == "MEAN":

        protein_keys = [f"protein_{idx+1}" for idx in range(10)]
        rename_dict = {x:i+1 for i,x in enumerate(protein_keys)}
        df = df.rename(columns=rename_dict)

        # threshold sets the limit = std*threshold, beyond which the outliers are detected
        threshold = 2.5

        for key in list(rename_dict.values()):

            new_df = df[['ID', key]].copy()

            # remove nans
            new_df = new_df.dropna()

            IDs_list = list(new_df.ID)

            z = np.abs(stats.zscore(new_df[key]))
            outliers_arr = np.where(z > threshold)

            if len(outliers_arr[0]) != 0:


                # printing those outlier values
                for outlier_idx in outliers_arr[0]:
                    # dropping those rows from the new_df
                    new_df = new_df.drop(IDs_list[outlier_idx])

            else:
                pass  # no outliers were detected

            # find the mean WITHOUT the outliers
            new_mean = new_df[key].mean()

            # fill the outliers in the original df with mean
            for outlier_idx in outliers_arr[0]:
                # dropping those rows from the new_df
                df.at[IDs_list[outlier_idx], key] = new_mean

        df = df.fillna(df.mean())
        df = df.drop(['ID'], axis=1)

        df.to_csv(os.path.join(out_dirname,'protein_fixed_mean.csv'), sep=',', header=True, index=False)

    elif imputation_method == "KNN":

        # REPLACE OUTLIERS WITH NAN
        protein_keys = [f"protein_{idx+1}" for idx in range(10)]
        rename_dict = {x:i+1 for i,x in enumerate(protein_keys)}
        df = df.rename(columns=rename_dict)

        for key in list(rename_dict.values()):

            new_df = df[['ID', key]].copy()

            # remove nans
            new_df = new_df.dropna()

            IDs_list = list(new_df.ID)

            z = np.abs(stats.zscore(new_df[key]))
            outliers_arr = np.where(z > 2.5)

            if len(outliers_arr[0]) != 0:


                # printing those outlier values
                for outlier_idx in outliers_arr[0]:
                    # dropping those rows from the new_df
                    new_df = new_df.drop(IDs_list[outlier_idx])

            else:
                pass  # no outliers were detected

            # find the mean WITHOUT the outliers
            new_mean = new_df[key].mean()

            # fill the outliers in the original df with mean
            for outlier_idx in outliers_arr[0]:
                # dropping those rows from the new_df
                df.at[IDs_list[outlier_idx], key] = np.nan


        # IMPUTER using KNN
        df = df.drop(['ID'], axis=1)
        imputer = KNNImputer(n_neighbors=3)  # k-nearest neighbors impute,

        df_mx = imputer.fit_transform(df)
        df = pd.DataFrame(df_mx)
        df = df.rename(columns={0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10})


        df.to_csv(os.path.join(out_dirname,'protein_fixed_knn.csv'), sep=',', header=True, index=False)

    return df


def do_clustering(clustering_method, silhouette_metrics_list, df, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    scores = {}

    X = df.values
    reduced_data2d = PCA(n_components=2).fit_transform(X)
    reduced_data3d = PCA(n_components=3).fit_transform(X)




    if clustering_method == "KMEANS":

        kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
        labels = kmeans.labels_


    elif clustering_method == "SPECTRAL":

        clustering_virus = SpectralClustering(n_clusters=5,
                                              affinity='nearest_neighbors',
                                              assign_labels="discretize",
                                              random_state=0).fit(X)

        labels = clustering_virus.labels_


    elif clustering_method == "GMM":

        gm = GaussianMixture(n_components=5, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1,
                             init_params='kmeans', weights_init=None, means_init=None, precisions_init=None,
                             random_state=None, warm_start=False, verbose=0, verbose_interval=10)

        gm = gm.fit(X)
        labels = gm.predict(X)  # labels


    draw_scatter('2D', reduced_data2d, labels, os.path.join(out_dir, "2D_PCA_Scatter.png"))
    draw_scatter('3D', reduced_data3d, labels, os.path.join(out_dir, "3D_PCA_Scatter.png"))

    draw_silhouette_graph(X, labels, os.path.join(out_dir, "Silhouette_graph.png"))


    for metric in silhouette_metrics_list:
        scores[f"silhouette_{metric}"] = metrics.silhouette_score(X, labels, metric=metric)

    scores["db_score"] = davies_bouldin_score(X, labels)

    # print all the scores
    with open(os.path.join(out_dir, "scores.txt"), 'w') as fd:
        for score in scores.keys():
            fd.write(f"Score: {str(score)} : {scores[score]}\n")

    return labels


def do_feature_selection(features_selection_method, multiclass_models, sfs_model, labels, df, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    if features_selection_method == "MULTICLASS":

        X = df.values

        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.15, random_state=42)

        models_amount = len(multiclass_models)
        features_amount = X.shape[1]
        support_mx = np.zeros((models_amount, features_amount))
        ranking_mx = np.zeros((models_amount, features_amount))
        acc_mx = np.zeros((models_amount, 1))

        for idx, model in enumerate(multiclass_models):
            # create the RFE model and select 5 attributes
            rfe = RFE(model, n_features_to_select=5)
            rfe = rfe.fit(X_train, y_train)

            support_mx[idx, :] = rfe.support_
            ranking_mx[idx, :] = rfe.ranking_

            y_pred = rfe.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            acc_mx[idx] = acc

        with open(os.path.join(out_dir, "multiclass_report.txt"), 'w') as fd:

            headers = [f"feature_{idx}" for idx in range(10)]

            rowIDs = ["LR", "DTC", "ETC"]
            table = tabulate(support_mx, headers, showindex=rowIDs)
            fd.write("\n\nSUPPORT MATRIX. 1 = Take this feature\n")
            fd.write(table)

            table = tabulate(ranking_mx, headers, showindex=rowIDs)
            fd.write("\n\nRANKING MATRIX. 1 = Good feature!\n")
            fd.write(table)

            table = tabulate(acc_mx, ['Accuracy'], showindex=rowIDs)
            fd.write("\n\nAccuracy on test set\n")
            fd.write(table)

            # choosing the 5 best best features according to the best classifier
            best_classifier_idx = np.argmax(acc_mx)
            best_features = np.where(support_mx[best_classifier_idx, :] == 1)

            fd.write("\nbest_features_list:")
            fd.write(str(list(best_features)))



    elif features_selection_method == "MUTUAL_INFO":

        X = df.values

        features_amount = X.shape[1]
        MI_scores = mutual_info_classif(X, labels)

        MI_scores_mx = np.zeros((1, features_amount))
        MI_scores_srt_mx = np.zeros((1, features_amount))

        MI_scores_mx[0, :] = MI_scores
        sorted_idx_list = np.argsort(MI_scores)

        for srt_idx in range(10):
            MI_scores_srt_mx[0, sorted_idx_list[srt_idx]] = srt_idx

        headers = [f"feature_{idx}" for idx in range(10)]
        with open(os.path.join(out_dir, "mutual_info_report.txt"), 'w') as fd:

            table = tabulate(MI_scores_mx, headers, floatfmt=".4f")
            fd.write("\n\nMutual Information\n")
            fd.write(table)

            table = tabulate(MI_scores_srt_mx, headers, floatfmt=".0f")
            fd.write("\n\nMutual Information sorted by value\n")
            fd.write(table)

            # sorted_features_list = sorted(range(10),key=MI_scores.__getitem__)
            sorted_features_list = [i[0] for i in sorted(enumerate(MI_scores), key=lambda x: x[1])]

            fd.write("\nHighest MI features:")
            fd.write(str(sorted_features_list))

            fd.write("\nbest_features_list:")
            fd.write(str(sorted_features_list[5:10]))

        best_features = sorted_features_list[5:10]



    elif features_selection_method == "PCA_IMPORTANCE":

        pca = PCA(n_components=5)
        pca.fit_transform(df)
        comp = pd.DataFrame(np.abs(pca.components_))
        comp = comp.rename(columns={0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10})

        comp.to_csv(os.path.join(out_dir, "pca_importance.csv"))

        # after analyzing the table as in the report, we choose:
        best_features = [4, 5 ,6, 1, 2]


    elif features_selection_method == "SFS":

        sfs_forward = SFS(sfs_model,
                          k_features=5,
                          forward=True,
                          floating=False,
                          verbose=0,
                          scoring='accuracy',
                          cv=5)

        sfs_forward = sfs_forward.fit(df, labels)

        best_features = list(sfs_forward.subsets_[5]["feature_names"])

    return best_features



def find_most_prevalent_mutations(labels, num_mutations, out_dir):

    np.unique(labels)
    num_samples = labels.shape[0]
    class_mx = np.zeros((5, 1))

    for class_test in range(5):
        class_sum = len([i for i in range(num_samples) if labels[i] == class_test])
        class_mx[class_test, 0] = class_sum

    muts_list = [f"Mutation {idx}" for idx in range(5)]
    table = tabulate(class_mx, floatfmt=".0f", headers=["Num cases"], showindex=muts_list)

    with open(os.path.join(out_dir, "mutations_to_treat.txt"), 'w') as fd:

        fd.write("\n\nNumber of cases for each mutation\n")
        fd.write(table)
        fd.write(f"\n\nMost prevalent {num_mutations} mutations")

        sorted_mutations = [i[0] for i in sorted(enumerate(class_mx[:,0]), key=lambda x: x[1])]

        fd.write("\nSorted mutations amount:")
        fd.write(str(sorted_mutations))

        fd.write("\nMutations to treat:")
        mutations_total_num = len(sorted_mutations)
        fd.write(str(sorted_mutations[num_mutations-1:mutations_total_num]))


def print_selected_proteins(best_features):

    with open("selected_proteins.txt", 'w') as fd:
        for feature in best_features:
            fd.write(f"'protein_{feature}'\n")

def print_labeled_data(labels):

    with open("clusters.csv", 'w') as fd:

        fd.write("ID,y\n")

        for i, label in enumerate(labels):
            fd.write(f"{i},{label}\n")

