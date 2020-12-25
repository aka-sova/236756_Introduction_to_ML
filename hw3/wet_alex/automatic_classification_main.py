
import pandas as pd                 # data analysis and manipulation tool
import numpy as np                  # Numerical computing tools


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.utils import Bunch


from data_preparation import *
from auto_classification import *



def init_automatic_classification(regenerate_features : bool, dataset_type : str):

    cur_dir = os.getcwd()
    outputs_folder_path = os.path.abspath(os.path.join(cur_dir, 'outputs_clf'))



    # 1. prepare the virus features using the hw2 features handling
    features_folder_path = 'outputs_csv'
    virus_dataset_path = os.path.join('input_ds', 'virus_hw2.csv')

    if regenerate_features == True or os.path.isdir(features_folder_path) == False:
        prepare_dataset(dataset_path=virus_dataset_path,
                        split_list = [0.75, 0.15, 0.10],
                        output_folder_name = features_folder_path)


    # 2. create the dataset based on the user choice.
    #       we use the iris dataset to verify the correctness of the algorithms
    #
    #       we are using the sklearn.utils.Bunch object for this, which is actually a dict
    #       we need it to have 'data', 'feature_names',
    #                           'target_types' (has dict for every target type of 'target_names', 'targets')

    if dataset_type == 'virus':

        dataset_virus = pd.read_csv(os.path.join(features_folder_path, 'train.csv'))
        disease_mapping = {'flue': 0, 'covid': 1, 'cmv': 2, 'cold': 3, 'measles': 4, 'notdetected': 5}
        spreader_mapping = {'NotSpreader' : 0, 'Spreader' : 1}
        at_risk_mapping = {'NotAtRisk': 0, 'atRisk': 1}


        train_dataset = Bunch()
        train_dataset.filename = os.path.join(features_folder_path, 'train.csv')
        train_dataset.target_types = {}

        # target_type = 0, Disease
        train_dataset.target_types['Disease'] = { 'target_names' : list(disease_mapping.keys()),
                                                       'targets' : dataset_virus['Disease'].to_numpy()}

        train_dataset.target_types['Spreader'] = { 'target_names' : list(spreader_mapping.keys()),
                                                       'targets'  : dataset_virus['Spreader'].to_numpy()}

        train_dataset.target_types['atRisk'] = { 'target_names' : list(at_risk_mapping.keys()),
                                                       'targets'  : dataset_virus['atRisk'].to_numpy()}


        # drop all the targets from the train dataframe
        for target_type in train_dataset.target_types.keys():
            dataset_virus = dataset_virus.drop(columns=target_type)

        # all that's left are features = data
        train_dataset.data = dataset_virus.to_numpy()
        train_dataset.feature_names = dataset_virus.columns.to_list()

        print("Dataset ready")


    else:
        iris_dataset = datasets.load_iris()

        train_dataset = Bunch()
        train_dataset.data = iris_dataset.data
        train_dataset.feature_names = iris_dataset.feature_names
        train_dataset.filename = iris_dataset.filename
        train_dataset.target_types = {}

        # add target type
        train_dataset.target_types['Iris_targets'] = { 'target_names' : iris_dataset.target_names ,
                                                       'targets' : iris_dataset.target}



    # 3. Find the best classifier for each task

    # 3.1 Define each task
    tasks = []

    if dataset_type == 'virus':
        tasks.append(Task(task_name = 'Spreader Detection',
                          target_type = 'Spreader',
                          main_metrics = metrics.accuracy_score))

        tasks.append(Task(task_name = 'At Risk Detection',
                          target_type = 'atRisk',
                          main_metrics = metrics.accuracy_score))

        tasks.append(Task(task_name = 'Disease Detection',
                          target_type = 'Disease',
                          main_metrics = metrics.accuracy_score))

    else:
        # iris
        tasks.append(Task(task_name = 'Iris type detection',
                          target_type = 'Iris_targets',
                          main_metrics = 'precision'))


    # 3.2 Define the models that should be tried
    #          use the tuple (model, parameters)
    #          those models are later inserted into the cross validation,

    models = []
    models.append((KNeighborsClassifier(), {'n_neighbors':[3, 5, 10]}))
    models.append((SVC(), {'kernel':('linear', 'rbf'), 'C':[1, 10]}))
    # models.append((SVC(), {'kernel': ['linear'], 'C': [1, 10]}))
    models.append((DecisionTreeClassifier(), {'max_depth':[5, 10, 15]}))

    # 2.3 Find best model for each task
    chosen_models_dict = choose_best_model(tasks, models, train_dataset, outputs_folder_path)

    # 2.4 print nicely
    print_best_task_models(chosen_models_dict, tasks, models)

    # 2.5 Make classification on the unseen data





if __name__ == "__main__":

    # dataset_type = virus / iris

    init_automatic_classification(regenerate_features = False, dataset_type = 'iris')


