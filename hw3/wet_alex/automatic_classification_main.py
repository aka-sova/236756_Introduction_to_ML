
import pandas as pd                 # data analysis and manipulation tool
import numpy as np                  # Numerical computing tools


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

import sklearn.metrics



from data_preparation import *
from auto_classification import *



def init_automatic_classification(regenerate_features : bool, evaluate_on_test : bool, dataset_type : str):

    cur_dir = os.getcwd()
    outputs_folder_path = os.path.abspath(os.path.join(cur_dir, 'outputs_clf'))
    os.makedirs(outputs_folder_path, exist_ok = True)
    logfd = open(os.path.join(outputs_folder_path, 'log.txt'), 'w')

    # 1. define general parameters features using the hw2 features handling
    features_folder_path = 'outputs_csv'
    virus_dataset_path = os.path.join('input_ds', 'virus_hw2.csv')
    split_list = [0.75, 0.15, 0.10] # train valid test
    targets_mappings = get_targets_mappings()




    # 1. Define each task

    tasks = []

    if dataset_type == 'virus':
        tasks.append(Task(task_name = 'Spreader_Detection',
                          target_type = 'Spreader',
                          main_metrics = metrics.f1_score,
                          pipeline = spread_pipe(dataset_path=virus_dataset_path,
                                               split_list=[0.75, 0.15, 0.10]),
                          mapping_dict = targets_mappings['spreader_mapping'],
                          order  = 1))

        tasks.append(Task(task_name = 'At_Risk_Detection',
                          target_type = 'atRisk',
                          main_metrics = metrics.accuracy_score,
                          pipeline=risk_pipe(dataset_path=virus_dataset_path,
                                             split_list=[0.75, 0.15, 0.10]),
                          mapping_dict = targets_mappings['at_risk_mapping'],
                          order  = 2))

        tasks.append(Task(task_name = 'Disease_Detection',
                          target_type = 'Disease',
                          main_metrics = metrics.accuracy_score,
                          pipeline=pca_srs_pipe(dataset_path=virus_dataset_path,
                                                split_list=[0.75, 0.15, 0.10]),
                          mapping_dict = targets_mappings['disease_mapping'],
                          order  = 0))

    else:
        # iris
        #       we use the iris dataset to verify the correctness of the algorithms
        tasks.append(Task(task_name = 'Iris type detection',
                          target_type = 'Iris_targets',
                          main_metrics = metrics.accuracy_score))



    # 2. Create the features datasets according to specified pipeline for each task
    #       create the dataset based on the user choice.
    #       we are using the sklearn.utils.Bunch object for this, which is actually a dict
    #       we need it to have 'data', 'feature_names',
    #                           'target_types' (has dict for every target type of 'target_names', 'targets')



    os.makedirs(features_folder_path, exist_ok=True)
    for task in tasks:

        if regenerate_features == True:
            prepare_dataset(dataset_path=virus_dataset_path,
                            split_list = split_list,
                            data_processing_pipe = task.pipeline,
                            output_folder_name = os.path.join(features_folder_path, task.task_name))

        task.datasets['train'], task.datasets['valid'], task.datasets['test'] = \
                make_datasets(dataset_type, os.path.join(features_folder_path, task.task_name), targets_mappings)




    # 3.2 Define the models that should be tried
    #          use the tuple (model, parameters)
    #          those models are later inserted into the cross validation,

    models = []
    models.append((KNeighborsClassifier(), {'n_neighbors':[3, 5, 10, 20, 50]}))
    # models.append((SVC(), {'kernel':('linear', 'rbf'), 'C':[1, 10]}))
    # models.append((SVC(), {'kernel': ('linear'), 'C': [1, 10]}))
    models.append((GaussianNB(), {}))
    models.append((DecisionTreeClassifier(), {'max_depth':[5, 10, 15, 20]}))
    # models.append((LogisticRegression(), {'max_iter': [1000, 3000, 10000]}))

    # 3.3 Choose the metrics for validation to display
    validation_metrics = []
    validation_metrics.append((metrics.accuracy_score, {}))
    validation_metrics.append((metrics.precision_score, {'average' : 'weighted'}))
    validation_metrics.append((metrics.f1_score, {'average' : 'weighted'}))
    validation_metrics.append((metrics.recall_score, {'average' : 'weighted'}))


    # 3.4 Find best model for each task according to the validation dataset
    chosen_models_dict = choose_best_model(tasks, models,
                                           validation_metrics, outputs_folder_path, logfd)

    # 3.5 print nicely
    print_best_task_models(chosen_models_dict, tasks, models, logfd)

    if dataset_type == 'virus' and evaluate_on_test == True:
        # 3.6 Check scores of the best model on the test dataset
        predicted_out_path = os.path.join(outputs_folder_path, 'test_predicted.csv')
        test_best_model(tasks, models, chosen_models_dict, predicted_out_path, logfd)


        # # 3.7 Check scores of the best model on the unseen dataset, print into 'predicted' file
        #       we need to create dataset for each task as previously


        external_datasets, patient_IDs = get_external_datasets(os.path.join('input_ds', 'virus_hw3_unlabeled.csv'),
                                                               tasks,
                                                               targets_mappings,
                                                               os.path.join('input_ds', 'virus_hw3_unlabeled'))

        predicted_out_path = os.path.join(outputs_folder_path, 'unseen_predicted.csv')
        test_best_model(tasks, models, chosen_models_dict, predicted_out_path, logfd, patient_IDs,
                        external_datasets, True)


    logfd.close()




if __name__ == "__main__":

    # dataset_type = virus / iris
    init_automatic_classification(regenerate_features = True,
                                  evaluate_on_test = True,
                                  dataset_type = 'virus')


