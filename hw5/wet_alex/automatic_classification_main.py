import pandas as pd  # data analysis and manipulation tool
import numpy as np  # Numerical computing tools

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier,VotingClassifier


import sklearn.metrics

from data_preparation import *
from auto_classification import *


def init_automatic_classification(regenerate_features: bool, evaluate_on_test: bool, dataset_type: str):
    cur_dir = os.getcwd()
    outputs_folder_path = os.path.abspath(os.path.join(cur_dir, 'outputs_clf'))
    os.makedirs(outputs_folder_path, exist_ok=True)
    logfd = open(os.path.join(outputs_folder_path, 'log.txt'), 'w')

    # 1. define general parameters features using the hw2 features handling
    features_folder_path = 'outputs_csv'
    virus_dataset_path = os.path.join('input_ds', 'virus_hw5.csv')
    split_list = [0.85, 0.10, 0.05]  # train valid test
    targets_mappings = get_targets_mappings()

    # 1. Define each task

    tasks = []

    if dataset_type == 'virus':
        tasks.append(Task(task_name='Spreader_Detection',
                          target_type='Spreader',
                          main_metrics=metrics.recall_score,
                          pipeline=given_features_pipe(dataset_path=virus_dataset_path,
                                                       split_list=split_list),
                          mapping_dict=targets_mappings['spreader_mapping'],
                          order=1,
                          output_column_name="Spreader"))

        tasks.append(Task(task_name='At_Risk_Detection',
                          target_type='atRisk',
                          main_metrics=metrics.f1_score,
                          pipeline=given_features_pipe(dataset_path=virus_dataset_path,
                                                       split_list=split_list),
                          mapping_dict=targets_mappings['at_risk_mapping'],
                          order=2,
                          output_column_name="Risk"))

        tasks.append(Task(task_name='Disease_Detection',
                          target_type='Disease',
                          main_metrics=metrics.accuracy_score,
                          pipeline=given_features_pipe(dataset_path=virus_dataset_path,
                                                       split_list=split_list),
                          mapping_dict=targets_mappings['disease_mapping'],
                          order=0,
                          output_column_name="Virus"))

    else:
        # iris
        #       we use the iris dataset to verify the correctness of the algorithms
        tasks.append(Task(task_name='Iris type detection',
                          target_type='Iris_targets',
                          main_metrics=metrics.accuracy_score))

    # 2. Create the features datasets according to specified pipeline for each task
    #       create the dataset based on the user choice.
    #       we are using the sklearn.utils.Bunch object for this, which is actually a dict
    #       we need it to have 'data', 'feature_names',
    #                           'target_types' (has dict for every target type of 'target_names', 'targets')

    os.makedirs(features_folder_path, exist_ok=True)
    for task in tasks:

        if regenerate_features == True:
            prepare_dataset(dataset_path=virus_dataset_path,
                            split_list=split_list,
                            data_processing_pipe=task.pipeline,
                            output_folder_name=os.path.join(features_folder_path, task.task_name))

        task.datasets['train'], task.datasets['valid'], task.datasets['test'] = \
            make_datasets(dataset_type, os.path.join(features_folder_path, task.task_name), targets_mappings)

    # 3 Define the models that should be tried
    #          use the tuple (model, parameters, use_cv)
    #          those models are later inserted into the cross validation,
    #          if parameters = {}, and the model is not MLP, set use_cv=True, to use the CV anyway.

    models = []
<<<<<<< Updated upstream
    models.append((KNeighborsClassifier(), {'n_neighbors':[3, 5, 10, 20, 50]}))
    models.append((LinearSVC(max_iter=1000), {} , True))
    # models.append((GaussianNB(), {}, True))
    # models.append((DecisionTreeClassifier(), {'max_depth':[5, 10, 15, 20]}))
=======
    # models.append((KNeighborsClassifier(), {'n_neighbors':[3, 5, 10, 20, 50]}))
    # models.append((LinearSVC(max_iter=1000), {}))
    # models.append((GaussianNB(), {}))
    models.append((DecisionTreeClassifier(), {'max_depth':[5, 10, 15, 20 ,50]}, True))
>>>>>>> Stashed changes
    # models.append((LogisticRegression(max_iter=1000), {}))
<<<<<<< HEAD
    # models.append((OneVsRestClassifier(DecisionTreeClassifier(max_depth=5)), {}))
    models.append((MLPClassifier(alpha=0.0001, max_iter=100000, hidden_layer_sizes = (100,500,500,100),
                                  solver='sgd', momentum=0.9, early_stopping=True), {}))
    #models.append((RandomForestClassifier(max_depth=3, n_estimators=500, max_features=10), {}))

    models.append((AdaBoostClassifier(), {'n_estimators': [50, 100],
<<<<<<< Updated upstream
                                          'base_estimator': [DecisionTreeClassifier(max_depth=1)]}))
    models.append( (VotingClassifier(estimators=[('mlp', MLPClassifier(alpha=0.0001, max_iter=100000, hidden_layer_sizes = (100,100,100))),
    ('rf', RandomForestClassifier(max_depth=3, n_estimators=500, max_features=10)),
    ('ada', AdaBoostClassifier(n_estimators=100, base_estimator=DecisionTreeClassifier(max_depth=5))) ],voting='hard'), {} ) )
=======
    models.append((OneVsRestClassifier(DecisionTreeClassifier(max_depth=5)), {}, True))


    models.append((RandomForestClassifier(max_depth=3, n_estimators=500, max_features=10), {}, True))

    # clf_adaboost = AdaBoostClassifier(random_state=0,
    #                                  algorithm = 'SAMME')
    #
    # models.append((clf_adaboost, {'n_estimators' : [100, 700, 1000],
    #                               'learning_rate' : [0.02, 0.07, 0.1],
    #                               'base_estimator' : [DecisionTreeClassifier(max_depth=3),
    #                                                   DecisionTreeClassifier(max_depth=4),
    #                                                   DecisionTreeClassifier(max_depth=5)]}))

    clf_adaboost = AdaBoostClassifier(random_state=0,
                                      algorithm = 'SAMME',
                                      n_estimators=700,
                                      learning_rate=0.07,
                                      base_estimator=DecisionTreeClassifier(max_depth=5))

    models.append((clf_adaboost, {}, True))

    # clf_mlp = MLPClassifier(random_state=1,
    #                         activation='relu',
    #                         solver='adam',
    #                         hidden_layer_sizes=(100, 1000, 1000, 100),
    #                         learning_rate_init=0.005,
    #                         learning_rate='adaptive',
    #                         alpha=0.0002,
    #                         shuffle=True,
    #                         early_stopping=True,
    #                         validation_fraction=0.15,
    #                         n_iter_no_change=5,
    #                         verbose=True,
    #                         max_iter=1000)

    # with smote, same net gave 0.677 on validation and 0.86 on training
    # without smote, this gave 0.7 on validation and validation score 0.74 during training

    clf_mlp = MLPClassifier(random_state=1,
                            activation='relu',
                            solver='sgd',
                            hidden_layer_sizes=(100, 1000, 1000, 100),
                            learning_rate_init=0.005,
                            learning_rate='adaptive',
                            alpha=0.0002,
                            shuffle=True,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=5,
                            verbose=False,
                            max_iter=1000)

    models.append((clf_mlp, {}, False))

>>>>>>> 9e08705bae64fa3d16a976c0706064cf693db872
    # 3.1 Choose the metrics for validation to display
=======
                                          'base_estimator': [DecisionTreeClassifier(max_depth=4)]}))
    models.append( (VotingClassifier(estimators=[('mlp1', MLPClassifier(alpha=0.0001, max_iter=150, hidden_layer_sizes = (100,150,150,100,10))),
    ('mlp2', MLPClassifier(alpha=0.0001, max_iter=150, hidden_layer_sizes = (50,100,200,200,100))),
    ('mlp3', MLPClassifier(alpha=0.0001, max_iter=150, hidden_layer_sizes = (50,100,100,200,100))),
    ('ada', AdaBoostClassifier(n_estimators=100, base_estimator=DecisionTreeClassifier(max_depth=4))) ],voting='hard'), {} , True) )
    #3.1 Choose the metrics for validation to display
>>>>>>> Stashed changes
    validation_metrics = []
    validation_metrics.append((metrics.accuracy_score, {}))
    validation_metrics.append((metrics.precision_score, {'average': 'weighted'}))
    validation_metrics.append((metrics.f1_score, {'average': 'weighted'}))
    validation_metrics.append((metrics.recall_score, {'average': 'weighted'}))

    # 3.2 Find best model for each task according to the validation dataset
    chosen_models_dict = choose_best_model(tasks, models, False,
                                           validation_metrics, outputs_folder_path, logfd)

    # 3.3 print nicely
    print_best_task_models(chosen_models_dict, tasks, models, logfd)

    if dataset_type == 'virus' and evaluate_on_test == True:
        # 3.4 Check scores of the best model on the test dataset
        predicted_out_path = os.path.join(outputs_folder_path, 'test_predicted.csv')
        test_best_model(tasks, models, chosen_models_dict, predicted_out_path, logfd)

        # # 3.5 Check scores of the best model on the unseen dataset, print into 'predicted' file
        #       we need to create dataset for each task as previously

        external_datasets, patient_IDs = get_external_datasets(os.path.join('input_ds', 'virus_hw5_test.csv'),
                                                               tasks,
                                                               targets_mappings,
                                                               os.path.join('input_ds', 'virus_hw5_unlabeled'))

        predicted_out_path = os.path.join(outputs_folder_path, 'unseen_predicted.csv')
        test_best_model(tasks, models, chosen_models_dict, predicted_out_path, logfd, patient_IDs,
                        external_datasets, True)

    logfd.close()


if __name__ == "__main__":
    # dataset_type = virus / iris
    init_automatic_classification(regenerate_features=True,
                                  evaluate_on_test=True,
                                  dataset_type='virus')
