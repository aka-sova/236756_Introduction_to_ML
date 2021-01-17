

import os
import pandas as pd                 # data analysis and manipulation tool
import numpy as np                  # Numerical computing tools

from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import timeit
from utils import print_all
from data_preparation import preprocess_csv_input, make_datasets, get_virus_dataset

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import mlxtend
from sklearn.neighbors import KNeighborsClassifier

class Task(object):
    def __init__(self, task_name : str, target_type : str, main_metrics : str,
                 pipeline, mapping_dict : dict = {}, order : int = None, output_column_name : str = None):
        self.task_name = task_name
        self.target_type = target_type
        self.main_metrics = main_metrics
        self.pipeline = pipeline
        self.mapping_dict = mapping_dict
        self.order = order
        self.datasets = {}
        self.output_column_name = output_column_name

    def print_attributes(self):
        print(f"task_name = {self.task_name}")
        print(f"target_type = {self.target_type}")
        print(f"main_metrics = {self.main_metrics}")
        print(f"pipeline = {self.pipeline}")
        print(f"mapping_dict = {self.mapping_dict}")
        print(f"order = {self.order}")



def choose_best_model( tasks : list, models: list, validation_metrics : list, output_folder_path : str, logfd):
    """Iterates over every task, and every model, returning the best model for each class."""

    # returns
    #   dict:       { task_1_idx : (model_2_idx, main_metrics, clf) ,
    #                 task_2_idx : (model_3_idx, main_metrics, clf),  ... etc }




    best_task_metrics = {task_idx : None for task_idx in range(len(tasks))}
    best_task_models = {task_idx: None for task_idx in range(len(tasks))}

    for task_idx, task in enumerate(tasks):
        print_all(logfd, "-"*40)
        print_all(logfd, f"\n\nInitializing model selection for task : {task.task_name}")
        # train on different classifiers
        metrics_dict = iterate_over_models(task, models, logfd)

        # choose the best according to the specified metric in the task
        for model_idx in metrics_dict.keys():

            # if this is the first model, take it
            if best_task_metrics[task_idx] == None:
                best_task_metrics[task_idx] = metrics_dict[model_idx][0]
                best_task_models[task_idx] = (model_idx, metrics_dict[model_idx][0], metrics_dict[model_idx][1])

            # if the metrics are higher than currently highest, choose this model for this task
            elif metrics_dict[model_idx][0] > best_task_metrics[task_idx]:
                best_task_metrics[task_idx] = metrics_dict[model_idx][0]
                best_task_models[task_idx] = (model_idx, metrics_dict[model_idx][0], metrics_dict[model_idx][1])

        # try the classifiers on the validation dataset and print the results
        print_all(logfd, "-"*40)
        print_all(logfd, f"\n\nTrying on the validation dataset with different metrics")

        test_metrics_on_validation(task, models, metrics_dict, validation_metrics, logfd)




    return best_task_models




def iterate_over_models(task, models: list, logfd):
    """For a specific task, iterates over models. Returns all kinds of metrics"""

    # inputs
    #   models -

    # returns
    #   {0 : [metric : <value>,  clf : <clf> ],
    #    1 : [metric : <value>,  clf : <clf> ], ....}
    #
    #   where 0,1, ..  = indexes in the models list, clf is the trained classifier,
    #   metric - values according to chosen metric

    metrics_dict = {}

    # obtain the correct X and y  (features and target) for this specific task
    # from the train_dataset

    X = task.datasets['train'].data
    y = task.datasets['train'].target_types[task.target_type]["targets"]

    for model_idx, model in enumerate(models):
        print_all(logfd, f"\nChecking model : {model[0]} over params {model[1]}")
        init_time = timeit.default_timer()
        # use the GridSearch cross validation method to find the best parameters for the model
        # from the specified parameters
        clf_scorer = metrics.make_scorer(task.main_metrics)

        if model[1] is not {}:
            clf = GridSearchCV(model[0], model[1], scoring = clf_scorer)
        else:
            clf = model[0]
        clf.fit(X, y)

        s = clf.cv_results_['rank_test_score']
        sorted_rank_idxs = sorted(range(len(s)), key=lambda k: s[k])

        print_all(logfd, f"Finished. Time elapsed: {round(((timeit.default_timer() - init_time)/60), 3)} [min]")

        print_all(logfd, "\tReached accuracies:")
        for rank_idx in sorted_rank_idxs:
            print_all(logfd, f"\t\tScore : {round(clf.cv_results_['mean_test_score'][rank_idx], 3)} , "
                  f"Params : {clf.cv_results_['params'][rank_idx]}")

        # apply all those models on the validation set and pick the best

        # take best score value
        metrics_dict[model_idx] = [max(clf.cv_results_['mean_test_score']), clf]

    return metrics_dict



def test_metrics_on_validation(task, models: list, metrics_dict : dict, validation_metrics, logfd):

    # 1. for each model, test each metric, organize all in a table

    X_valid = task.datasets['valid'].data
    y_true = task.datasets['valid'].target_types[task.target_type]["targets"]
    metrics_data = [] # list of lists for each model

    is_multiclass_task = False
    if len(set(y_true)) > 2:
        is_multiclass_task = True

    for model_idx, model in enumerate(models):

        # predict on the validation data
        model_ut = metrics_dict[model_idx][1]
        y_pred = model_ut.predict(X_valid)

        metrics_data.append([])
        metrics_data[-1].append(str(model_ut.estimator))



        for i in range(len(validation_metrics)):
            # for multitask we need to specify the 'average' parameter
            if is_multiclass_task:
                metric_calculated = validation_metrics[i][0](y_true, y_pred, **validation_metrics[i][1])
            else:
                metric_calculated = validation_metrics[i][0](y_true, y_pred)

            metrics_data[-1].append(round(metric_calculated, 3))


    titles_metrics = [metrics[0].__name__ for metrics in validation_metrics]
    row_format = "{:>60}" + "{:>25}" * (len(titles_metrics))

    print_all(logfd, row_format.format("", *titles_metrics)) # titles
    for row in metrics_data:
        print_all(logfd, row_format.format(*row))


def test_best_model(tasks, models, chosen_models_dict, predicted_out_path, logfd, patient_IDs : list = [],
                    external_datasets = None, use_external_datasets : bool = False, merge_outputs : bool = False):

    # the dict with best model for each task
    # chosen_models_dict:  { task_1_idx : (model_2_idx, main_metrics, best_classifier) ,
    #                        task_2_idx : (model_3_idx, main_metrics, best_classifier), ... etc }

    print_all(logfd, f"\n\n\nEvaluating best model on the test dataset")
    print_all(logfd, f"\nResult will be printed in : \t{predicted_out_path}")


    # for each task, get the best model, and get predictions

    # sort the tasks according to the order of the TestResultsCode
    sorted_task_idxs = sorted(range(len(tasks)), key=lambda k: tasks[k].order)
    final_results_list = []

    for task_idx in sorted_task_idxs:

        task = tasks[task_idx]
        best_model = chosen_models_dict[task_idx][2]

        if use_external_datasets == False:
            X = task.datasets['test'].data
        else:
            X = external_datasets[task_idx].data

        y_pred = best_model.predict(X)

        # if targets exist, evaluate on the targets
        if use_external_datasets == False:
            if task.datasets['test'].target_types != {}:
                y_true = task.datasets['test'].target_types[task.target_type]["targets"]
                metrics_result = task.main_metrics(y_true = y_true, y_pred = y_pred)
                print_all(logfd, f"\tTask : {task.task_name}, targets were supplied."
                                 f" {task.main_metrics.__name__} = {metrics_result}")

        # convert the predictions into the actual target names according to mapping
        task_inv_mapping = {v: k for k, v in task.mapping_dict.items()}
        targets_mapped = [task_inv_mapping[y_pred_single] for y_pred_single in y_pred]
        final_results_list.append(targets_mapped)

    results_dict = {}

    for result in final_results_list:
        if results_dict == {}:
            results_dict = {idx : specific_result for idx, specific_result in enumerate(result)}
        else:
            results_dict = {idx: results_dict[idx] + '_' + specific_result for idx, specific_result in enumerate(result)}

    final_results_list_united = list(results_dict.values())

    # check if input csv has patient ID
    if patient_IDs != []:
        patient_id_list = patient_IDs
    else:
        patient_id_list = range(len(final_results_list_united))


    # put those results in an output csv file
    if merge_outputs:
        output_pd = pd.DataFrame({'PatientID' : patient_id_list, 'TestResultsCode' :  final_results_list_united})
    else:
        output_pd = pd.DataFrame({'PatientID': patient_id_list,
                                  tasks[sorted_task_idxs[0]].output_column_name: final_results_list[0],
                                  tasks[sorted_task_idxs[1]].output_column_name: final_results_list[1],
                                  tasks[sorted_task_idxs[2]].output_column_name: final_results_list[2]
                                  })
    output_pd.to_csv(predicted_out_path, index=False)



def print_best_task_models(chosen_models_dict : dict, tasks : list, models : list, logfd):

    print_all(logfd, "="*40)
    print_all(logfd, "SUMMARY")
    print_all(logfd, "=" * 40)

    for task_idx, task in enumerate(tasks):
        print_all(logfd, f"\nFor task : {tasks[task_idx].task_name}")
        print_all(logfd, f"\tChosen model : {models[chosen_models_dict[task_idx][0]]}")
        print_all(logfd, f"\tMetrics : {tasks[task_idx].main_metrics.__name__}")
        print_all(logfd, f"\tValue : {round(chosen_models_dict[task_idx][1], 3)}")


def forward_select(X_train, y_train, scoring):
    """ FSF to find best features,
    receives scoring(string) and dataset
    """
    knn = KNeighborsClassifier(n_neighbors=2)

    sfs1 = SFS(estimator=knn, 
            k_features=3,
            forward=True, 
            floating=False, 
            scoring=scoring,
            cv=5)

    pipe = Pipeline([('sfs', sfs1), 
                    ('knn', knn)])

    param_grid = [
    {'sfs__k_features': [5,7],
    'sfs__estimator__n_neighbors': [1, 5, 10]}
    ]

    gs = GridSearchCV(estimator=pipe, 
                    param_grid=param_grid, 
                    scoring='f1', 
                    n_jobs=1, 
                    cv=5,
                    iid=True,
                    refit=True)

    # run gridearch
    gs = gs.fit(X_train, y_train)
    return gs.best_estimator_.steps[0][1].k_feature_idx_ #returns list of best features by index



def get_external_datasets(input_csv_filename, tasks, targets_mappings, output_folder_name):

    datasets = []
    patient_IDs = []
    os.makedirs(output_folder_name, exist_ok=True)

    for task_idx, task in enumerate(tasks):

        patient_IDs = preprocess_csv_input(input_csv_filename,
                                           os.path.join(output_folder_name, f'virus_hw5_unlabeled_{task.task_name}.csv'),
                                           task.pipeline, False, True)

        task_dataset = get_virus_dataset(os.path.join(output_folder_name, f'virus_hw5_unlabeled_{task.task_name}.csv'),
                                           targets_mappings,
                                           has_targets = False)
        datasets.append(task_dataset)

    return datasets, patient_IDs