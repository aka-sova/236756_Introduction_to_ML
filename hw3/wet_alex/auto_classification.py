

import os
import pandas as pd                 # data analysis and manipulation tool
import numpy as np                  # Numerical computing tools

from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import timeit
from utils import print_all


class Task(object):
    def __init__(self, task_name : str, target_type : str, main_metrics : str, mapping_dict : dict = {}, order : int = None):
        self.task_name = task_name
        self.target_type = target_type
        self.main_metrics = main_metrics
        self.mapping_dict = mapping_dict
        self.order = order

    def print_attributes(self):
        print(f"task_name = {self.task_name}")
        print(f"target_type = {self.target_type}")
        print(f"main_metrics = {self.main_metrics}")
        print(f"mapping_dict = {self.mapping_dict}")
        print(f"order = {self.order}")



def choose_best_model( tasks : list, models: list, train_dataset : Bunch, valid_dataset : Bunch,
                       validation_metrics : list, output_folder_path : str, logfd):
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
        metrics_dict = iterate_over_models(task, models, train_dataset, logfd)

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

        test_metrics_on_validation(task, models, metrics_dict, valid_dataset,
                                   validation_metrics, logfd)




    return best_task_models




def iterate_over_models(task, models: list, train_dataset : Bunch, logfd):
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

    X = train_dataset.data
    y = train_dataset.target_types[task.target_type]["targets"]

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



def test_metrics_on_validation(task, models: list, metrics_dict : dict, valid_dataset, validation_metrics, logfd):

    # 1. for each model, test each metric, organize all in a table

    y_true = valid_dataset.target_types[task.target_type]["targets"]
    X_valid = valid_dataset.data
    metrics_data = [] # list of lists for each model

    for model_idx, model in enumerate(models):

        # predict on the validation data
        model_ut = metrics_dict[model_idx][1]
        y_pred = model_ut.predict(X_valid)

        metrics_data.append([])
        metrics_data[-1].append(str(model_ut.estimator))

        for metrics in validation_metrics:
            metrics_data[-1].append(round(metrics[0](y_true = y_true, y_pred = y_pred, **metrics[1]), 3))


    titles_metrics = [metrics[0].__name__ for metrics in validation_metrics]
    row_format = "{:>30}" + "{:>25}" * (len(titles_metrics))

    print_all(logfd, row_format.format("", *titles_metrics)) # titles
    for row in metrics_data:
        print_all(logfd, row_format.format(*row))


def test_best_model(tasks, models, chosen_models_dict, test_dataset, predicted_out_path, logfd, patient_IDs : list = []):

    # the dict with best model for each task
    # chosen_models_dict:  { task_1_idx : (model_2_idx, main_metrics, best_classifier) ,
    #                        task_2_idx : (model_3_idx, main_metrics, best_classifier), ... etc }

    print_all(logfd, f"\n\n\nEvaluating best model on the dataset: \t{test_dataset.filename}")
    print_all(logfd, f"\nResult will be printed in : \t{predicted_out_path}")

    X = test_dataset.data
    # for each task, get the best model, and get predictions

    sorted_task_idxs = sorted(range(len(tasks)), key=lambda k: tasks[k].order)
    final_results_list = []

    for task_idx in sorted_task_idxs:

        task = tasks[task_idx]
        best_model = chosen_models_dict[task_idx][2]
        y_pred = best_model.predict(X)

        # if targets exist, evaluate on the targets
        if test_dataset.target_types != {}:
            y_true = test_dataset.target_types[task.target_type]["targets"]
            metrics_result = task.main_metrics(y_true = y_true, y_pred = y_pred)
            print_all(logfd, f"\tTargets were supplied. {task.main_metrics.__name__} = {metrics_result}")

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
    output_pd = pd.DataFrame({'PatientID' : patient_id_list, 'TestResultsCode' :  final_results_list_united})
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



