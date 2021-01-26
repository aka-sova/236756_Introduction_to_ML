


import pandas as pd                 # data analysis and manipulation tool
import numpy as np                  # Numerical computing tools
import seaborn as sns               # visualization library
import matplotlib.pyplot as plt     # another visualization library

from sklearn import datasets
from sklearn.utils import Bunch

from utils import *
from feature_handlers import *
from dataset_operations import *



def given_features_pipe(dataset_path : str, split_list : list):

    # virus_df = pd.read_csv(dataset_path)

    # The given features to use are:
    relevant_features = ["DisciplineScore", "TimeOnSocialActivities", "AgeGroup", "StepsPerYear",
                         "pcrResult4", "pcrResult1", "pcrResult12", "pcrResult5", "pcrResult16",
                         "pcrResult14", "SyndromeClass"]

    mean_discipline = 4.88
    mean_social_activity_time = 55.03
    mean_age_group = 3.36

    steps_per_year_delta = 1.75*(10**7) / 150 # from graph
    mean_steps_per_year = 11538977.98

    # PCR mean values for each feature without outliers on the training set
    pcr_mean = {}
    pcr_mean["pcrResult1"] = 0.14324074
    pcr_mean["pcrResult2"] = -0.31776428
    pcr_mean["pcrResult3"] = -2.34695487
    pcr_mean["pcrResult4"] = 0.11646489
    pcr_mean["pcrResult5"] = 0.00498924
    pcr_mean["pcrResult6"] = 0.0219719
    pcr_mean["pcrResult7"] = 0.03309889
    pcr_mean["pcrResult8"] = 0.01644742
    pcr_mean["pcrResult9"] = -0.27317063
    pcr_mean["pcrResult10"] = 0.00210894
    pcr_mean["pcrResult11"] = 0.02639129
    pcr_mean["pcrResult12"] = -0.63057441
    pcr_mean["pcrResult13"] = 0.98307771
    pcr_mean["pcrResult14"] = 4.87424804
    pcr_mean["pcrResult15"] = 0.04666256
    pcr_mean["pcrResult16"] = 1.55639375

    pcr_std = {}
    pcr_std["pcrResult1"] = 1.08148849
    pcr_std["pcrResult2"] = 8.8707197
    pcr_std["pcrResult3"] = 2.18932363
    pcr_std["pcrResult4"] = 1.08690174
    pcr_std["pcrResult5"] = 0.58354487
    pcr_std["pcrResult6"] = 2.52054654
    pcr_std["pcrResult7"] = 4.72226956
    pcr_std["pcrResult8"] = 1.23795492
    pcr_std["pcrResult9"] = 18.18598079
    pcr_std["pcrResult10"] = 1.50848103
    pcr_std["pcrResult11"] = 6.19488781
    pcr_std["pcrResult12"] = 4.3001735
    pcr_std["pcrResult13"] = 6.04664937
    pcr_std["pcrResult14"] = 25.74138099
    pcr_std["pcrResult15"] = 2.52811328
    pcr_std["pcrResult16"] = 0.80399315

    z_threshold = 2

    # Scaling parameters from the training dataset
    feature_mean = {}
    feature_mean["AgeGroup"] = 3.360444136
    feature_mean["DisciplineScore"] = 4.8840801559
    feature_mean["StepsPerYear"] = 11538977.981471933
    feature_mean["SyndromeClass"] = 2.9898888889
    feature_mean["TimeOnSocialActivities"] = 55.0343859087
    feature_mean["pcrResult1"] = 0.1602452074
    feature_mean["pcrResult12"] = -0.8498761528
    feature_mean["pcrResult14"] = 12.7732660013
    feature_mean["pcrResult16"] = 1.6052960456
    feature_mean["pcrResult4"] = 0.112480797
    feature_mean["pcrResult5"] = 0.0054885582


    feature_std = {}
    feature_std["AgeGroup"] = 1.1370979914
    feature_std["DisciplineScore"] = 1.365627197
    feature_std["StepsPerYear"] = 6320772.303302099
    feature_std["SyndromeClass"] = 1.2604267306
    feature_std["TimeOnSocialActivities"] = 15.1141476421
    feature_std["pcrResult1"] = 0.942804486
    feature_std["pcrResult12"] = 3.865892107
    feature_std["pcrResult14"] = 28.0027211667
    feature_std["pcrResult16"] = 0.7108454621
    feature_std["pcrResult4"] = 0.9461853683
    feature_std["pcrResult5"] = 0.5031145828



    results_fields = ['Disease', 'Spreader', 'atRisk']

    # push the dataframe through the pipeline
    data_processing_pipe = customPipeline(steps =
                                          [('DisciplineScoreHandler', DisciplineScoreHandler(mean_discipline=mean_discipline)),
                                           ('SocialActivitiesHandler', SocialActivitiesHandler(mean_social_activity_time = mean_social_activity_time)),
                                           ('AgeHandler', NewAgeHandler(mean_age=mean_age_group)),
                                           ('StepsHandler', NewStepsHandler(delta=steps_per_year_delta, mean=mean_steps_per_year)),
                                           ('PCR_results_handler', PCR_imputation(pcr_mean, pcr_std, z_threshold)),
                                           ('SyndromClassHandler', SyndromClassHandler()),
                                           ('Modify_Results_Code', Modify_Results_Code()),
                                           ('Leave_Relevant_Features', Leave_Relevant(relevant_features, results_fields)),
                                           ('Total_Scaler', Scale_All_Manually(results_fields, feature_mean, feature_std)),
                                           ('DropNA', DropNA()),
                                           ])

    return data_processing_pipe


def pca_srs_pipe_scaled(dataset_path : str, split_list : list):

    virus_df = pd.read_csv(dataset_path)
    df = split_the_data(virus_df, split_list)

    mean_bmi = 29
    mean_discipline = 4.95
    mean_social_activity_time = 5.32

    # PCR results learning
    pcr_scaler, pcr_pca = learn_pcr_transform(df["train"], n_components=5)

    # features that should stay according to SRS
    srs_features_to_stay = ["BMI", "DisciplineScore", "TimeOnSocialActivities"]
    PCA_components = range(5)
    pca_components_fields = ["PCA_" + str(i) for i in PCA_components]
    results_fields = ['Disease', 'Spreader', 'atRisk']
    self_declaration_categories = ['DiarrheaInt', 'Nausea_or_vomitingInt', 'Shortness_of_breathInt',
                                   'Congestion_or_runny noseInt', 'HeadacheInt', 'FatigueInt',
                                   'Muscle_or_body_achesInt', 'ChillsInt', 'Skin_rednessInt',
                                   'New_loss_of_taste_or_smellInt', 'Sore_throatInt']
    other_features_to_stay = ["SexInt", "BloodTypeInt", "SyndromeClass"] + \
                              pca_components_fields + \
                             self_declaration_categories +\
                             results_fields



    # push the dataframe through the pipeline
    data_processing_pipe = customPipeline(steps =
                                          [('Drop_Irrelevant', Drop_Irrelevant()),
                                           ('PCR_results_handler', PCR_results_handler(pcr_scaler, pcr_pca)),
                                           ('SexHandler', SexHandler()),
                                           ('BMI_handler', BMI_handler(max_threshold=45, mean_bmi=mean_bmi)),
                                           ('DisciplineScoreHandler', DisciplineScoreHandler(mean_discipline=mean_discipline)),
                                           ('BloodTypeHandler', BloodTypeHandler()),
                                           ('SocialActivitiesHandler', SocialActivitiesHandler(mean_social_activity_time = mean_social_activity_time)),
                                           ('SelfDeclaration_to_Categories', SelfDeclaration_to_Categories()),
                                           ('SyndromClassHandler', SyndromClassHandler()),
                                           ('Modify_Results_Code', Modify_Results_Code()),
                                           ('RemoveNotSRSColumns', RemoveNotSRSColumns(srs_features_to_stay,
                                                                                       other_features_to_stay)),
                                           ('Total_Scaler', Scale_All(results_fields)),
                                           ('DropNA', DropNA()),
                                           ])

    return data_processing_pipe


def pca_srs_pipe(dataset_path : str, split_list : list):

    virus_df = pd.read_csv(dataset_path)
    df = split_the_data(virus_df, split_list)

    mean_bmi = 29
    mean_discipline = 4.95
    mean_social_activity_time = 5.32

    # PCR results learning
    pcr_scaler, pcr_pca = learn_pcr_transform(df["train"], n_components=5)

    # features that should stay according to SRS
    srs_features_to_stay = ["BMI", "DisciplineScore", "TimeOnSocialActivities"]
    PCA_components = range(5)
    pca_components_fields = ["PCA_" + str(i) for i in PCA_components]
    results_fields = ['Disease', 'Spreader', 'atRisk']
    self_declaration_categories = ['DiarrheaInt', 'Nausea_or_vomitingInt', 'Shortness_of_breathInt',
                                   'Congestion_or_runny noseInt', 'HeadacheInt', 'FatigueInt',
                                   'Muscle_or_body_achesInt', 'ChillsInt', 'Skin_rednessInt',
                                   'New_loss_of_taste_or_smellInt', 'Sore_throatInt']
    other_features_to_stay = ["SexInt", "BloodTypeInt", "SyndromeClass"] + \
                              pca_components_fields + \
                             self_declaration_categories +\
                             results_fields



    # push the dataframe through the pipeline
    data_processing_pipe = customPipeline(steps =
                                          [('Drop_Irrelevant', Drop_Irrelevant()),
                                           ('PCR_results_handler', PCR_results_handler(pcr_scaler, pcr_pca)),
                                           ('SexHandler', SexHandler()),
                                           ('BMI_handler', BMI_handler(max_threshold=45, mean_bmi=mean_bmi)),
                                           ('DisciplineScoreHandler', DisciplineScoreHandler(mean_discipline=mean_discipline)),
                                           ('BloodTypeHandler', BloodTypeHandler()),
                                           ('SocialActivitiesHandler', SocialActivitiesHandler(mean_social_activity_time = mean_social_activity_time)),
                                           ('SelfDeclaration_to_Categories', SelfDeclaration_to_Categories()),
                                           ('SyndromClassHandler', SyndromClassHandler()),
                                           ('Modify_Results_Code', Modify_Results_Code()),
                                           ('RemoveNotSRSColumns', RemoveNotSRSColumns(srs_features_to_stay,
                                                                                       other_features_to_stay)),
                                           ('DropNA', DropNA()),
                                           ])

    return data_processing_pipe


def disease_pipe(dataset_path : str, split_list : list):

    virus_df = pd.read_csv(dataset_path)
    df = split_the_data(virus_df, split_list)

    mean_bmi = 29
    mean_discipline = 4.95
    mean_social_activity_time = 5.32

    # PCR results learning
    pcr_scaler, pcr_pca = learn_pcr_transform(df["train"], n_components=10)

    # features that should stay according to SRS
    srs_features_to_stay = ["BMI", "DisciplineScore", "TimeOnSocialActivities"]
    PCA_components = range(10)
    pca_components_fields = ["PCA_" + str(i) for i in PCA_components]
    pcr_components_fields = ["pcrResult" + str(i) for i in range(17)]
    results_fields = ['Disease', 'Spreader', 'atRisk']
    self_declaration_categories = ['DiarrheaInt', 'Nausea_or_vomitingInt', 'Shortness_of_breathInt',
                                   'Congestion_or_runny noseInt', 'HeadacheInt', 'FatigueInt',
                                   'Muscle_or_body_achesInt', 'ChillsInt', 'Skin_rednessInt',
                                   'New_loss_of_taste_or_smellInt', 'Sore_throatInt']
    other_features_to_stay = pcr_components_fields + \
                             pca_components_fields + \
                             self_declaration_categories +\
                             results_fields



    # push the dataframe through the pipeline
    data_processing_pipe = customPipeline(steps =
                                          [('Drop_Irrelevant', Drop_Irrelevant()),
                                           ('PCR_results_handler', PCR_standart_scaler_handler()),
                                           ('SexHandler', SexHandler()),
                                           ('BMI_handler', BMI_handler(max_threshold=45, mean_bmi=mean_bmi)),
                                           ('DisciplineScoreHandler', DisciplineScoreHandler(mean_discipline=mean_discipline)),
                                           ('BloodTypeHandler', BloodTypeHandler()),
                                           ('SocialActivitiesHandler', SocialActivitiesHandler(mean_social_activity_time = mean_social_activity_time)),
                                           ('SelfDeclaration_to_Categories', SelfDeclaration_to_Categories()),
                                           ('SyndromClassHandler', SyndromClassHandler()),
                                           ('Modify_Results_Code', Modify_Results_Code()),
                                           ('RemoveNotSRSColumns', RemoveNotSRSColumns(srs_features_to_stay,
                                                                                       other_features_to_stay)),
                                           ('DropNA', DropNA()),
                                           ])

    return data_processing_pipe


def risk_pipe(dataset_path : str, split_list : list):

    virus_df = pd.read_csv(dataset_path)
    df = split_the_data(virus_df, split_list)

    mean_bmi = 29
    mean_discipline = 4.95
    mean_social_activity_time = 5.32

    # PCR results learning
    pcr_scaler, pcr_pca = learn_pcr_transform(df["train"], n_components=5)

    # features that should stay according to SFS
    srs_features_to_stay = ["BMI", "DisciplineScore", "TimeOnSocialActivities","AgeGroup",'AvgMinSportsPerDay','AvgHouseholdExpenseOnPresents','HappinessScore','StepsPerYear','NrCousins']
    PCA_components = range(0) #no pac components
    pca_components_fields = ["PCA_" + str(i) for i in PCA_components]
    results_fields = ['Disease', 'Spreader', 'atRisk']
    #self_declaration_categories = ['DiarrheaInt', 'Nausea_or_vomitingInt', 'Shortness_of_breathInt',
    #                               'Congestion_or_runny noseInt', 'HeadacheInt', 'FatigueInt',
    #                               'Muscle_or_body_achesInt', 'ChillsInt', 'Skin_rednessInt',
    #                               'New_loss_of_taste_or_smellInt', 'Sore_throatInt']
    other_features_to_stay = ["SexInt", "BloodTypeInt", "SyndromeClass"] + \
                              pca_components_fields + \
                             results_fields



    # push the dataframe through the pipeline
    data_processing_pipe = customPipeline(steps =
                                          [('Drop_Irrelevant', Drop_Irrelevant()),
                                           ('PCR_results_handler', PCR_results_handler(pcr_scaler, pcr_pca)),
                                           ('SexHandler', SexHandler()),
                                           ('AgeHandler', AgeHandler()),
                                           ('StepsHandler', StepsHandler()),
                                           ('SportsHandler', SportsHandler()),
                                           ('NrCousins', CousinsHandler()),
                                           ('AvgHouseholdExpenseOnPresents', PresentsHandler()  ),  
                                           ('HappinessScore', HappyHandler()),
                                           ('BMI_handler', BMI_handler(max_threshold=45, mean_bmi=mean_bmi)),
                                           ('DisciplineScoreHandler', DisciplineScoreHandler(mean_discipline=mean_discipline)),
                                           ('BloodTypeHandler', BloodTypeHandler()),
                                           ('SocialActivitiesHandler', SocialActivitiesHandler(mean_social_activity_time = mean_social_activity_time)),
                                           ('SelfDeclaration_to_Categories', SelfDeclaration_to_Categories()),
                                           ('SyndromClassHandler', SyndromClassHandler()),
                                           ('Modify_Results_Code', Modify_Results_Code()),
                                           ('RemoveNotSRSColumns', RemoveNotSRSColumns(srs_features_to_stay,
                                                                                       other_features_to_stay)),
                                           ('DropNA', DropNA()),
                                           ])

    return data_processing_pipe



#pipeline optimizing dataset for classifying virus type
def virus_pipe(dataset_path : str, split_list : list):

    virus_df = pd.read_csv(dataset_path)
    df = split_the_data(virus_df, split_list)

    mean_bmi = 29
    mean_discipline = 4.95
    mean_social_activity_time = 5.32

    # PCR results learning
    pcr_scaler, pcr_pca = learn_pcr_transform(df["train"], n_components=5)

    # features that should stay according to SFS
    srs_features_to_stay = []#["BMI", "DisciplineScore", "TimeOnSocialActivities","AgeGroup",'AvgMinSportsPerDay','AvgHouseholdExpenseOnPresents','HappinessScore','StepsPerYear','NrCousins']
    PCA_components = range(10) #no pac components
    pca_components_fields = ["PCA_" + str(i) for i in PCA_components]
    results_fields = ['Disease', 'Spreader', 'atRisk']
    self_declaration_categories = ['DiarrheaInt', 'Nausea_or_vomitingInt', 'Shortness_of_breathInt',
                                   'Congestion_or_runny noseInt', 'HeadacheInt', 'FatigueInt',
                                   'Muscle_or_body_achesInt', 'ChillsInt', 'Skin_rednessInt',
                                   'New_loss_of_taste_or_smellInt', 'Sore_throatInt']
    other_features_to_stay = ["SyndromeClass"] + \
                              pca_components_fields + \
                              self_declaration_categories +\
                               results_fields
    #"SexInt", "BloodTypeInt",
    # push the dataframe through the pipeline
    data_processing_pipe = customPipeline(steps =
                                          [('Drop_Irrelevant', Drop_Irrelevant()),
                                           ('PCR_results_handler', PCR_results_handler(pcr_scaler, pcr_pca)),
                                           ('SexHandler', SexHandler()),
                                           ('AgeHandler', AgeHandler()),
                                           ('StepsHandler', StepsHandler()),
                                           ('SportsHandler', SportsHandler()),
                                           ('NrCousins', CousinsHandler()),
                                           ('AvgHouseholdExpenseOnPresents', PresentsHandler()  ),  
                                           ('HappinessScore', HappyHandler()),
                                           ('BMI_handler', BMI_handler(max_threshold=45, mean_bmi=mean_bmi)),
                                           ('DisciplineScoreHandler', DisciplineScoreHandler(mean_discipline=mean_discipline)),
                                           ('BloodTypeHandler', BloodTypeHandler()),
                                           ('SocialActivitiesHandler', SocialActivitiesHandler(mean_social_activity_time = mean_social_activity_time)),
                                           ('SelfDeclaration_to_Categories', SelfDeclaration_to_Categories()),
                                           ('SyndromClassHandler', SyndromClassHandler()),
                                           ('Modify_Results_Code', Modify_Results_Code()),
                                           ('RemoveNotSRSColumns', RemoveNotSRSColumns(srs_features_to_stay,
                                                                                       other_features_to_stay)),
                                           ('DropNA', DropNA()),
                                           ])

    return data_processing_pipe
def prepare_dataset(dataset_path : str, split_list : list, data_processing_pipe, output_folder_name : str):

    # 1. load the csv input
    virus_df = pd.read_csv(dataset_path)
    df = split_the_data(virus_df, split_list)

    # apply all the transforms one by one
    df["train"] = data_processing_pipe.apply_transforms(df["train"])
    df["valid"] = data_processing_pipe.apply_transforms(df["valid"])
    df["test"] = data_processing_pipe.apply_transforms(df["test"])

    save_csv_files(df, output_folder_name)

def preprocess_csv_input(input_csv_path : str, output_csv_path : str, data_processing_pipe,
                         dropResults : bool, return_patient_IDs : bool):

    input_df = pd.read_csv(input_csv_path)
    if dropResults == True:
        input_df = input_df.drop(columns = 'TestResultsCode')

    if return_patient_IDs == True:
        patient_IDs = input_df["PatientID"].to_list()

    output_df = data_processing_pipe.apply_transforms(input_df)
    output_df.to_csv(output_csv_path, index=False)

    if return_patient_IDs == True:
        return patient_IDs
    return



def make_datasets(dataset_type : str, features_folder_path : str, targets_mappings : dict):
    """Make datasets for the classifier in our special format"""

    if dataset_type == 'virus':

        train_dataset = get_virus_dataset(os.path.join(features_folder_path, 'train.csv'), targets_mappings)
        valid_dataset = get_virus_dataset(os.path.join(features_folder_path, 'valid.csv'), targets_mappings)
        test_dataset = get_virus_dataset(os.path.join(features_folder_path, 'test.csv'), targets_mappings)

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

        valid_dataset = Bunch()
        valid_dataset = train_dataset

        test_dataset = Bunch()
        test_dataset = train_dataset

    return train_dataset, valid_dataset, test_dataset


def get_virus_dataset(csv_filename : str, targets_mappings : dict, has_targets : bool = True):
    """Create a Bunch object with data and targets according to the mapping"""

    pd_dataset = pd.read_csv(csv_filename)

    disease_mapping = targets_mappings['disease_mapping']
    spreader_mapping = targets_mappings['spreader_mapping']
    at_risk_mapping = targets_mappings['at_risk_mapping']

    dataset = Bunch()
    dataset.filename = csv_filename
    dataset.target_types = {}

    if has_targets:
        dataset.target_types['Disease'] = {'target_names': list(disease_mapping.keys()),
                                                 'targets': pd_dataset['Disease'].to_numpy()}

        dataset.target_types['Spreader'] = {'target_names': list(spreader_mapping.keys()),
                                                  'targets': pd_dataset['Spreader'].to_numpy()}

        dataset.target_types['atRisk'] = {'target_names': list(at_risk_mapping.keys()),
                                                'targets': pd_dataset['atRisk'].to_numpy()}

    # drop all the targets from the train dataframe
    for target_type in dataset.target_types.keys():
        pd_dataset = pd_dataset.drop(columns=target_type)

    # all that's left are features = data
    dataset.data = pd_dataset.to_numpy()
    dataset.feature_names = pd_dataset.columns.to_list()

    return dataset

def get_targets_mappings():

    mappings_dict = {}

    mappings_dict['disease_mapping'] = {'flue': 0, 'covid': 1, 'cmv': 2, 'cold': 3, 'measles': 4, 'notdetected': 5}
    mappings_dict['spreader_mapping'] = {'NotSpreader': 0, 'Spreader': 1}
    mappings_dict['at_risk_mapping'] = {'NotAtRisk': 0, 'atRisk': 1}

    return mappings_dict

if __name__ == "__main__":

    dataset_path = os.path.join('input_ds', 'virus_hw2.csv')
    prepare_dataset(dataset_path = dataset_path, split_list = [0.75, 0.15, 0.10], output_folder_name = 'outputs_csv')
