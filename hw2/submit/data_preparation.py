


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


    mean_bmi = 29
    mean_discipline = 4.95
    mean_social_activity_time = 5.32

    # PCR results learning
    pcr_scaler, pcr_pca = learn_pcr_transform(df["train"], n_components=5)

    # features that should stay according to SRS
    srs_features_to_stay = ["BMI", "DisciplineScore", "TimeOnSocialActivities"]
    PCA_components = range(5)
    pca_components_fields = ["PCA_" + str(i) for i in PCA_components]
    results_fields = ['cmv', 'cold', 'covid', 'flue', 'measles', 'notdetected', 'Spreader', 'atRisk']
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

    # data_processing_pipe = customPipeline(steps = [('BMI_handler', BMI_handler(max_threshold=50)),
    #                                                ('PCR_results_handler', PCR_results_handler())])

    # apply all the transforms one by one
    df["train"] = data_processing_pipe.apply_transforms(df["train"])
    df["valid"] = data_processing_pipe.apply_transforms(df["valid"])
    df["test"] = data_processing_pipe.apply_transforms(df["test"])


    print(df["train"].head(10))
    

    save_csv_files(df, "csv_outputs")

if __name__ == "__main__":
    main()
