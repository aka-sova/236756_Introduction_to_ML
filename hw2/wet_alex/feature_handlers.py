

import sklearn
from sklearn.pipeline import Pipeline

# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC

import pandas as pd
import numpy as np

from utils import customPipeline, CustomFeatureHandler

"""
Put all the operators on the features here
Implement in a pipeline as in the 'data_preparation.py' file
"""

class Template(CustomFeatureHandler):
    """DESCRIPTION"""

    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):

        return df

"""=========================================================="""

class BloodTypeHandler(CustomFeatureHandler):
    """Add blood type to categories"""

    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):
        df.BloodType = df.BloodType.astype('category')
        df.BloodType = df.BloodType.cat.add_categories("Unknown")
        df.BloodType = df.BloodType.fillna("Unknown")

        df["BloodTypeInt"] = df.BloodType.cat.rename_categories\
            (range(df.BloodType.nunique())).astype(int)
        df = df.drop(columns="BloodType")

        return df


class SexHandler(CustomFeatureHandler):
    """Deal with Sex categories"""

    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):
        df.Sex = df.Sex.astype('category')
        df.Sex = df.Sex.cat.add_categories("No Gender")
        df.Sex = df.Sex.fillna("No Gender")

        df["SexInt"] = df.Sex.cat.rename_categories(range(df.Sex.nunique())).astype(int)
        df = df.drop(columns="Sex")

        return df



class SelfDeclaration_to_Categories(CustomFeatureHandler):
    """Parse each declaration into separate declarations"""

    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):

        unique_self_declaracion_classes = []

        # string parser
        def parse_syndromes(syndromes_str: str) -> list:
            syndromes_list = []
            syndromes_str = syndromes_str.split(sep=";")

            for syndrome in syndromes_str:
                syndrome = syndrome.strip()
                syndromes_list.append(syndrome)

            return syndromes_list

        # create list of all syndromes
        for syndromes_list in df.SelfDeclarationOfIllnessForm.unique():
            if not isinstance(syndromes_list, str):
                continue

            syndromes = parse_syndromes(syndromes_list)
            for syndrome in syndromes:
                if syndrome not in unique_self_declaracion_classes:
                    unique_self_declaracion_classes.append(syndrome)

        # for each syndromes list, will return the dict with binary encoding for each syndrome
        def add_syndrom_categories(virus_df, syndromes_classes) -> pd.DataFrame:
            # parse each sample of syndromes_df
            # and set True for specific category

            syndromes_dict = {}

            for syndrom in syndromes_classes:
                syndromes_dict[syndrom] = np.zeros(virus_df.shape[0])

            for idx, sample in enumerate(virus_df.SelfDeclarationOfIllnessForm):
                if not isinstance(sample, str):
                    continue
                syndromes_list = parse_syndromes(sample)

                for syndrome in syndromes_list:
                    syndromes_dict[syndrome][idx] = 1

            return syndromes_dict

        # dict for each symptom, list of booleans
        syndromes_dict = add_syndrom_categories(df, unique_self_declaracion_classes)

        # change each to category in the dataset
        for key in syndromes_dict.keys():
            df[key] = syndromes_dict[key]
            df[key] = df[key].astype('category')

            # create the binary equivalent
            df[f"{key}Int"] = df[key].cat.rename_categories(range(df[key].nunique())).astype(int)
            df = df.drop(columns=key)

        df.drop(columns=['SelfDeclarationOfIllnessForm'])

        return df

class Drop_Irrelevant(CustomFeatureHandler):
    """Drop irrelevant features"""

    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):
        df = df.drop(columns=['PatientID', 'Address']) # meaningless to disease diagnosis
        return df


class Modify_Results_Code(CustomFeatureHandler):
    """one-hot encode the test results"""

    def __init__(self):
        super().__init__()

    def transform(self, df: pd.DataFrame()):
        """Replace the test results with the new dataframe"""

        # test result classified by labels
        li = df.TestResultsCode.tolist()
        labels = [item.split('_') for item in li]
        for item in labels:
            if len(item) == 4:
                add = item[0] + item[1]
                item = item.insert(0, add)
        for item in labels:
            if 'not' in item:
                item.remove('not')
            if 'detected' in item:
                item.remove('detected')

        disease = [la[0] for la in labels]
        spread = [la[1] for la in labels]
        risk = [la[2] for la in labels]

        disease_encode = pd.Series(disease).str.get_dummies()
        spread_encode = pd.Series(spread).str.get_dummies()
        risk_encode = pd.Series(risk).str.get_dummies()

        disease_encode = pd.DataFrame(disease_encode)
        spread_encode = pd.DataFrame(spread_encode)
        risk_encode = pd.DataFrame(risk_encode)


        spread_encode = spread_encode.drop(['NotSpreader'], axis=1)
        risk_encode = risk_encode.drop(['NotatRisk'], axis=1)

        frames = [df, disease_encode, spread_encode, risk_encode]
        df = pd.concat(frames, axis=1)

        # drop the original label
        df = df.drop(columns='TestResultsCode')

        return df

class BMI_handler(CustomFeatureHandler):
    """Apply transformation on the BMI parameters"""

    def __init__(self, max_threshold: int):
        self.max_threshold = max_threshold

    def transform(self, df : pd.DataFrame()):

        mean_train_BMI = np.mean(df.BMI)
        outlier_bmi_mask = df.BMI > self.max_threshold

        df.loc[outlier_bmi_mask, "BMI"] = mean_train_BMI
        return df


from sklearn import decomposition
import pandas as pd
class PCR_results_handler(CustomFeatureHandler):
    """From the analysis, the PCR results 3, 12, and 16 can be removed"""
    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):

        pca = decomposition.PCA(n_components=5)
        pca.fit_transform(df_scalar) #plug in scaled values ( with outliers )
        V = pca.components_

        cov = np.zeros( (5,4) )
        for i in range(5):
            sort = np.sort( np.absolute(V[i]) )
            for j in range(4):
                cov[i][j]=sort[15-j]
        cov_idx = np.zeros( (5,4) )

        for i in range(5):
            where =  [ (idx+1)  for idx, item in enumerate(V[i]) if np.absolute(item) >= cov[i][3] ] #  indices of 3 maximal coefficients
        for j in range(4):
            cov_idx[i][j] = where[j]
        keep = [ int(cov_idx[i][j])  for i in range(3) for j in range(4)]
        keep_cols =[ 'pcrResult{}'.format(itr) for itr in keep]

        
        return df[keep_cols]



class DropNA(CustomFeatureHandler):
    """Drop all columns, which have any N/A value"""

    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):
        return df.dropna()
