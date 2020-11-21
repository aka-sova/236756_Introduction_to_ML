import pandas as pd
import numpy as np

import sklearn
from sklearn.pipeline import Pipeline
import abc


class customPipeline(Pipeline):
    """Extension to the existing pipeline class. We add a function to simply apply transforms one after another"""

    def __init__(self, steps):
        super().__init__(steps)

    def apply_transforms(self, df_dut: pd.DataFrame()):

        for _, name, transform in self._iter(with_final=True):
            df_dut = transform.transform(df_dut)

        return df_dut


class CustomFeatureHandler():
    """Apply transformation on the some feature/s

    Any custom transform function has to include:
    the "transform" function which takes and returns dataframe
    and "fit" function which won't be used in our case.

    """
    def __init__(self, max_threshold: int):
        self.max_threshold = max_threshold

    @abc.abstractmethod
    def fit(self):
        # no fit required
        return self

    @abc.abstractmethod
    def transform(self, df_dut : pd.DataFrame()):
        return df_dut



def split_the_data(df: pd.DataFrame(), split_list):

    train_ds_part, valid_ds_part, test_ds_part = split_list
    total_ds_rows_n = df.shape[0]

    train_ds_part_int = round(total_ds_rows_n * train_ds_part)
    valid_ds_part_int = round(total_ds_rows_n * valid_ds_part)
    test_ds_part_int = total_ds_rows_n - valid_ds_part_int - train_ds_part_int

    train_end_idx = train_ds_part_int
    valid_end_idx = train_ds_part_int + valid_ds_part_int

    split_data = np.split(df, [train_end_idx, valid_end_idx], axis=0)

    print(f"Total size: {total_ds_rows_n}\n\
    Train df: {split_data[0].shape[0]}\n\
    Valid df: {split_data[1].shape[0]}\n\
    Test df: {split_data[2].shape[0]}")

    ds_obj = {}

    ds_obj["train"] = split_data[0]
    ds_obj["train_original"] = split_data[0]

    ds_obj["valid"] = split_data[1]
    ds_obj["valid_original"] = split_data[1]

    ds_obj["test"] = split_data[2]
    ds_obj["test_original"] = split_data[2]

    return ds_obj