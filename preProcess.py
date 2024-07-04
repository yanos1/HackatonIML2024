import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import re


class Preprocess:
    def _init_(self, path):
        self._data = pd.read_csv(path)
        # self._data = pd.get_dummies(self._data)

    def normalize_to_10(self, col):  # this might not be great yet.
        return (col - col.min()) / (col.max() - col.min()) * 10

    def remove_ranges(self):
        return self._data[[col for col in self._data.columns if not col.startswith('d_')]]

    def fill_majority_element_for_missing_string_values(self):
        for col in self._data.select_dtypes(include='object').columns:
            self._data[col].fillna(self._data[col].mode()[0], inplace=True)

    def update_race_score(self):
        self._data['shared_ethnicity'] = self._data[
            'shared_ethnicity'].replace(
            {"b'0'": 0, "b'1'": 1})
        condition = (self._data['shared_ethnicity'] == 1) & (
                self._data['shared_importance'] > 5)
        # Update shared_ethnicity based on condition
        self._data.loc[condition, 'significance_shared_ethnicity'] += \
            self._data.loc[
                condition, 'significance_shared_ethnicity']
        condition = (self._data['shared_ethnicity'] == 1) & (
                self._data['shared_importance'] <= 5)
        self._data.loc[condition, 'significance_shared_ethnicity'] += \
            (self._data.loc[
                condition, 'significance_shared_ethnicity']) // 2

        self._data.drop(columns=['significance_shared_ethnicity'])

    def preprocess_data(self):
        for col in self._data.select_dtypes(include=np.number).columns:
            if col != 'match':  # Skip the column named 'match'
                self._data[col].fillna(self._data[col].mean(), inplace=True)
                self._data[col] = self.normalize_to_10(self._data[col])

        self.fill_majority_element_for_missing_string_values()

        # handle ranges (probably useless to keep them)
        self._data = self.remove_ranges()

        # handle race
        self.update_race_score()

        #TODO not sure true
        self._data.drop(["professional_role"])
        self._data.drop(["ethnic_background"])
        self._data.drop(["ethnic_background_o"])
        self._data.drop(["unique_id"])
        self._data.drop(["has_missing_features"])
        self._data.drop(["years_of_experience"])
        self._data.drop(["years_of_experience_o"])
        self._data.drop(["wave"])
        ##

        selector = VarianceThreshold(threshold=0.01)
        self._data = pd.DataFrame(selector.fit_transform(self._data), columns=self._data.columns[selector.get_support()])






