import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import re


class preprocess:
    def __init__(self, path):
        self._data = pd.read_csv(path)
        # self._data = pd.get_dummies(self._data)

    def normalize_to_10(col):
        return (col - col.min()) / (col.max() - col.min()) * 10

    def convert_range_to_mean(value):
        if isinstance(value, str) and re.match(r'\[\d+-\d+\]', value):
            numbers = list(map(int, re.findall(r'\d+', value)))
            return np.mean(numbers)
        return value

    def preprocess_data(self):
        for col in self._data.select_dtypes(include=np.number).columns:
            self._data[col].fillna(self._data[col].mean(), inplace=True)
        for col in self._data.select_dtypes(include='object').columns:
            self._data[col].fillna(self._data[col].mode()[0], inplace=True)
        for col in self._data.select_dtypes(include=np.number).columns:
            self._data[col] = self.normalize_to_10(self._data[col])

        for col in self._data.columns:
            if col != "match":
                self._data[col] = self._data[col].apply(self.convert_range_to_mean)
        selector = VarianceThreshold(threshold=0.01)
        self._data = pd.DataFrame(selector.fit_transform(self._data), columns=self._data.columns[selector.get_support()])
        self._data.drop(["professional_role"])
        self._data.drop(["professional_role_o"])
        #TODO not sure true
        self._data.drop(["d_d_years_experience"])
        self._data.drop(["ethnic_background"])
        self._data.drop(["ethnic_background_o"])



