import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
import matplotlib.pyplot as plt


class Preprocess:
    def __init__(self, path):
        self._data = pd.read_csv(path)


    def normalize_to_10(self, col):
        return (col - col.min()) / (col.max() - col.min()) * 10

    def remove_ranges(self):
        return self._data[[col for col in self._data.columns if not col.startswith('d_')]]

    def fill_majority_element_for_missing_string_values(self):
        for col in self._data.select_dtypes(include='object').columns:
            self._data[col].fillna(self._data[col].mode()[0], inplace=True)

    def update_race_score(self):
        self._data['shared_ethnicity'] = self._data['shared_ethnicity'].replace({"b'0'": 0, "b'1'": 1})
        condition = (self._data['shared_ethnicity'] == 1) & (self._data['significance_shared_ethnicity'] > 5)
        self._data.loc[condition, 'significance_shared_ethnicity'] += self._data.loc[
            condition, 'significance_shared_ethnicity']
        condition = (self._data['shared_ethnicity'] == 1) & (self._data['significance_shared_ethnicity'] <= 5)
        self._data.loc[condition, 'significance_shared_ethnicity'] += (self._data.loc[
            condition, 'significance_shared_ethnicity']) // 2

        self._data.drop(columns=['significance_shared_ethnicity'], inplace=True)

    def update_characteristic_score(self):
        characteristics = ["communication_skills", "reliability",
                           "intelligence", "creativity", "ambitious",
                           "shared_interests"]

        def calculate_match(row, x1, x2):
            pref_value = row[x1]
            other_value = row[x2]
            if pref_value >= 5 and other_value >= 5:
                return max(pref_value, other_value)
            elif pref_value > 6 and other_value < 5:
                return -max(pref_value, other_value)
            elif pref_value < 5 and other_value > 5:
                return 3
            else:
                return 2

        for char in characteristics:
            char_importance_for_partner = f"pref_o_{char}"
            char_grade_of_partner_of_me = f"{char}_o"

            char_importance_for_me = f"{char}_important"
            char_grade_for_partner_from_me = f"{char}_partner"

            new_col_me = f"{char}_match_partner_perspective"
            new_col_partner = f"{char}_match_my_perspective"

            self._data[new_col_me] = self._data.apply(calculate_match,
                                                      axis=1, args=(
                char_importance_for_partner, char_grade_of_partner_of_me))
            self._data[new_col_partner] = self._data.apply(calculate_match,
                                                           axis=1, args=(
                    char_importance_for_me, char_grade_for_partner_from_me))
            # Debug print to check columns
            print(f"Dropping columns: {char_importance_for_partner}, {char_grade_of_partner_of_me}, {char_importance_for_me}, {char_grade_for_partner_from_me}")
            self._data.drop([char_importance_for_partner, char_grade_of_partner_of_me, char_importance_for_me, char_grade_for_partner_from_me], axis=1, inplace=True)

    def preprocess_data(self):
        # self._data = pd.get_dummies(self._data)
        self._data.rename(columns={'intellicence_important': 'intelligence_important'}, inplace=True)
        self._data.rename(columns={'ambitous_o': 'ambitious_o'}, inplace=True)
        self._data.rename(columns={'ambition_important': 'ambitious_important'}, inplace=True)
        self._data.rename(columns={'ambition_partner': 'ambitious_partner'}, inplace=True)

        for col in self._data.select_dtypes(include=np.number).columns:
            if col != 'match':  # Skip the column named 'match'
                self._data[col].fillna(self._data[col].mean(), inplace=True)
                self._data[col] = self.normalize_to_10(self._data[col])

        self.fill_majority_element_for_missing_string_values()

        # handle ranges (probably useless to keep them)
        self._data = self.remove_ranges()

        # handle race
        self.update_race_score()

        # Drop unnecessary columns
        columns_to_drop = [
            "professional_role", "ethnic_background", "ethnic_background_o",
            "unique_id", "has_missing_features", "years_of_experience",
            "years_of_experience_o", "wave", "employee_benefits",
            "diversity_and_inclusion", "study_field", 'social_events'
        ]
        self._data.drop(columns=columns_to_drop, inplace=True)

        # Remove columns with low variance
        selector = VarianceThreshold(threshold=0.01)
        self._data = pd.DataFrame(selector.fit_transform(self._data),
                                  columns=self._data.columns[selector.get_support()])
        self.update_characteristic_score()

    def correlation_analysis(self, sample_fraction=0.15, threshold=0.8):
        # Sample the data
        sampled_data = self._data.sample(frac=sample_fraction, random_state=1)

        # Calculate correlation matrix
        corr_matrix = sampled_data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than the threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        print("Columns to drop due to high correlation:", to_drop)

        # Plot the heatmap
        plt.figure(figsize=(16, 12))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix (Sampled Data)')
        plt.show()

        return to_drop


# Example usage:
preprocess = Preprocess(r"C:\Users\mayah\Downloads\mixer_event_training.csv")
preprocess.preprocess_data()
columns_to_drop = preprocess.correlation_analysis(sample_fraction=0.15)
preprocess._data.drop(columns=columns_to_drop, inplace=True)
