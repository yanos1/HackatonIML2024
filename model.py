import numpy as np
import sklearn
import sklearn as sk
import pandas as pd
import csv
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


## TODO - REMOVE THE FUNC AND ALL CALLS TO IT BEFORE SUBMITTING
def testing(data):
    num_ones = 0
    num_zeros = 0
    num_NA = 0
    for i in data['match']:
        if i == 0:
            num_zeros += 1
        elif i == 1:
            num_ones += 1
        else:
            num_NA += 1
    print("num zeros:", num_zeros)
    print("num ones:", num_ones)
    print("num nas:", num_NA)


def first_baseline(data):
    np.random.seed(0)

    data = data.dropna(subset=['match'])
    # data = data.sample(frac=0.15)

    model = LogisticRegression(solver='liblinear')
    X, y = data.drop("match", axis=1), data.match
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # f1_match= f1_score(y_test, y_pred)
    # print(f1_match)
    match_predictions = pd.DataFrame({'unique_id': X_test.index, 'match': y_pred})
    match_predictions.to_csv('predictions/match_predictions.csv', index=False)

    return match_predictions


def second_baseline(data):
    np.random.seed(0)
    #data = data.sample(frac=0.15)
    data = data.dropna(subset=['match'])

    model = LinearRegression()

    creativity_predictions = single_regression_prediction(data, 'creativity_important', model)

    ambition_predictions = single_regression_prediction(data, 'ambition_important', model)

    importance_rating_predictions = pd.merge(creativity_predictions, ambition_predictions, on='unique_id')
    importance_rating_predictions.to_csv('predictions/importance_ratings_predictions.csv', index=False)

    return importance_rating_predictions


def single_regression_prediction(data, target_column, model):
    np.random.seed(0)
    data = data.dropna(subset=[target_column])

    X = data.drop([target_column], axis=1)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    predictions = pd.DataFrame({'unique_id': X_test.index, target_column: y_pred})
    # mse = mean_squared_error(y_test, y_pred)
    # print(target_column, " ", mse)
    return predictions


def clustering():
    np.random.seed(0)
    data = pd.read_csv('mixer_event_training.csv')

    # Convert categorical variables to dummy variables
    data = pd.get_dummies(data)

    # Fill missing values with the mean of each column except 'match'
    for col in data.select_dtypes(include=[np.number]).columns:
        if col != 'match':
            data[col] = data[col].fillna(data[col].mean())

    data_without_match = data.drop('match', axis=1)

    # Apply K-Means clustering
    k_means_model = KMeans(n_clusters=2)
    k_means_model.fit(data_without_match)

    # Create DataFrame for cluster labels and distances
    cluster_labels = pd.DataFrame({'cluster': k_means_model.labels_})
    distances = k_means_model.transform(data_without_match)
    distance_to_centroid = pd.DataFrame(
        {'distance_to_centroid': [distances[i, label] for i, label in enumerate(k_means_model.labels_)]})

    # Concatenate the new columns to the original data
    data = pd.concat([data, cluster_labels, distance_to_centroid], axis=1)

    # Calculate the probability of each match label within each cluster
    cluster_probabilities = {}
    for cluster in data['cluster'].unique():
        cluster_data = data[data['cluster'] == cluster]
        match_data = cluster_data['match'].dropna()
        total_count = len(match_data)
        if total_count > 0:
            match_probabilities = match_data.value_counts() / total_count
            cluster_probabilities[cluster] = match_probabilities.to_dict()

    # Assign the match label based on the highest probability within each cluster
    for cluster in data['cluster'].unique():
        if cluster in cluster_probabilities:
            match_probs = cluster_probabilities[cluster]
            max_prob_label = max(match_probs, key=match_probs.get)
            data.loc[(data['cluster'] == cluster) & (data['match'].isna()), 'match'] = max_prob_label

    # Drop the temporary columns
    data = data.drop(columns=['cluster', 'distance_to_centroid'])

    # Test if 'match' column is in the data
    if 'match' in data.columns:
        print("Test Passed: 'match' column is present in the data.")
    else:
        print("Test Failed: 'match' column is not present in the data.")

    testing(data)
    # Print only the 'match' column
    return data


if __name__ == '__main__':
    np.random.seed(0)
    data = pd.read_csv('mixer_event_training.csv')
    data = pd.get_dummies(data)
    for col in data.select_dtypes(include=[np.number]).columns:
        if col != 'match':
            data[col] = data[col].fillna(data[col].mean())
    testing(data)
    # first_baseline(data)
    second_baseline(data)
    data = clustering()
    print(data.head())
