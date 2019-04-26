import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFdr, SelectKBest


dataset = pd.read_csv('merge_features.csv')
dataset = dataset.drop(columns='Sample')

y = dataset['Subgroup']
X = dataset.drop(columns='Subgroup')

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

X.shape

X_new = SelectKBest(k=100).fit(X_train, y_train)

features_list = X.columns
scores = X_new.scores_
pairs = zip(features_list[1:], scores)

k_best_features = pd.DataFrame(list(pairs), columns=['feature', 'score'])
k_best_features = k_best_features.sort_values('score', ascending=False)

k_best_features['feature'][2140]

selectedFeatures = pd.DataFrame({'feature': X['17_35286565_35336158'], 'Subgroup': y})
selectedFeatures.to_csv('selected_features.csv', index=False)

X_new.scores_
X_new.get_support([0])
