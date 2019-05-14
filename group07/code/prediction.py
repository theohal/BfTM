#!usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
%matplotlib inline
from collections import defaultdict

# import original dataset
df = pd.read_csv('./input/df_sorted_Subgroup.csv')
# most important features
features = pd.read_csv('./output/final_features.csv')

# create a list with the important important
ftl = list(features.values[:,0])
# add the sample to the beginning and subgroup at the end
ftl.insert(0,'Sample')
ftl.insert(len(ftl), 'Subgroup')

# create a dataframe with the selected features
df_ft = df[ftl]

#import packages for LogRes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# regions: 110 in total
X = df_ft.iloc[:,1:111]
# labels of the subgroups
y = df_ft.iloc[:,111]

optimisation_results = assignment_slp_v5.optimiseParametersAndFeatures(X, y)
data_loader.save_optimisation_results(optimisation_results, 'optimisation_results.txt')

# split data in train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# import code and packages
import data_loader, assignment_slp_v5
import warnings, os

# create an empty dictionary and train the model
train_few_feature = {}
trained_model = assignment_slp_v5.train(X_train,y_train,0,train_few_feature)

# test the modeled set and create an empty dictionary
test_new_feature = {}
assignment_slp_v5.test(trained_model, X_test, y_test, test_new_feature)

# import the validation file
validation_file = data_loader.prepare_dataset("./input/Validation_call.txt")

# remove the sample and the subgroup columns
ftl.remove('Sample')
ftl.remove('Subgroup')

# extract only certain features from the validation_file
go_validation = validation_file[ftl]

# make predictions
predictions = trained_model.predict(go_validation)

# export the predictions to a file: 'predictions.txt'
import csv
out_put = pd.DataFrame({'Sample': validation_file.index, 'Subgroup': predictions})
out_put.to_csv('./output/predictions.txt', index=None, quotechar='"', sep='\t', quoting=csv.QUOTE_NONNUMERIC)
