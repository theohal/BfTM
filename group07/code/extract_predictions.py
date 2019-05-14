import pandas as pd
import numpy as np
from sklearn.externals import joblib

# import original dataset
df = pd.read_csv('./input/df_sorted_Subgroup.csv')

# regions: 110 in total
X = df.iloc[:,2:2836]
# labels of the subgroups
y = df.iloc[:,2836]

# import code and packages
import data_loader, assignment_slp_v5
import warnings, os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=712)

# create an empty dictionary and train the model
train_few_feature = {}
trained_model = assignment_slp_v5.train(X_train,y_train,712,train_few_feature)

# import the validation file
validation_file = data_loader.prepare_dataset("Validation_call.txt")

# extract only certain features from the validation_file
go_validation = validation_file.iloc[:,0:]
go_validation.shape

# make predictions
predictions = LogRegmodel.predict(validation_file)

# export the predictions to a file: 'predictions.txt'
import csv
out_put = pd.DataFrame({'Sample': validation_file.index, 'Subgroup': predictions})
out_put.to_csv('predictions.txt', index=None, quotechar='"', sep='\t', quoting=csv.QUOTE_NONNUMERIC)
