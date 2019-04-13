import numpy as np
import pandas as pd


md_TC = pd.read_csv('~/BfTM/modified_train_call.csv')
TClinical = pd.read_csv('~/BfTM/input/Train_clinical.txt', sep='\t')

result = pd.merge(md_TC, TClinical[['Sample', 'Subgroup']],
                  on=['Sample'], how='left')


features_list = np.array(result.columns).tolist()
remove_features = list()
for feature in features_list:
    len_feature = len(pd.unique(result[feature]))
    if len_feature == 1:
        remove_features.append(feature)

len(remove_features)
result[remove_features[0]]


def diff_count(curr_list, next_list):
    len_list = len(curr_list)
    diff = 0
    for i in range(len_list):
        if curr_list[i] != next_list[i]:
            diff += 1

    return diff


same_features = list()
for i in range(len(features_list)-1):
    feature_curr = features_list[i]
    feature_next = features_list[i+1]
    if diff_count(result[feature_curr], result[feature_next]) == 0:
        same_features.append(feature_curr)
        same_features.append(feature_next)


curr_features = list()
next_features = list()
rename_features = list()
len_features = len(features_list)
i = 0
while i < (len_features-1):
    feature_curr = features_list[i]
    feature_next = features_list[i+1]
    if diff_count(result[feature_curr], result[feature_next]) == 0:
        feature_rename = feature_curr + '-' + feature_next
        curr_features.append(feature_curr)
        next_features.append(feature_next)
        rename_features.append(feature_rename)
    i += 1


for i in range(len(next_features)):
    # result.drop(columns=curr_features[i], inplace=True)
    result.rename(columns={next_features[i]: rename_features[i]}, inplace=True)
