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


len_features = len(features_list)
i = 0
name_changed = False
feature_rename = ""
while i < (len_features-1):
    if name_changed:
        feature_curr = feature_rename
        name_changed = False
    else:
        feature_curr = features_list[i]
    feature_next = features_list[i+1]
    if diff_count(result[feature_curr], result[feature_next]) == 0:
        feature_rename = feature_curr + '-' + feature_next
        name_changed = True
        result.drop(columns=feature_curr, inplace=True)
        result.rename(columns={feature_next: feature_rename}, inplace=True)
    i += 1

result.to_csv('merge_features.csv', index=False)


# feature selection
HER2_data = result[result['Subgroup'] == 'HER2+']
HR_data = result[result['Subgroup'] == 'HR+']
TN_data = result[result['Subgroup'] == 'Triple Neg']
