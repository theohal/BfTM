import data_loader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


subtypes = ['HER2+', 'HR+', 'Triple Neg']
subtype = 'HER2+'


def subtype_select(subtype):
    """
    this function returns Features (X) and labels (y) representing subtype.
    """
    X = data_loader.load_dataset()
    y = data_loader.load_target()

    X = X.subtract(X.mean())

    subtypes = ['HER2+', 'HR+', 'Triple Neg']
    subtypes.remove(subtype)

    y = y.replace(subtype, 1)
    y = y.replace(subtypes, 0)

    return X, y


def feature_weights(subtype, threshold):
    """
    """
    X, y = subtype_select(subtype)
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0)
    classifier = LogisticRegressionCV(class_weight='balanced')
    classifier.fit(X_train, y_train)

    # LogisticRegressionCV(Cs=10, class_weight='balanced', cv='warn', dual=False,
    #            fit_intercept=True, intercept_scaling=1.0, max_iter=100,
    #            multi_class='warn', n_jobs=None, penalty='l2',
    #            random_state=None, refit=True, scoring=None, solver='lbfgs',
    #            tol=0.0001, verbose=0)

    feature_importances = pd.DataFrame({'feature': X_train.columns,
                                        'coef': classifier.coef_[0]})

    feature_importances['coef_mean'] = feature_importances['coef'].divide(feature_importances['coef'].abs().max())
    # return feature_importances

    feature_importances['coef_abs'] = feature_importances['coef_mean'].abs()
    feature_importances = feature_importances.sort_values(by='coef_abs', ascending=False)

    features = feature_importances[feature_importances['coef_abs'] > threshold]['feature'].values

    return features


# features_HER2['coef_abs'] = features_HER2['coef'].abs()
# feature_importances = features_HER2.sort_values(by='coef_abs', ascending=False)
# feature_importances[feature_importances['coef_abs'] > 0.1]['feature'].values


features_HER2 = feature_weights('HER2+', 0.2)

plt.plot(features_HER2['coef_mean'])


features_HR = feature_weights('HR+', 0.4)
plt.plot(features_HR['coef_mean'])

features_TN = feature_weights('Triple Neg', 0.4)
plt.plot(features_TN['coef_mean'])


len(features_HER2), len(features_HR), len(features_TN)

final_features = list(set(features_HER2.tolist() + features_HR.tolist() + features_TN.tolist()))
df_features = pd.DataFrame({'features': final_features})
df_features.to_csv('final_features.csv', index=False)
len(final_features)



validation_filename = 'data/Validation_call.txt'
validation = data_loader.prepare_dataset(validation_filename)
