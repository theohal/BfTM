import data_loader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def select_subtype(subtype):
    """
    this function returns Features (X) and labels (y) representing subtype.
    """
    X = data_loader.load_dataset()
    y = data_loader.load_target()

    # apply mean centering to for each region
    X = X.subtract(X.mean())

    # all subtypes
    subtypes = ['HER2+', 'HR+', 'Triple Neg']
    # remove the current subtype from the whole subtype set
    subtypes.remove(subtype)

    # re-coding the subtypes to 0 and 1
    y = y.replace(subtype, 1)
    y = y.replace(subtypes, 0)

    return X, y


def select_features(subtype, threshold):
    """
    this function returns the most important features within a input threshold,
    using the LogisticRegressionCV to fit the data,
    feature importance is ranked by absolute coefficient
    """
    X, y = select_subtype(subtype)
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

    # calculate the absolute values
    feature_importances['coef_abs'] = feature_importances['coef'].abs()
    # find out the largest absolute value.
    coef_abs_max = feature_importances['coef_abs'].max()
    # normalize the coefficient by dividing the largest absolute value.
    feature_importances['coef_nor'] = feature_importances['coef'].divide(coef_abs_max)
    # return feature_importances

    # calculate the absolute values of normalized values
    feature_importances['coef_nor_abs'] = feature_importances['coef_nor'].abs()
    # ranking features by normalized absolute coefficient values
    feature_importances = feature_importances.sort_values(by='coef_nor_abs', ascending=False)

    # select features larger than threshold
    features = feature_importances[feature_importances['coef_nor_abs'] > threshold]['feature'].values

    return features


def save_selected_features(th_her2, th_hr, th_tn):
    """
    this function will save selected features into a csv file
    """
    features_HER2 = select_features('HER2+', th_her2)
    features_HR = select_features('HR+', th_hr)
    features_TN = select_features('Triple Neg', th_tn)

    # remove the duplicated features
    final_features = list(set(features_HER2.tolist() + features_HR.tolist() + features_TN.tolist()))

    df_features = pd.DataFrame({'features': final_features})
    df_features.to_csv('final_features.csv', index=False)


def feature_coefficient(subtype):
    """
    this function returns all normalized coefficient of features,
    using the LogisticRegressionCV to fit the data
    """
    X, y = select_subtype(subtype)
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0)
    classifier = LogisticRegressionCV(class_weight='balanced')
    classifier.fit(X_train, y_train)

    feature_importances = pd.DataFrame({'feature': X_train.columns,
                                        'coef': classifier.coef_[0]})

    # calculate the absolute values
    feature_importances['coef_abs'] = feature_importances['coef'].abs()
    # find out the largest absolute value.
    coef_abs_max = feature_importances['coef_abs'].max()
    # normalize the coefficient by dividing the largest absolute value.
    feature_importances['coef_nor'] = feature_importances['coef'].divide(coef_abs_max)

    return feature_importances


def plot_feature_importance(subtype):
    """
    """
    features = feature_coefficient(subtype)
    # plot the features
    plt.plot(features['coef_nor'])


# validation_filename = 'data/Validation_call.txt'
# validation = data_loader.prepare_dataset(validation_filename)
