import data_loader
import warnings, os
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.exceptions import ConvergenceWarning
# for mean and standard deviations of match performance
from numpy import mean, std

''' Implementing cross-validation as per L.F.A. Wessels et al. paper (Figure 1)

Pseudocode:

load dataset
transform dataset

for 100 iterations:
    split transformed dataset into 3 equal parts
    assign trainingset 2/3 parts
    assign testingset 1/3 part

    For 10 iterations on trainingset:
        split trainingset into trainingfold (9/10) and validationfold (1/10)
        [use StratifiedKFold (10 fold) to improve distribution of subtypes in train/test sets]
        train model on trainingfold
        determine model trainingscore by validating model on validationfold

    determine average of 10 model trainingscores
    re-train model based on trainingfold with highest trainingscore
        from training run determine model testscore
        by validating trained model on testingset
'''


# The training step (Block 4)
# in lieu of scikitlearn's built-in cross_validate function:
def train(dataset, target, random_state):
    '''prepare the classifier for training, then use GridSearchCV
    for parameter optimisation and RFECV for feature elimination and fitting'''
    classifier = LogisticRegression(multi_class='auto', random_state=random_state, n_jobs=-1)

    parameters = {
            'solver': ('liblinear', 'newton-cg', 'sag', 'saga', 'lbfgs'),
            'C': (0.1, 1, 10, 100, 1000)}
    # Using StratifiedKFold to have a better distribution
    # of subtypes in the split groups
    skfold = StratifiedKFold(n_splits=10, random_state=random_state)

    # To get rid of ConvergenceWarnings (fail to converge)
    # and UserWarnings (n_jobs n/a for liblinear)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # also subprocesses
        os.environ["PYTHONWARNINGS"] = "ignore"

        gridcv = GridSearchCV(classifier, parameters, iid=False, cv=skfold, n_jobs=-1, return_train_score=True)
        # hyperparameter optimisation
        gridcv.fit(dataset, target)
        # Recursive Feature Elimination for training the model
        rfecv = RFECV(estimator=gridcv.best_estimator_, cv=skfold, n_jobs=-1)
        # Feature selection and model training
        rfecv.fit(dataset, target)

        # re-enable warnings
        os.environ["PYTHONWARNINGS"] = "default"

    records = {
        # "random_state" : random_state, # appending random_state used to the current results dictionary for reproducibility after analysis
        "mean_test_score": gridcv.cv_results_['mean_test_score'],
        'std_test_score': gridcv.cv_results_['std_test_score'],
        "best_params" : gridcv.best_params_, # best classifier parameters (from grid search)
        # "used_number_of_regions" : rfecv.n_features_, # remaining number of regions after elimination
        "used_regions" : list(dataset.columns[rfecv.support_]), # using indices of used features to print column headers (regions)
        # "all_region_rankings" : list(zip(dataset.columns, rfecv.ranking_)), # significance ranking position per column (region)
        "training_scores" : list(rfecv.grid_scores_)
        }
    # store results in the result record provided as argument to this function
    # result_record.append(records) # validation scores per iteration

    # return the trained model
    return records, rfecv


# The validation step (Block 5):
def test(trained_model, dataset, target):
    # the trained model here is an RFECV instance
    # with the best_estimator from the gridsearch
    test_score = trained_model.score(dataset, target)

    # print("Testing score:", test_score)
    # appending test score to the current results dictionary
    # result_record.append(test_score)

    return test_score


def optimiseParametersAndFeatures(dataset, target):
    # optimisation_results = []
    curr_results = []
    result_record = []
    models = []
    for i in range(100):
        print("iteration #", i)

        training_set, test_set, training_labels, test_labels = train_test_split(dataset, target, test_size=0.33, random_state=i)

        # create a dictionary to store the train and test results
        # in (as used in below functions)
        # using 2/3 original dataset - applying in the inner loop
        records, rfecv = train(training_set, training_labels, i)
        curr_results.append(records)
        models.append(rfecv)
        # using 1/3 original dataset
        result_record.append(test(rfecv, test_set, test_labels))
        # store run summary of each iteration
        # optimisation_results.append(curr_results)
        # print run summary per iteration
        # print(curr_results)
        # print("---------------------------------------------------")

    return curr_results, result_record, models


if __name__ == "__main__":
    dataset = data_loader.load_dataset()
    target = data_loader.load_target()

    optimisation_results = optimiseParametersAndFeatures(dataset, target)
    data_loader.save_optimisation_results(optimisation_results)
