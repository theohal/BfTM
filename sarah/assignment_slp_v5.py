import data_loader
import warnings, os
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.exceptions import ConvergenceWarning
from numpy import mean, std # for mean and standard deviations of match performance

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
    re-train model based on trainingfold with highest trainingscore from training run
    determine model testscore by validating trained model on testingset


'''


# The training step (Block 4) - in lieu of scikitlearn's built-in cross_validate function:
def train(dataset, target, random_state, result_record):
    ''' prepare the classifier for training, then use GridSearchCV
    for parameter optimisation and RFECV for feature elimination and fitting '''
    classifier = LogisticRegression(multi_class='auto', random_state=random_state, n_jobs=-1)

    parameters = {
        'solver' : ('liblinear', 'newton-cg', 'sag', 'saga', 'lbfgs'),
        'C' : (0.1, 1, 10, 100, 1000) }

    skfold = StratifiedKFold(n_splits=10, random_state=random_state)   # Using StratifiedKFold to have a better distribution of subtypes in the split groups

    # To get rid of ConvergenceWarnings (fail to converge) and UserWarnings (n_jobs n/a for liblinear)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore" # also subprocesses

        gridcv = GridSearchCV(classifier, parameters, iid=False, cv=skfold, n_jobs=-1)
        gridcv.fit(dataset, target) # hyperparameter optimisation

        rfecv = RFECV(estimator=gridcv.best_estimator_, cv=skfold, n_jobs=-1) # Recursive Feature Elimination for training the model
        rfecv.fit(dataset, target) # Feature selection and model training

        os.environ["PYTHONWARNINGS"] = "default" #re-enable warnings

    # store results in the result record provided as argument to this function
    result_record.update({
        'iteration' : random_state, # appending random_state used to the current results dictionary for reproducibility after analysis
        'grid_search_cv_results' : gridcv.cv_results_, # detailed results of the grid_search cv
        'best_params' : gridcv.best_params_, # best classifier parameters (from grid search)
        'used_number_of_regions' : rfecv.n_features_, # remaining number of regions after elimination
        'used_regions' : list(dataset.columns[rfecv.support_]), # using indices of used features to print column headers (regions)
        'all_region_rankings' : list(zip(dataset.columns, rfecv.ranking_)), # significance ranking position per column (region)
        'refcv_training_scores' : list(rfecv.grid_scores_)}) # validation scores per iteration

    # return the trained model
    return rfecv

# The validation step (Block 5):
def test(trained_model, dataset, target, result_record):
    # the trained model here is an RFECV instance with the best_estimator from the gridsearch
    test_score = trained_model.score(dataset, target)

    print("Testing score:", test_score)
    result_record.update({'test_score' : test_score}) # appending test score to the current results dictionary

def optimiseParametersAndFeatures(dataset, target):
    optimisation_results = []

    for i in range(100):
        print("iteration #", i)

        training_set, test_set, training_labels, test_labels = train_test_split(dataset, target, test_size=0.33, random_state=i)

        # create a dictionary to store the train and test results in (as used in below functions)
        curr_results = {}

        trained_model = train(training_set, training_labels, i, curr_results)  # using 2/3 original dataset - applying in the inner loop

        test(trained_model, test_set, test_labels, curr_results) # using 1/3 original dataset

        optimisation_results.append(curr_results) # store run summary of each iteration

        print(curr_results) # print run summary per iteration
        print("---------------------------------------------------")

    return optimisation_results


if __name__ == "__main__":
    dataset = data_loader.load_dataset()
    target = data_loader.load_target()

    optimisation_results = optimiseParametersAndFeatures(dataset, target)
    data_loader.save_optimisation_results(optimisation_results)
