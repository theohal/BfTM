import data_loader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
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
def train(rfecv, dataset, target):
    rfecv.fit(dataset, target) # Feature selection and model training

    return {"used_number_of_regions" : rfecv.n_features_, # remaining number of regions after elimination
            "used_regions" : list(dataset.columns[rfecv.support_]), # using indices of used features to print column headers (regions)
            "all_region_rankings" : list(zip(dataset.columns, rfecv.ranking_)), # significance ranking position per column (region)
            "training_scores" : list(rfecv.grid_scores_) } # validation scores per iteration

# The validation step (Block 5):
def test(rfecv, dataset, target):
    test_score = rfecv.score(dataset, target)

    print("Testing score:", test_score)
    return test_score

def optimiseFeatures(dataset, target):
    ''' From compare_classifiers.py : comparing test results between different
    classifiers shows that LogisticRegression produces the highest accuracy score.
    Applying GridSearchCV shows that solver=liblinear and C=1 gives the highest
    mean score (~0.75) and the lowest standard deviation (~0.10) '''
    classifier = LogisticRegression(solver='liblinear', C=1, multi_class='auto')

    optimisation_results = []

    for i in range(100):
        print("iteration #", i)
        skfold = StratifiedKFold(n_splits=10, random_state=i)   # Using StratifiedKFold to have a better distribution of subtypes in the split groups
        rfecv = RFECV(estimator=classifier, cv=skfold, n_jobs=-1) # Recursive Feature Elimination for training the model

        training_set, test_set, training_labels, test_labels = train_test_split(dataset, target, test_size=0.33, random_state=i)

        curr_results = train(rfecv, training_set, training_labels)  # using 2/3 original dataset - applying in the inner loop

        curr_testing_score = test(rfecv, test_set, test_labels) # using 1/3 original dataset
        curr_results['test_score'] = curr_testing_score   # appending test score to the current results dictionary

        # add the iteration number for random_state reproducibility after analysis
        curr_results['iteration'] = i   # appending iteration number to the current results dictionary

        print(curr_results) # run summary per iteration
        optimisation_results.append(curr_results)
        print("---------------------------------------------------")

    return optimisation_results


if __name__ == "__main__":
    dataset = data_loader.load_dataset()
    target = data_loader.load_target()

    optimisation_results = optimiseFeatures(dataset, target)
    data_loader.save_optimisation_results(optimisation_results)
