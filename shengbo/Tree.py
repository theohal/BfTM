import data_loader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, RFECV
from numpy import mean, std
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# The training step (Block 4) - in lieu of scikitlearn's built-in cross_validate function:
def train(rfecv, dataset, target):
    rfecv.fit(dataset, target) # Feature selection and model training

    print("Number of regions used:", rfecv.n_features_) # remaining number of regions after elimination
    print("Regions used:", dataset.columns[rfecv.support_]) # using indices of used features to print column headers (regions)
    print("Region rankings:", rfecv.ranking_)   # significance ranking position per column (region)
    print("Training scores:", rfecv.grid_scores_) # validation scores per iteration

    return {"training_scores" : rfecv.grid_scores_,
            "training_scores_mean" : mean(rfecv.grid_scores_),
            "training_scores_std" : std(rfecv.grid_scores_),
            "region_rankings" : rfecv.ranking_,
            "used_regions_by_index" : rfecv.support_,
            "used_regions_by_name" : dataset.columns[rfecv.support_] }


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
    classifier = ExtraTreesClassifier()

    optimisation_results = []
    i = 0
    for i in range(100):
        print("iteration #", i)
        # Using StratifiedKFold to have a better distribution of subtypes in the split groups
        skfold = StratifiedKFold(n_splits=10, random_state=i)
        # Recursive Feature Elimination for training the model
        rfecv = RFECV(estimator=classifier, cv=skfold)

        training_set, test_set, training_labels, test_labels = train_test_split(dataset, target, test_size=0.33, random_state=i)

        # using 2/3 original dataset - applying in the inner loop
        curr_results = train(rfecv, training_set, training_labels)

        # using 1/3 original dataset
        curr_testing_score = test(rfecv, test_set, test_labels)
        # appending test score to the current results dictionary
        curr_results['test_score'] = curr_testing_score

        # add the iteration number for random_state reproducibility after analysis
        # appending iteration number to the current results dictionary
        curr_results['iteration'] = i

        # run summary per iteration
        print(curr_results)
        optimisation_results.append(curr_results)
        print("---------------------------------------------------")

    return optimisation_results


if __name__ == "__main__":
    dataset = data_loader.load_dataset()
    target = data_loader.load_target()

    optimisation_results = optimiseFeatures(dataset, target)
    data_loader.save_optimisation_results(optimisation_results)
