import data_loader
import warnings
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from numpy import mean, std # for mean and standard deviations of match performance

# Try running gridsearch for all of the classifiers!
def createGridSearchCV(classifier, parameters):
    return GridSearchCV(classifier, parameters, iid=False, cv=StratifiedKFold(n_splits=10))

def prepareGridSearchForSVC():
    classifier = SVC()
    parameters = {
        'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),
        #'kernel' : ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'),
        'gamma' : (0.01, 0.1, 1, 10, 100),
        'C' : (0.1, 1, 10, 100, 1000) }

    return createGridSearchCV(classifier, parameters)

def prepareGridSearchForLRC():
    classifier = LogisticRegression(multi_class='auto')
    parameters = {
        'solver' : ('liblinear', 'newton-cg', 'sag', 'saga', 'lbfgs'),
        'C' : (0.1, 1, 10, 100, 1000) }

    return createGridSearchCV(classifier, parameters)

def prepareGridSearchForRFC():
    classifier = RandomForestClassifier(n_estimators=100)
    parameters = {
        'max_features' : ('log2', 'sqrt', 0.2),
        'n_estimators' : (1, 10, 100, 200, 500, 1000),
        'min_samples_leaf' : (10, 30, 50, 70, 90),
        'bootstrap' : ('True', 'False')
    }

    return createGridSearchCV(classifier, parameters)

def prepareGridSearchForMLP():
    classifier = MLPClassifier()
    parameters = {
        'hidden_layer_sizes': [(10, 10), (30, 30, 30), (100, 100), (100, 100, 100)],
        #'solver' : ('adam', 'lbfgs', 'sgd'),       # this makes the run time very long (70+ mins)
        'tol': (1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
        'epsilon': (1e-3, 1e-7, 1e-8, 1e-9, 1e-8)
    }

    return createGridSearchCV(classifier, parameters)

def runGridSearch(dataset, target, name, classifier):
    print("Running grid search on", name)

    classifier.fit(dataset[:70], target[:70])   # ~2/3 data split
    test_score = classifier.score(dataset[70:100], target[70:100])  # ~1/3 data split

    print(classifier.cv_results_['mean_test_score'])
    print(classifier.cv_results_['std_test_score'])

    print("Best training result: {} (index: {})".format(classifier.best_score_, classifier.best_index_))
    print("---------------------------------------------------")

    return {'name' : name,
            'cv_results' : classifier.cv_results_,
            'classifier' : classifier.best_estimator_,
            'parameters' : classifier.best_params_,
            'max_training_score' : classifier.best_score_,
            'test_score' : test_score}

def compareClassifiers(dataset, target):
    results = []

    # To get rid of ConvergenceWarnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        SVC_results = runGridSearch(dataset, target, "SupportVectorClassifier", prepareGridSearchForSVC())
        results.append(SVC_results)

        RFC_results = runGridSearch(dataset, target, "RandomForestClassifier", prepareGridSearchForRFC())
        results.append(RFC_results)

        LRC_results = runGridSearch(dataset, target, "LogisticRegressionClassifier", prepareGridSearchForLRC())
        results.append(LRC_results)

        MLPC_results = runGridSearch(dataset, target, "MultiLayerPerceptronClassifier", prepareGridSearchForMLP())
        results.append(MLPC_results)

    # return result with the highest test score first
    ranked_results = sorted(results, key=lambda result: result['test_score'], reverse=True)

    for index, result in enumerate(ranked_results, start=1):
        print("#{}: {} (test_score: {}) with parameters {}".format(index, result['name'], result['test_score'], result['parameters']))

    return ranked_results


if __name__ == "__main__":
    dataset = data_loader.load_dataset()
    target = data_loader.load_target()

    comparison_results = compareClassifiers(dataset, target)
    data_loader.save_comparison_results(comparison_results)

    optimal_classifier = max(comparison_results, key=lambda result: result['test_score'])

    print("---------------------------------------------------")
    print("Optimal classifier: {} with parameters {}".format(optimal_classifier['name'], optimal_classifier['parameters']))
    print("Test score:", optimal_classifier['test_score'])
    print("---------------------------------------------------")
