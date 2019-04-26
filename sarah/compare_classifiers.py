import data_loader
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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
        'solver' : ('liblinear', 'newton-cg', 'lbfgs'),
        #'solver' : ('liblinear', 'newton-cg', 'sag', 'saga', 'lbfgs'),
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
        #'solver' : ('adam', 'lbfgs', 'sgd'),
        'tol': (1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
        'epsilon': (1e-3, 1e-7, 1e-8, 1e-9, 1e-8)
    }

    return createGridSearchCV(classifier, parameters)

def runGridSearch(dataset, target, name, classifier):
    print("Running grid search on", name)

    classifier.fit(dataset[:70], target[:70])
    test_score = classifier.score(dataset[70:100], target[70:100])

    #print(classifier.cv_results_['mean_test_score'])
    #print(classifier.cv_results_['std_test_score'])

    return {'name' : name,
            'cv_results' : classifier.cv_results_,
            'classifier' : classifier.best_estimator_,
            'parameters' : classifier.best_params_,
            'test_score' : test_score }

def compareClassifiers(dataset, target):
    results = []

    SVC_results = runGridSearch(dataset, target, "SupportVectorClassifier", prepareGridSearchForSVC())
    results.append(SVC_results)

    RFC_results = runGridSearch(dataset, target, "RandomForestClassifier", prepareGridSearchForRFC())
    results.append(RFC_results)

    LRC_results = runGridSearch(dataset, target, "LogisticRegressionClassifier", prepareGridSearchForLRC())
    results.append(LRC_results)

    MLPC_results = runGridSearch(dataset, target, "MultiLayerPerceptronClassifier", prepareGridSearchForMLP())
    results.append(MLPC_results)

    # return result with the highest test score
    ranked_results = sorted(results, key=lambda result: result['test_score'])

    for index, result in enumerate(ranked_results):
        print("#{}: {} (test_score: {}) with parameters {}", index, result['name'], result['test_score'], result['parameters'])

    return ranked_results[0]


if __name__ == "__main__":
    dataset = data_loader.load_dataset()
    target = data_loader.load_target()

    result = compareClassifiers(dataset, target)

    print("---------------------------------------------------")
    print("Optimal classifier: {} with parameters {}".format(result['name'], result['parameters']))
    print("Test score:", result['test_score'])
