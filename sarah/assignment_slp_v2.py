import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from numpy import mean, std # for mean and standard deviations of match performance


def prepare_dataset(filename):
    dataset = pd.read_csv(filename, delimiter='\t') # import original 'call' file
    # we want to reduce the number of columns by concatenation:
    dataset['Chromosome'] = dataset.apply(lambda row:
        "%d_%d_%d"%(row['Chromosome'], row['Start'], row['End']), axis=1)

    dataset.rename(columns={'Chromosome':'Sample'}, inplace=True)
    dataset.drop(['Start', 'End', 'Nclone'], axis=1, inplace=True)

    dataset = dataset.T # transposed 'call' data to to match the 'clinical' data file

    dataset.columns = dataset.iloc[0] # assign 1st row of dataframe as column headers
    dataset = dataset[1:]   # ...so we can ignore the first row

    return dataset

def prepare_target(filename):
    target = pd.read_csv(filename, delimiter='\t') # import original 'clinical' file
    target.set_index('Sample', inplace=True)
    return target["Subgroup"]

def createSVC():
    return SVC(kernel='linear', gamma='auto')

def createRFC():
    return RandomForestClassifier(n_estimators=100)

def createLRC():
    ''' Comparing test results between different classifiers shows LogisticRegression
    comes up with the highest accuracy score:
    GridSearchCV shows that solver=liblinear and C=1 gives highest mean score (0.81)
    and lowest std (0.08) '''
    return LogisticRegression(solver='liblinear', C=1, multi_class='auto')

def createMLPC():
    return MLPClassifier()

def findOptimalParamsForLRC(dataset, target, cv):
    parameters = {
        'solver' : ('liblinear', 'newton-cg', 'sag', 'saga', 'lbfgs'),
        'C' : range(1,10) }

    classifier = GridSearchCV(createLRC(), parameters, cv=cv)
    classifier.fit(dataset, target)
    print(classifier.cv_results_)


def compareClassifiers(dataset, target):
    cv = StratifiedKFold(n_splits=10)
    classifiers = {
        "SupportVectorClassifier" : createSVC(),
        "RandomForestClassifier" : createRFC(),
        "LogisticRegressionClassifier" : createLRC(),
        "MultiLayerPerceptronClassifier" : createMLPC() }

    for name, classifier in classifiers.items():
        print("Running", name)
        result = cross_validate(classifier, dataset, target, cv=cv)
        print(result)
        print("{} mean score: {}".format(name, result['test_score'].mean()))



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


# The training step (Block 4) - in lieu of scikitlearn's built-in cross_validate function
def train(classifier, dataset, target, random_state):
    skfold = StratifiedKFold(n_splits=10, random_state=random_state)
    training_scores = []

    max_score = 0.0
    optimal_training_fold = None

    for training_fold, validation_fold in skfold.split(dataset, target):
        classifier.fit(dataset.iloc[training_fold], target.iloc[training_fold]) # Actual model training done here
        validation_score = classifier.score(dataset.iloc[validation_fold], target.iloc[validation_fold]) # Trained model applied
        training_scores.append(validation_score)

        if validation_score > max_score:
            max_score = validation_score
            optimal_training_fold = training_fold

    # Train final predictor (Block 3)
    classifier.fit(dataset.iloc[optimal_training_fold], target.iloc[optimal_training_fold]) # Re-training model with the best training fold

    print("Training scores:", training_scores)
    return training_scores

# The validation step (Block 5)
def validate(classifier, dataset, target):
    test_score = classifier.score(dataset, target)

    print("Testing score:", test_score)
    return test_score

if __name__ == "__main__":
    dataset = prepare_dataset('data/train_call.txt')
    target = prepare_target('data/train_clinical.txt')

    #compareClassifiers(dataset[:99], target[:99], classifiers, cv)
    #findOptimalParamsForLRC(dataset, target)
    classifier = createLRC()

    training_scores = []
    testing_scores = []

    for i in range(100):
        print("iteration #", i)
        training_set, test_set, training_labels, test_labels = train_test_split(dataset, target, test_size=0.33, random_state=i)

        curr_training_scores = train(classifier, training_set, training_labels, random_state=i)
        training_scores.append(curr_training_scores)

        curr_testing_score = validate(classifier, test_set, test_labels)
        testing_scores.append(curr_testing_score)
        print("---------------------------------------------------")

    print("Overall training performance: mean {}, stdev {}".format(mean(training_scores), std(training_scores)))
    print("Overall testing performance: mean {}, stdev {}".format(mean(testing_scores), std(testing_scores)))
    print("---------------------------------------------------")
