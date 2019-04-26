import data_loader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
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

    # Train final predictor (Block 3):
    classifier.fit(dataset.iloc[optimal_training_fold], target.iloc[optimal_training_fold]) # Re-training model with the best training fold

    print("Training scores:", training_scores)
    return training_scores

# The validation step (Block 5):
def validate(classifier, dataset, target):
    test_score = classifier.score(dataset, target)

    print("Testing score:", test_score)
    return test_score


if __name__ == "__main__":
    dataset = data_loader.load_dataset()
    target = data_loader.load_target()

    ''' Comparing test results between different classifiers shows LogisticRegression
    comes up with the highest accuracy score:
    GridSearchCV shows that solver=liblinear and C=1 gives highest mean score (0.81)
    and lowest std (0.08) '''
    classifier = LogisticRegression(solver='liblinear', C=1, multi_class='auto')

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
