'''
model training pseudocode:

load dataset
transform dataset

for 100 iterations:
    split dataset into 3 equal parts
    assign trainingset 2/3 parts
    assign testingset 1/3 part

    find size n of trainingset

    #perform leave-one_out cross-validation
    For n iterations:
        assign validationset unique 1/n part (i.e. not selected previously)
        train model on remaining parts using cross-validation
        validate model on validationset
        determine model score

    determine average model score
'''
import pandas as pd
import random
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import accuracy_score # maybe we don't need this one
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score
from sklearn.model_selection import LeaveOneOut, train_test_split






def loo_cross_validation(training_set):
    """
    In this function, we perform Leave One Out cross validation on the outer loop
    split training set as our original transformed dataset is limited to 100 rows
    """
    loo = LeaveOneOut()
    for train, test in loo.split(training_set):
        pass


    return # CV scores go here

def prepare_dataset(filename):
    dataset = pd.read_csv(filename, delimiter='\t') # input is a .txt file
    dataset['Chromosome'] = dataset.apply(lambda row: # concatenate 3 columns into 1
        "%d_%d_%d"%(row['Chromosome'], row['Start'], row['End']), axis=1)
    dataset.rename(columns={'Chromosome':'Sample'}, inplace=True)
    dataset.drop(['Start', 'End', 'Nclone'], axis=1, inplace=True)

    dataset = dataset.T # transpose as it's more useful to have the call data match the clinical data format

    dataset.columns = dataset.iloc[0] # assign 1st row of data as column headers
    dataset = dataset[1:]   # then ignore the first row

    return dataset

def prepare_target(filename):
    target = pd.read_csv(filename, delimiter='\t')
    target.set_index('Sample', inplace=True)
    return target

def createSVM():
    return svm.SVC(kernel='linear', C=2, gamma='auto')

def createLM():
    return linear_model.LogisticRegression(solver='lbfgs', max_iter=500, multi_class='auto')

if __name__ == "__main__":
    dataset = prepare_dataset('data/train_call.txt')
    target = prepare_target('data/train_clinical.txt')

# for testing:
#    result = pd.merge(dataset, target, left_index=True, right_index=True)
#    print(result['Subgroup'])
    '''classifier = createLM()
    result = cross_validate(classifier, dataset, target["Subgroup"], cv=LeaveOneOut())
    print(result)
    print("test score:", sum(result['test_score']))
    exit()'''

    classifier = svm.SVC(gamma='auto')
    loo = LeaveOneOut()

    '''
    label_preds = cross_val_predict(classifier, dataset, target['Subgroup'], cv=loo)
    score = cross_val_score(classifier, dataset, target['Subgroup'], cv=loo)
    print("score: ", sum(score))
    exit()
    '''
    for i in range(100):
        train, test, train_labels, test_labels = train_test_split(dataset, target, test_size=0.33, random_state=i)
        # checking the dataframe split:
        #print("train: ", train.head())

    # checking the dimentions of the dataframes:
    #print(train.shape, train_labels.shape)
    #print(test.shape, test_labels.shape)

        #num_training_splits = loo.get_n_splits(train)
        #print("Number of times to cross-validate training set: ", num_training_splits)

        '''
        for train_index, test_index in loo.split(train):
            print("TRAIN:", train_index, "TEST:", test_index)'''

        #cross_validate(classifier, train, train_labels["Subgroup"], cv=loo)
        label_preds = cross_val_predict(classifier, train, train_labels["Subgroup"], cv=loo)
        #label_preds = cross_val_predict(classifier, train, train_labels['Subgroup'], cv=loo)
        accuracy = accuracy_score(train_labels["Subgroup"], label_preds)
        print("accuracy_score: ", accuracy)
        #score = cross_val_score(classifier, test, test_labels['Subgroup'], cv=loo)
        #print("iteration #{} score: {}/{} ".format(i, sum(score), 33))
        #print("label_preds: ", label_preds)
        #print("train_labels: ", train_labels["Subgroup"])
