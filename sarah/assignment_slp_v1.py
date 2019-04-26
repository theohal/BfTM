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
from sklearn import svm, linear_model
from sklearn.model_selection import cross_validate, KFold

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


if __name__ == "__main__":
    dataset = prepare_dataset('data/train_call.txt')
    target = prepare_target('data/train_clinical.txt')

    classifier = svm.SVC(kernel='linear', gamma='auto')
    cv = KFold(n_splits=10)

    result = cross_validate(classifier, dataset, target["Subgroup"], cv=cv)
    print(result)
    print("mean score: ", result['test_score'].mean())
