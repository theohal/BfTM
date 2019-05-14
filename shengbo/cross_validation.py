import numpy as np
import pandas as pd
import random


def split_into_three_datasets(dataset):
    """
    In this function, we split the input dataset into 3 datasets:
    training dataset: 60% samples
    validation dataset: 10% samples (for selecting paremeters)
    testing dataset: 30% samples (only test once)
    All datasets include random samples from each subgroup

    The assumption is that there are 100 rows in the dataset but this function
    accounts for variable-sized datasets (however there are possible rounding errors)
    """

    randomised_dataset = dataset.sample(frac=1)
    dataset_size = len(dataset)

    train_proportion = int(dataset_size * 0.6)
    valid_proportion = int(dataset_size * 0.1)
    test_proportion = int(dataset_size * 0.3)

    training = randomised_dataset[0:train_proportion] # row range is [0:60]
    validation = randomised_dataset[train_proportion:train_proportion+valid_proportion] # row range is [60:70]
    testing = randomised_dataset[train_proportion+valid_proportion:] # row range is [70:end]

    return training, validation, testing


def calculate_FDR(test, control):
    """
    In this function, we need to calculate the FDR for test.
    Output should be significant features (regions) for test
    """
    significant_features = []

    return significant_features


def cross_validation(model, K):
    """
    In this function, we do cross validation for input model

    """

# This part is just for testing purposes
if __name__ == "__main__":
    dataset = pd.read_csv('merge_features.csv')
    training, validation, testing = split_into_three_datasets(dataset)
    print("Training Set:")
    print(training)
    print("Validation Set")
    print(validation)
    print("Testing Set")
    print(testing)
