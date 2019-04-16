import numpy as np
import pandas as pd


dataset = pd.read_csv('merge_features.csv')


def split_into_three_datasets(dataset):
    """
    In this function, we should split the input dataset into 3 datasets:
    traing dataset: 60 samples
    validation dataset: 10 samples (for selecting paremeters)
    testing dataset: 30 samples (only test once)
    All datasets should be include random equal samples from each subgroup
    """
    training = {}
    validation = {}
    testing = {}

    return training, validation, testing


def calculate_FDR(test, control):
    """
    In this function, we need to calculate the FDR for test.
    Output should be significant features (regions) for test
    """
    significant_features = []

    return significant_features


def cross_validation():
    """
    
    """
