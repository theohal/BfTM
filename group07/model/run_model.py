#!/usr/bin/env python3
"""Reproduce your result by your saved model.

This is a script that helps reproduce your prediction results using your saved
model. This script is unfinished and you need to fill in to make this script
work. If you are using R, please use the R script template instead.

The script needs to work by typing the following commandline (file names can be
different):

python3 run_model.py -i unlabelled_sample.txt -m model.pkl -o output.txt

"""

# author: Chao (Cico) Zhang
# date: 31 Mar 2017

import argparse
import sys
# Start your coding

# import the library you need here
import csv
import pandas as pd
from sklearn.externals import joblib


def prepare_dataset(filename):
    # import original 'call' file
    dataset = pd.read_csv(filename, delimiter='\t')
    # we want to reduce the number of columns by concatenation:
    dataset['Chromosome'] = dataset.apply(
        lambda row: "%d_%d_%d" % (row['Chromosome'], row['Start'], row['End']),
        axis=1)

    dataset.rename(columns={'Chromosome': 'Sample'}, inplace=True)
    dataset.drop(['Start', 'End', 'Nclone'], axis=1, inplace=True)
    # transposed 'call' data to to match the 'clinical' data file
    dataset = dataset.T
    # assign 1st row of dataframe as column headers
    dataset.columns = dataset.iloc[0]
    # ...so we can ignore the first row
    dataset = dataset[1:]

    return dataset


# End your coding


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Reproduce the prediction')
    parser.add_argument('-i', '--input', required=True, dest='input_file',
                        metavar='unlabelled_sample.txt', type=str,
                        help='Path of the input file')
    parser.add_argument('-m', '--model', required=True, dest='model_file',
                        metavar='model.pkl', type=str,
                        help='Path of the model file')
    parser.add_argument('-o', '--output', required=True,
                        dest='output_file', metavar='output.txt', type=str,
                        help='Path of the output file')
    # Parse options
    args = parser.parse_args()

    if args.input_file is None:
        sys.exit('Input is missing!')

    if args.model_file is None:
        sys.exit('Model file is missing!')

    if args.output_file is None:
        sys.exit('Output is not designated!')

    # Start your coding

    # suggested steps
    # Step 1: load the model from the model file
    model = joblib.load(args.model_file)
    # Step 2: apply the model to the input file to do the prediction
    validation = prepare_dataset(args.input_file)
    # Step 3: write the prediction into the desinated output file
    predictions = model.predict(validation)
    # export the predictions to a file: 'predictions.txt'
    out_put = pd.DataFrame({'Sample': validation.index, 'Subgroup': predictions})
    out_put.to_csv(args.output_file, index=None, quotechar='"', sep='\t', quoting=csv.QUOTE_NONNUMERIC)
    # End your coding


if __name__ == '__main__':
    main()
