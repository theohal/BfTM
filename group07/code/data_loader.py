import pandas as pd
from sklearn.externals import joblib

# list of all files involved
# input data files
dataset_filename = 'data/train_call.txt'
target_filename = 'data/train_clinical.txt'

# model file
# haven't created this yet!
model_filename = 'model/model.pkl'

# output files (for analysis purposes)
comparison_results_filename = 'results/comparison.txt'
optimisation_results_filename = 'results/optimisation.txt'
region_rankings_filename = 'results/region_rankings.txt'
iteration_rankings_filename = 'results/iteration_rankings.txt'


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


def prepare_target(filename):
    # import original 'clinical' file
    target = pd.read_csv(filename, delimiter='\t')
    target.set_index('Sample', inplace=True)

    return target["Subgroup"]


def load_dataset():
    dataset = prepare_dataset(dataset_filename)
    return dataset


def load_target():
    target = prepare_target(target_filename)
    return target


def save_classifier(classifier):
    joblib.dump(classifier, model_filename)


def load_classifier():
    classifier = joblib.load(model_filename)
    return classifier


def save_object_to_file(object, filename):
    with open(filename, 'w') as file:
        print(object, file=file)


def load_object_from_file(filename):
    with open(filename, 'r') as file:
        object = eval(file.read())
        return object


def save_comparison_results(results):
    save_object_to_file(results, comparison_results_filename)


def load_comparison_results():
    return load_object_from_file(comparison_results_filename)


def save_optimisation_results(results):
    save_object_to_file(results, optimisation_results_filename)


def load_optimisation_results():
    return load_object_from_file(optimisation_results_filename)


def save_region_rankings(region_rankings):
    save_object_to_file(region_rankings, region_rankings_filename)


def load_region_rankings():
    return load_object_from_file(region_rankings_filename)


def save_iteration_rankings(iteration_rankings):
    save_object_to_file(iteration_rankings, iteration_rankings_filename)


def load_iteration_rankings():
    return load_object_from_file(iteration_rankings_filename)
