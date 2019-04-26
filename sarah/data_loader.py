import pandas as pd

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

def load_dataset():
    dataset = prepare_dataset('data/train_call.txt')
    return dataset

def load_target():
    target = prepare_target('data/train_clinical.txt')
    return target
