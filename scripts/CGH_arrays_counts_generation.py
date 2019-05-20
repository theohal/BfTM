import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
%matplotlib inline
from collections import Counter

df = pd.read_csv('./df_sorted_Subgroup.csv')

pred = pd.read_csv('./predictions_100.csv')

# separate df to three different datasets
her2df = df[0:32]
hrdf = df[32:68]
tndf = df[68:100]

her2df.shape
hrdf.shape
tndf.shape

# create a function to plot array CGH per type
def plot_ptype(dataframe):
    '''
    input: dataframe of a particular type
    '''
    # add a new dictionary that will store the counts per CGH values
    final_dict = CGH_counts = {}
    CGH_counts[-1] = CGH_counts[0] = CGH_counts[1] = CGH_counts[2] = count = 0
    # turn the dataframe into list
    column_list = list(dataframe.columns[2:2836])
    # set the starting value for the first chromosome
    pre_chr = 1
    # iterate through the columns and count the CGH array values
    for i in column_list:
        # check the chromosome
        chr_num = int(column_list[count].split("_")[0])
        # create a statement to check if chromosome has changed
        if i == column_list[-1]:
            vpcl = values_per_column(dataframe, i)
            # add the values to the whole
            CGH_counts[-1] = CGH_counts[-1] + vpcl[-1]
            CGH_counts[0] = CGH_counts[0] + vpcl[0]
            CGH_counts[1] = CGH_counts[1] + vpcl[1]
            CGH_counts[2] = CGH_counts[2] + vpcl[2]

            #dict_chr_number = {chr_num}
            chr_name = 'chr'+str(chr_num)
            final_dict[chr_name] = CGH_counts
            CGH_counts = {}
            CGH_counts[-1] = CGH_counts[0] = CGH_counts[1] = CGH_counts[2] = 0
            break
        elif pre_chr == chr_num:
            pass
        else:
            #dict_chr_number = {chr_num}
            chr_name = 'chr'+str(pre_chr)
            final_dict[chr_name] = CGH_counts
            CGH_counts = {}
            CGH_counts[-1] = CGH_counts[0] = CGH_counts[1] = CGH_counts[2] = 0
        # set the previous chromosome equal to the new
        pre_chr = chr_num
        # create an empty dictionary to store the values per column
        vpc = values_per_column(dataframe,i)

        # add the values to the whole
        CGH_counts[-1] = CGH_counts[-1] + vpc[-1]
        CGH_counts[0] = CGH_counts[0] + vpc[0]
        CGH_counts[1] = CGH_counts[1] + vpc[1]
        CGH_counts[2] = CGH_counts[2] + vpc[2]

        count += 1

    fix_error_chr1 = {}
    fix_error_chr1[-1] = final_dict[-1]
    fix_error_chr1[0] = final_dict[0]
    fix_error_chr1[1] = final_dict[1]
    fix_error_chr1[2] = final_dict[2]

    final_dict.pop(-1,None)
    final_dict.pop(0,None)
    final_dict.pop(1,None)
    final_dict.pop(2,None)
    final_dict.pop('chr1',None)

    final_dict['chr1'] = fix_error_chr1

    return final_dict

def values_per_column(dataframe, column_string):
    '''
    give as input the dataframe you want to use and the string of the
    column that you want to be counted
    '''
    values_column = {}
    values_column = dataframe[str(column_string)].value_counts()
    if -1 not in values_column:
        values_column[-1] = 0
    if 0 not in values_column:
        values_column[0] = 0
    if 1 not in values_column:
        values_column[1] = 0
    if 2 not in values_column:
        values_column[2] = 0

    return values_column

her2dic = plot_ptype(her2df)
hrdic = plot_ptype(hrdf)
tndic = plot_ptype(tndf)

def create_plot(dataframe, figure_name):
    # change the position of chr1 to be in the beginning and make keys in a list
    keylist = list(dataframe.keys())
    keylist.remove('chr1')
    keylist.insert(0,'chr1')

    # number of chromosomes
    N = 23
    # -1 values for the dataframe
    minusone = (dataframe['chr1'][-1], dataframe['chr2'][-1], dataframe['chr3'][-1],
                dataframe['chr4'][-1], dataframe['chr5'][-1], dataframe['chr6'][-1],
                dataframe['chr7'][-1], dataframe['chr8'][-1], dataframe['chr9'][-1],
                dataframe['chr10'][-1], dataframe['chr11'][-1], dataframe['chr12'][-1],
                dataframe['chr13'][-1], dataframe['chr14'][-1], dataframe['chr15'][-1],
                dataframe['chr16'][-1], dataframe['chr17'][-1], dataframe['chr18'][-1],
                dataframe['chr19'][-1], dataframe['chr20'][-1], dataframe['chr21'][-1],
                dataframe['chr22'][-1], dataframe['chr23'][-1])
    # 0 values for the dataframe
    zero = (dataframe['chr1'][0], dataframe['chr2'][0], dataframe['chr3'][0],
            dataframe['chr4'][0], dataframe['chr5'][0], dataframe['chr6'][0],
            dataframe['chr7'][0], dataframe['chr8'][0], dataframe['chr9'][0],
            dataframe['chr10'][0], dataframe['chr11'][0], dataframe['chr12'][0],
            dataframe['chr13'][0], dataframe['chr14'][0], dataframe['chr15'][0],
            dataframe['chr16'][0], dataframe['chr17'][0], dataframe['chr18'][0],
            dataframe['chr19'][0], dataframe['chr20'][0], dataframe['chr21'][0],
            dataframe['chr22'][0], dataframe['chr23'][0])
    # 1 values for the dataframe
    one = (dataframe['chr1'][1], dataframe['chr2'][1], dataframe['chr3'][1],
            dataframe['chr4'][1], dataframe['chr5'][1], dataframe['chr6'][1],
            dataframe['chr7'][1], dataframe['chr8'][1], dataframe['chr9'][1],
            dataframe['chr10'][1], dataframe['chr11'][1], dataframe['chr12'][1],
            dataframe['chr13'][1], dataframe['chr14'][1], dataframe['chr15'][1],
            dataframe['chr16'][1], dataframe['chr17'][1], dataframe['chr18'][1],
            dataframe['chr19'][1], dataframe['chr20'][1], dataframe['chr21'][1],
            dataframe['chr22'][1], dataframe['chr23'][1])
    # 2 values for the dataframe
    two = (dataframe['chr1'][2], dataframe['chr2'][2], dataframe['chr3'][2],
            dataframe['chr4'][2], dataframe['chr5'][2], dataframe['chr6'][2],
            dataframe['chr7'][2], dataframe['chr8'][2], dataframe['chr9'][2],
            dataframe['chr10'][2], dataframe['chr11'][2], dataframe['chr12'][2],
            dataframe['chr13'][2], dataframe['chr14'][2], dataframe['chr15'][2],
            dataframe['chr16'][2], dataframe['chr17'][2], dataframe['chr18'][2],
            dataframe['chr19'][2], dataframe['chr20'][2], dataframe['chr21'][2],
            dataframe['chr22'][2], dataframe['chr23'][2])
    ind = np.arange(N)
    width = 0.2

    plt.figure(figsize=(14, 14), dpi=400)

    # here I removed the zeros-normal because I think they do not add much
    twos = plt.bar(ind, two, width, color='b')
    ones = plt.bar(ind+ width, one, width, color='g')
    #zeros = plt.bar(ind+ width+ width, zero, width)
    minusones = plt.bar(ind+ width+ width, minusone, width, color='r')
    plt.ylabel('CGH-array counts', fontsize=18)
    # name the title
    name_title = 'CGH array values for ' + str(figure_name)
    plt.title(name_title, fontsize=20)
    plt.xticks(ind, ('chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
                    'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
                    'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22',
                    'chr23'), fontsize=12)

    # max value that sets the borders
    max_border = max(max(minusone), max(one), max(two))
    plt.yticks(np.arange(0, max_border+300, 500), fontsize=16)
    plt.legend((twos[0], ones[0], minusones[0]), ('Amplification', 'Gain', 'Loss'), fontsize=18)

    fig_plus_png = str(figure_name) + '.png'
    plt.savefig(fig_plus_png, format='png')

    return plt.show()

create_plot(hrdic, 'HR+')
create_plot(her2dic,'HER2+')
create_plot(tndic,'Triple Negative')
