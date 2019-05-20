#Import packages
import pandas as pd
import numpy as np
from numpy import mean
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import seaborn as sn
import data_loader, assignment_slp_v5
import warnings, os

'''In this script Test scores and RFECV scores are plottef
for each iteration '''

#Load data with the results
optimisation_results = data_loader.load_optimisation_results()

#Create empty lists
test_scores=[]
refcv_training_scores=[]
itr=[]
#Add to lists the appropriate elements of the optimisation file
for result in optimisation_results:
    
    test_scores.append(result['test_score'])
    refcv_training_scores.append(mean(result['rfecv_training_scores']))
    itr.append(result['iteration'])

#Plotting
plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Scores")
plt.plot(itr, test_scores,'r',itr,refcv_training_scores,'b')
plt.title('Test and Rfecv Scores')
plt.gca().legend(('Test Scores','Rfecv Scores'))
plt.show()
