# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 22:11:05 2018

@author: Felix Jose Farias Fueyo
"""
#Loading libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import datetime

plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm

#loading data
dataset = pd.read_csv("Data/dataset_mood_smartphone.csv")
#dataset['time'] = pd.to_datetime(dataset['time'], format = '%Y%m%D', errors = 'coerce') #convert dates to datetime object (or a series), seems to miscreate 'datetime'
dataset['time'] = pd.to_datetime(dataset['time'])
variables = dataset['variable'].unique()

#Missing values? 
dataset['variable'].isnull().any()

#Aggregation to daily average 
dataset = pd.DataFrame(dataset)
dAgg = dataset.groupby(['id', 'variable', dataset['time'].dt.date])['value'].mean() #hierarchical grouping: id > time > variable, averaging the values of the variable. Seems to lose the exact date information
print(dAgg.head(3000)) #to check grouping and values

#Distribution: target variable 
sns.distplot(dAgg[:,'mood',:]) #Left-skewed distribution.

#Define training set to 2/3 of the complete set
temp = round(dAgg[:,'mood',:].index.get_level_values('time').unique().size/3) #All dates
train_dates = dAgg[:,'mood',:].index.get_level_values('time').unique().values
train_dates.sort()
train_dates = train_dates[1:2*temp] #Will be used to loop over dates when training..
del temp
train = dAgg.loc[train_dates.max()>=dAgg.index.get_level_values('time')] #training set

#Separate variables into categorical/binary/numeric types
bin_data = ['sms', "call"]
num_data = train.index.get_level_values('variable').unique().values
num_data = num_data[np.logical_and('sms' != num_data,'call' != num_data)]

#Numeric Data Analysis
#Correlations: Across variables 
train = train.reorder_levels(['variable','id','time']) #Variables placed at highest indexing level
#Define correlation matrix
corr = np.full([num_data.size, num_data.size], np.nan)
#loop through variables to obtain correlations
for i in range(0,num_data.size-1):
    for j in range(0,num_data.size-1):
        corr[i,j] = train[num_data[i],:,:].corr(train[num_data[j],:,:])
#Heatmap of correlations  
corr = pd.DataFrame(corr)
corr.columns = num_data
corr.index = num_data
sns.heatmap(corr)
