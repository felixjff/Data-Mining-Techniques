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
datasetFrame = pd.DataFrame(dataset)
datasetFrameGroup = datasetFrame.groupby(['id', datasetFrame['time'].dt.day, 'variable'])['value'].mean() #hierarchical grouping: id > time > variable, averaging the values of the variable. Seems to lose the exact date information
print(datasetFrameGroup.head(3000)) #to check grouping and values
