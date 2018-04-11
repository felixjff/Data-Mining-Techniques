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
dataset['time'] = pd.to_datetime(dataset['time'], format = '%Y%m%d', unit ='d', errors = 'coerce')
variables = dataset['variable'].unique()

#Missing values? 
dataset['variable'].isnull().any()

#Aggregation to daily average 
dataset = pd.DataFrame(dataset)
dAgg = pd.DataFrame() #creates a new dataframe that's empty
   
    