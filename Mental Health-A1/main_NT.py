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
dataset = pd.read_csv('Data/dataset_mood_smartphone.csv')
#dataset['time'] = pd.to_datetime(dataset['time'], format = '%Y%m%D', errors = 'coerce') #convert dates to datetime object (or a series), seems to miscreate 'datetime'
dataset['time'] = pd.to_datetime(dataset['time'])
variables = dataset['variable'].unique()

#Aggregation to daily average 
dataset = pd.DataFrame(dataset)
dAgg = dataset.groupby(['id', 'variable', dataset['time'].dt.date])['value'].mean() #hierarchical grouping: id > time > variable, averaging the values of the variable. Seems to lose the exact date information


#create numpy array and fill it with the level names
#TODO: fix proper copy? Length of copy?
id_lvl = np.tile(dAgg.axes[0].levels[0],(1, dAgg.axes[0].levels[1].size * dAgg.axes[0].levels[2].size))
var_lvl = np.tile(dAgg.axes[0].levels[1],(1,id_lvl.size))
time_lvl = np.tile(dAgg.axes[0].levels[2],(1,id_lvl.size))
arrays = [id_lvl,var_lvl,time_lvl]
###mi = pd.MultiIndex.from_arrays(arrays, names=('id', 'variable', 'time')) ## Need to be same length? Better Tiling?

pd.MultiIndex.copy

#create array to be filled with variable scores for <window_size>-days and the target score
nt_array = np.zeros((dAgg.axes[0].levels[0].size * dAgg.axes[0].levels[1].size * dAgg.axes[0].levels[2].size, 2))
print(nt_array.shape)
idx = pd.IndexSlice
window_size = 5

new_usr_array = np.zeros(dAgg.axes[0].levels[0].size * (len(dAgg.axes[0].levels[2]) - window_size) ) #create new_usr_array, with #new_usr = #usr*(days-window_size)
for usr in dAgg.axes[0].levels[0]: # loop over all users to generate time window data
	print('usr == == ==', usr)

	for d in range(len(dAgg.axes[0].levels[2]) - window_size):
		x = dAgg.loc[idx[usr,:,slice(dAgg.axes[0].levels[2][d],dAgg.axes[0].levels[2][d+window_size-1])]].mean(level='variable')
		

#MultiIndex(levels=[['AS14.01', 'AS14.02', 'AS14.03', 'AS14.05', 'AS14.06', 'AS14.07', 'AS14.08', 'AS14.09', 'AS14.12', 'AS14.13', 'AS14.14', 'AS14.15', 'AS14.16', 'AS14.17', 'AS14.19', 'AS14.20', 'AS14.23', 'AS14.24', 'AS14.25', 'AS14.26', 'AS14.27', 'AS14.28', 'AS14.29', 'AS14.30', 'AS14.31', 'AS14.32', 'AS14.33'], 
#['activity', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather', 'call', 'circumplex.arousal', 'circumplex.valence', 'mood', 'screen', 'sms'],
#[2014-02-17, 2014-02-18,..]],
#labels=[[],[]],
#names=['id','variable','time']
