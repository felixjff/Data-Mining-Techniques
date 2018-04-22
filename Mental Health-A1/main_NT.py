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

#Loading data
def loadData(datapath):
    dataset = pd.read_csv(datapath)
    dataset['time'] = pd.to_datetime(dataset['time'])
    return dataset

#Aggregation to daily average 
dataset = pd.DataFrame(loadData('Data/dataset_mood_smartphone.csv'))
dAgg = dataset.groupby(['id', 'variable', dataset['time'].dt.date])['value'].mean() #hierarchical grouping: id > time > variable, averaging the values of the variable. Seems to lose the exact date information

#Create dummy frame to fill with values of moving average
dataset2 = pd.DataFrame(loadData('Data/dataset_mood_smartphone.csv'))
new_df = dataset2.groupby(['id', 'variable', dataset['time'].dt.date])['value'].mean()

idx = pd.IndexSlice
window_size = 5

#create new user array to be filled with variable scores for <window_size>-days and the target score
#usr_list = dAgg.axes[0].levels[0].tolist()
#var_list = dAgg.axes[0].levels[1].tolist()
#tim_list = dAgg.axes[0].levels[2].tolist()
#all_list = [usr_list, var_list, tim_list]
#mi = pd.MultiIndex.from_product(all_list,names=['id', 'variable', 'time'])
#series = mi.to_series()
#new_series = series.groupby(['id', 'variable', 'time'])
#new_df = new_series
#new_df = mi.to_frame()
#new_df.set_index(['id', 'variable', 'time'])


#new_usr_array = np.zeros(dAgg.axes[0].levels[0].size * dAgg.axes[0].levels[1].size * (len(dAgg.axes[0].levels[2]) - window_size) ) #create new_usr_array, with #new_usr = #usr*variables*(days-window_size)
new_usr_array = np.empty
for usr in dAgg.axes[0].levels[0]: # loop over all users to generate time window data
    print('usr == == ==', usr)
    
    for d in range(len(dAgg.axes[0].levels[2]) - window_size):
        if d == 0:
            print('window: ', dAgg.axes[0].levels[2][d], dAgg.axes[0].levels[2][d+window_size-1])
            print('slice: ', dAgg.loc[idx[usr,:,slice(dAgg.axes[0].levels[2][d],dAgg.axes[0].levels[2][d+window_size-1])]].mean(level='variable'))
        
        insert = dAgg.loc[idx[usr,:,slice(dAgg.axes[0].levels[2][d],dAgg.axes[0].levels[2][d+window_size-1])]].mean(level='variable')
        #print(new_df[usr])
        #print(new_df.loc[usr])
        new_df.loc[idx[usr,:,dAgg.axes[0].levels[2][d]]].value = insert
                   #,:,dAgg.axes[0].levels[1],dAgg.axes[0].levels[2][d]]
moving = dAgg.rolling(window_size).mean()

#MultiIndex(levels=[['AS14.01', 'AS14.02', 'AS14.03', 'AS14.05', 'AS14.06', 'AS14.07', 'AS14.08', 'AS14.09', 'AS14.12', 'AS14.13', 'AS14.14', 'AS14.15', 'AS14.16', 'AS14.17', 'AS14.19', 'AS14.20', 'AS14.23', 'AS14.24', 'AS14.25', 'AS14.26', 'AS14.27', 'AS14.28', 'AS14.29', 'AS14.30', 'AS14.31', 'AS14.32', 'AS14.33'], 
#['activity', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather', 'call', 'circumplex.arousal', 'circumplex.valence', 'mood', 'screen', 'sms'],
#[2014-02-17, 2014-02-18,..]],
#labels=[[],[]],
#names=['id','variable','time']
