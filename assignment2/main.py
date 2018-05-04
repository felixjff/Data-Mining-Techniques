# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 20:05:00 2018

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
dataset = pd.read_csv('Data/training_set_VU_DM_2014.csv')

''' Missing Value Analysis '''
## Start: Visitor History ##
dataset = dataset.assign(starrating_diff = pd.Series(np.zeros(len(dataset))))
dataset = dataset.assign(usd_diff = pd.Series(np.zeros(len(dataset))))
dataset['starrating_diff'] = abs(dataset['visitor_hist_starrating'] - dataset['prop_starrating'])
dataset['usd_diff'] = abs(np.log10(dataset['visitor_hist_adr_usd']) - np.log10(dataset['price_usd']))

pct_usd = pd.DataFrame(np.zeros((11,2))) #10 categories plus NAs
pct_usd.columns = ['Booking', 'Click']
pct_usd.index.names = ['Category']
pct_usd['Category'] = pct_usd.index
pct_star = pd.DataFrame(np.zeros((6,2)))
pct_star.columns = ['Booking', 'Click']
pct_star.index.names = ['Category']
pct_star['Category'] = pct_star.index
for i in range(10):
    dataset.loc[np.logical_and( i/10 < dataset['usd_diff'], dataset['usd_diff'] <= (i+1)/10 ), 'usd_diff'] = i+1
    pct_usd.iloc[i] = dict(Booking = dataset.loc[dataset['usd_diff'] == i, 'booking_bool'].sum()/len(dataset.loc[dataset['usd_diff'] == i]),
                Click = dataset.loc[dataset['usd_diff'] == i, 'click_bool'].sum()/len(dataset.loc[dataset['usd_diff'] == i]))
pct_usd.iloc[10] = dict(Booking = dataset.loc[dataset['usd_diff'].isnull(), 'booking_bool'].sum()/len(dataset['usd_diff'].isnull()),
                Click = dataset.loc[dataset['usd_diff'].isnull(), 'click_bool'].sum()/len(dataset['usd_diff'].isnull()))

for i in range(5):
    dataset.loc[np.logical_and( i < dataset['starrating_diff'], dataset['starrating_diff'] <= (i+1) ), 'starrating_diff'] = i+1
    pct_star.iloc[i] = dict(Booking = dataset.loc[dataset['starrating_diff'] == i, 'booking_bool'].sum()/len(dataset.loc[dataset['starrating_diff'] == i]),
                 Click = dataset.loc[dataset['starrating_diff'] == i, 'click_bool'].sum()/len(dataset.loc[dataset['starrating_diff'] == i]))
pct_star.iloc[5] = dict(Booking = dataset.loc[dataset['starrating_diff'].isnull(), 'booking_bool'].sum()/len(dataset['starrating_diff'].isnull()),
                 Click = dataset.loc[dataset['starrating_diff'].isnull(), 'click_bool'].sum()/len(dataset['starrating_diff'].isnull()))

pct_star[['Booking', 'Click']] = pct_star[['Booking', 'Click']]*100
pct_usd[['Booking', 'Click']] = pct_usd[['Booking', 'Click']]*100
pct_star.columns = ['% of Hotels being Booked', 'Pct of Hotels being Click', 'Category']
pct_usd.columns = ['% of Hotels being Booked', 'Pct of Hotels being Click', 'Category']
sns.set(style="whitegrid", color_codes=True) #It can be seen history is important!!
sns.barplot(x = 'Category', y = '% of Hotels being Booked', data=pct_usd).set_title('Matching and Mismatching of log10(price)')
sns.barplot(x = 'Category', y = '% of Hotels being Click', data=pct_usd).set_title('Matching and Mismatching of log10(price)')
sns.barplot(x = 'Category', y = '% of Hotels being Click', data=pct_star).set_title('Matching and Mismatching of Starrating')
sns.barplot(x = 'Category', y = '% of Hotels being Booked', data=pct_star).set_title('Matching and Mismatching of lStarrating')

#Categories representing smaller difference have higher booking %s. Hence, create low/high difference categorical variable
#to incorporate a user's history into the analysis. 
usd_diff_high = [0, 1, 2] #Observed from plot(s)
usd_diff_low = [7, 8, 9]
star_diff_high = [0, 1] #Observed from plot(s)
star_diff_low = [3, 4]

dataset = dataset.assign(dummy_starrating_diff_high = pd.Series(np.zeros(len(dataset))))
dataset = dataset.assign(dummy_starrating_diff_low = pd.Series(np.zeros(len(dataset))))
dataset = dataset.assign(dummy_usd_diff_high = pd.Series(np.zeros(len(dataset))))
dataset = dataset.assign(dummy_usd_diff_low = pd.Series(np.zeros(len(dataset))))

for i in usd_diff_low:
    dataset.loc[dataset['usd_diff'] == i, 'dummy_usd_diff_low'] = 1 
for i in usd_diff_high:
    dataset.loc[dataset['usd_diff'] == i, 'dummy_usd_diff_high'] = 1 
for i in star_diff_low:
    dataset.loc[dataset['starrating_diff'] == i, 'dummy_starrating_diff_low'] = 1 
for i in star_diff_high:
    dataset.loc[dataset['starrating_diff'] == i, 'dummy_starrating_diff_high'] = 1
    
dataset = dataset.drop(['usd_diff', 'starrating_diff', 'visitor_hist_adr_usd', 'visitor_hist_starrating'])    
## Visitor History ##

## Hotel Descriptions ##

## Hotel Descriptions ##