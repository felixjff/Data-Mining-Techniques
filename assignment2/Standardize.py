# -*- coding: utf-8 -*-
"""
Created on Mon May 14 21:17:51 2018

@author: Felix Jose Farias Fueyo
"""

#Loading libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import math
import datetime

plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm

#loading data
testing = 0
if testing == 0:
    train = pd.read_csv('data/training_set_VU_DM_2014.csv')
else:
    train = pd.read_csv('Data/MissingValueAnalysis/test_set_VU_DM_2014.csv')

##Decrease Sample Size##
np.random.seed(10)

train_srch_id = np.random.choice(a = train.srch_id.unique(), size = round(len(train.srch_id.unique())*0.35), replace = False)
train = train[pd.Series(train.srch_id).isin(train_srch_id)]
#del train_srch_id

train = train.drop(['comp2_rate', 'comp2_inv','comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv',
                    'comp3_rate', 'comp3_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv',
                    'comp4_rate', 'comp4_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv',
                    'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv',
                    'comp8_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff',
                    'comp2_rate_percent_diff', 'comp1_inv' 'comp1_rate_percent_diff'], axis = 1)

#Parse date_time to obtain months
train['month'] =  [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').month for date in train.date_time.values]

keys = ['month', 'visitor_location_country_id', 'srch_destination_id', 
        'srch_length_of_stay', 'prop_country_id']
#Not included as keys due to memory and speed issues: 'prop_id','srch_id','srch_room_count'

std_vars = ['prop_starrating', 'prop_review_score', 'prop_location_score1', 'prop_location_score2', 'price_usd', 
            'srch_query_affinity_score', 'orig_destination_distance', 'price_difference',
            'hotel_position_avg', 'hotel_quality_click', 'hotel_quality_booking']

#Standardize with respect to each key variable
from sklearn.preprocessing import scale

for k in keys:
    for i in std_vars:
        n = i + '_' + k + "std"
        train[n] = math.nan
        it = 0 
        for kv in train[k].unique():
            train.loc[train[k] == kv, n] = scale(train.loc[train[k] == kv, i])
            it = it + 1
            print(it/len(train[k].unique()))

train = train.drop(['month', 'visitor_location_country_id', 'prop_country_id', 'srch_length_of_stay', 'srch_destination_id'], axis = 1)
train = train.drop(std_vars, axis = 1)
            
std_vars2 = ['price_rank', 'star_rank', 'prop_starrating_monotonic', 'prop_log_historical_price', 
             'orig_destination_distance', 'price_difference_rank', 'srch_booking_window',
             ]

for i in std_vars2:
    train[i] = scale(train[i])
        
if testing == 0:
    train.to_csv('training_set_VU_DM_2014.csv')
else:
    train.to_csv('test_set_VU_DM_2014.csv')

