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
testing = 1
if testing == 0:
    train = pd.read_csv('data/MissingValueAnalysis/training_set_VU_DM_2014.csv')
else:
    train = pd.read_csv('data/MissingValueAnalysis/test_set_VU_DM_2014.csv')

train = train.drop(['comp2_rate', 'comp2_inv','comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv',
                    'comp3_rate', 'comp3_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv',
                    'comp4_rate', 'comp4_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv',
                    'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv',
                    'comp8_rate_percent_diff'], axis = 1)

#np.random.seed(10)

#train_srch_id = np.random.choice(a = train.srch_id.unique(), size = round(len(train.srch_id.unique())*0.70), replace = False)
#train = train[pd.Series(train.srch_id).isin(train_srch_id)]
#del sample
#del train_srch_id


#Parse date_time to obtain months
train['month'] =  [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').month for date in train.date_time.values]

keys = ['month', 'srch_id', 'visitor_location_country_id', 'srch_destination_id',
        'srch_length_of_stay', 'prop_id']
#Not included as keys due to memory issues: 'prop_country_id','srch_room_count'

std_vars = ['prop_starrating', 'prop_review_score', 'prop_location_score1', 'price_usd',
            'srch_query_affinity_score', 'orig_destination_distance', 'price_difference']

#Standardize with respect to each key variable
from sklearn.preprocessing import scale

for k in keys:
    for i in std_vars:
        n = i + '_' + k + "std"
        train[n] = math.nan
        for kv in train[k].unique():
            train.loc[train[k] == kv, n] = scale(train.loc[train[k] == kv, i])
    train = train.drop(k)

std_vars2 = ['prop_starrating_monotonic', 'prop_location_score2',
             'prop_log_historical_price', 'orig_destination_distance', 'price_difference_rank',
             'hotel_position_avg', 'hotel_quality_click', 'hotel_quality_booking']

for i in std_vars2:
    train[i] = scale(train[i])

if testing == 0:
    train.to_csv('data/standardise/training_set_VU_DM_2014.csv')
else:
    train.to_csv('data/standardise/test_set_VU_DM_2014.csv')
