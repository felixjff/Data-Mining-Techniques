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
dataset = pd.read_csv('data/feature_extraction/training_set_VU_DM_2014.csv')

#Parse date_time to obtain months
dataset['month'] =  [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').month for date in dataset.date_time.values]

keys = ['month', 'srch_id', 'visitor_location_country_id', 'prop_country_id', 'srch_destination_id', 
        'srch_length_of_stay', 'prop_id', 'srch_room_count', 'srch_adult_count', 'srch_children_count']

norm_vars = ['prop_starrating_monotinic', 'prop_starrating', 'prop_review_score', 'prop_location_score1', 'prop_location_score2', 
             'prop_log_historical_price', 'price_usd', 'srch_query_affinity_score', 'orig_desination_distance',
             'price_difference_rank', 'hotel_position_avg', 'hotel_quality_click', 'hotel_quality_booking',
             'price_difference']

#Standardize with respect to each key variable
from sklearn.preprocessing import scale

for k in keys:
    for i in norm_vars:
        n = i + '_' + k + "std"
        dataset[n] = math.nan
        for kv in dataset[k].unique():
            dataset.loc[dataset[k] == kv, n] = scale(dataset.loc[dataset[k] == kv, i])
        

