# -*- coding: utf-8 -*-
"""
Created on Thu May 17 21:31:01 2018

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


''' MODEL CALIBRATION '''

#Open training set
with open('training_set_VU_DM_2014.csv', 'r') as csvfile:
    train = pd.read_csv(csvfile)

#Determine if a column has missing values (mistakes in previous analysis)
nans = list([])
for i in train.columns:
    x = train[i].isnull().values.any()
    if x != False:
        nans.append(i)

train = train.drop(nans, axis = 1) #remove columns with NaNs

#Define all variables that are not used for training each model


'Recurrent Neural Network'

from MachineLearning import NeuralNetwork
from MachineLearning import NeuronLayer

#Different variable selections
m1_rnn = ['hotel_quality_click_srch_length_of_staystd',
       'hotel_quality_booking_srch_length_of_staystd', 'hotel_quality_click_prop_country_idstd',
       'hotel_quality_click_srch_destination_idstd', 
       'hotel_quality_booking_srch_destination_idstd','hotel_quality_booking_prop_country_idstd',
       'hotel_quality_click_visitor_location_country_idstd',
       'hotel_quality_booking_visitor_location_country_idstd','hotel_quality_click_monthstd',
       'hotel_quality_booking_monthstd']  #Diversification only on key

m2_rnn= ['prop_brand_bool', 'promotion_flag', 'price_rank', 'star_rank', 'price_difference_rank',
         'prop_starrating_monotonic', 'dummy_starrating_diff_high', 'dummy_starrating_diff_low', 
         'dummy_usd_diff_high', 'dummy_usd_diff_low','hotel_quality_click_monthstd',
         'hotel_quality_booking_monthstd'] #Diversification only on explanatory variables

m3_rnn= ['prop_brand_bool', 'promotion_flag', 'price_rank', 'star_rank', 
         'prop_starrating_monotonic',  'dummy_starrating_diff_low', 
         'dummy_usd_diff_high', 'dummy_usd_diff_low',
         'hotel_quality_click_monthstd',
         'hotel_quality_booking_monthstd',
         'prop_location_score1_monthstd',
         'prop_location_score2_monthstd',
         'hotel_position_avg_monthstd'
         ] #Diversification only on explanatory variables

np.random.seed(10)

#Define and train neural network
neurons1 = 8
neurons2 = 1

layer1 = NeuronLayer(neurons1, len(train[m3_rnn].columns))
layer2 = NeuronLayer(neurons2, neurons1)

activation_function = 'Sigmoid'

neural_network = NeuralNetwork(layer1, layer2, activation_function)
neural_network.train(train[m3_rnn], np.matrix(train['booking_bool']).T, 30000)
hidden_state, output = neural_network.think(train[m3_rnn])

#Insample Fit 
result = train[['srch_id','booking_bool', 'click_bool']]
result = result.assign(pred = output)
in_nn_ndcg = neural_network.ndcg(result)

#Test Performance 




