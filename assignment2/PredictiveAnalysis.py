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
import sys
import pickle

plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import scale

def ndcg(result):
    all_ndcg = pd.DataFrame(result['srch_id'].unique())
    all_ndcg.columns = ['srch_id']
    all_ndcg = all_ndcg.set_index('srch_id')
    result = result.set_index('srch_id')
    all_ndcg['ndcg'] = math.nan
    all_ndcg['dcg'] = math.nan
    result['rel'] = result['booking_bool']*5 + result['click_bool'] #should have 5 as maximum
    result.loc[result['rel'] == 6, 'rel'] = 5 # 6 can only occure if click and bool are 1. Hence, cap at 5.
    result = result.drop(['booking_bool', 'click_bool'], axis = 1)
    it = 0
    for i in result.index.unique().values:
        rel_temp = result.loc[i]  #Extract the search query for which the accuracy will be evaluated
        rel_temp = rel_temp.sort_values('pred', ascending = False) #Sort the query based on the estimated probabilities
        irel = rel_temp.sort_values('rel', ascending = False) #compute ideal scenario (highest DCG score)
        IDCG = 0
        DCG = 0 #Compute discounted cumulative gain
        for j in range(1,len(rel_temp)+1):
            DCG = DCG + (2**rel_temp.iloc[j-1].rel-1)/np.log2(j + 1)
            IDCG = IDCG + (2**irel.iloc[j-1].rel-1)/np.log2(j + 1)
        all_ndcg.loc[i, 'ndcg'] = DCG/IDCG  #Compute normalized discounted cumulative gain. (within [0,1] range)
        all_ndcg.loc[i, 'dcg'] = DCG
        it = it + 1
        tot = it/len(result.index.unique())
        sys.stdout.write('{} {}\r'.format(tot, DCG/IDCG))
        sys.stdout.flush()
    return np.mean(all_ndcg)

def prediction_to_output_file(dataframe, outfile_name):
    dataframe = dataframe.sort_values(['srch_id', 'pred'], ascending=[True, False])
    output = dataframe[['srch_id', 'prop_id']]
    output.columns = ['SearchId', 'PropertyId']
    output.to_csv('predictions/{}.csv'.format(outfile_name), index=False)
    return None

''' MODEL CALIBRATION '''

with open('data/standardise/training_set_VU_DM_2014.csv', 'r') as csvfile:
    train = pd.read_csv(csvfile)

with open('data/standardise/test_set_VU_DM_2014.csv', 'r') as csvfile:
    test = pd.read_csv(csvfile)

#Determine if a column has missing values (mistakes in previous analysis)
nans_train = list([])
for i in train.columns:
    x = train[i].isnull().values.any()
    if x != False:
        nans_train.append(i)

train = train.drop(nans_train, axis = 1) #remove columns with NaNs
nans_test = list([])
for i in test.columns:
    x = test[i].isnull().values.any()
    if x != False:
        nans_test.append(i)

test = test.drop(nans_test, axis = 1)

#Scale price_rank and star_rank as they where mistakenly not scaled
train['price_rank'] = scale(train['price_rank'])
train['star_rank'] = scale(train['star_rank'])
test['price_rank'] = scale(test['price_rank'])
test['star_rank'] = scale(test['star_rank'])

#Add some new features to do with position based on train data to test data
prop_id_position_mean = train.groupby(['prop_id']).apply(lambda x: np.mean(x.position))
prop_id_position_q1 = train.groupby(['prop_id']).apply(lambda x: np.percentile(x.position, 25) )
prop_id_position_q3 = train.groupby(['prop_id']).apply(lambda x: np.percentile(x.position, 75) )
#change the index so that new series values can be appended correctly
train = train.set_index('prop_id')
train['prop_id_position_mean'] = prop_id_position_mean
train['prop_id_position_q1'] = prop_id_position_q1
train['prop_id_position_q3'] = prop_id_position_q3
train['prop_id_position_mean_std'] = scale(train['prop_id_position_mean'])
train['prop_id_position_q1_std'] = scale(train['prop_id_position_q1'])
train['prop_id_position_q3_std'] = scale(train['prop_id_position_q3'])

# srch_id_prop_id_position_mean_std = train.groupby(['srch_id']).apply(lambda x: scale(x.prop_id_position_mean))
# srch_id_prop_id_position_q1_std = train.groupby(['srch_id']).apply(lambda x: scale(x.prop_id_position_q1))
# srch_id_prop_id_position_q3_std = train.groupby(['srch_id']).apply(lambda x: scale(x.prop_id_position_q3))
# train = train.set_index('srch_id')
# train['srch_id_prop_id_position_mean_std'] = srch_id_prop_id_position_mean_std
# train['srch_id_prop_id_position_q1_std'] = srch_id_prop_id_position_q1_std
# train['srch_id_prop_id_position_q3_std'] = srch_id_prop_id_position_q3_std

#there are more prop_ids in test than in train, set those equal to 0
test = test.set_index('prop_id')
index_diff = test.index.difference(train.index)
test['prop_id_position_mean'] = 1
test['prop_id_position_q1'] = 1
test['prop_id_position_q3'] = 1
test['prop_id_position_mean'] = prop_id_position_mean
test['prop_id_position_q1'] = prop_id_position_q1
test['prop_id_position_q3'] = prop_id_position_q3
test.prop_id_position_mean = test.prop_id_position_mean.fillna(0)
test.prop_id_position_q1 = test.prop_id_position_q1.fillna(0)
test.prop_id_position_q3 = test.prop_id_position_q3.fillna(0)
test['prop_id_position_mean_std'] = scale(test['prop_id_position_mean'])
test['prop_id_position_q1_std'] = scale(test['prop_id_position_q1'])
test['prop_id_position_q3_std'] = scale(test['prop_id_position_q3'])


#reset index, drop rows which are not standardised
train = train.reset_index()
test = test.reset_index()
train = train.drop(['prop_id_position_mean', 'prop_id_position_q1', 'prop_id_position_q3'], axis=1)
test = test.drop(['prop_id_position_mean', 'prop_id_position_q1', 'prop_id_position_q3'], axis=1)
#########################################################
#RUN AGAIN ON THE TRUE TEST SET (I.E. NOT VALIDATION SET)
#########################################################

#########################################################
#INTRODUCING HOTEL POSITION DUMMIES
#########################################################

hotel_position_series = pd.DataFrame(train.groupby(['prop_id']).apply(lambda x: x.position.mean()))
hotel_position_series.columns = ['position']

hotel_position_series.loc[~np.logical_and(hotel_position_series['position'] > 20, hotel_position_series['position'] <= 50), 'dummy_top2050_avg'] = 0
hotel_position_series.loc[np.logical_and(hotel_position_series['position'] > 20, hotel_position_series['position'] <= 50), 'dummy_top2050_avg'] = 1
hotel_position_series.loc[~np.logical_and(hotel_position_series['position'] > 15, hotel_position_series['position'] <= 20), 'dummy_top1520_avg'] = 0
hotel_position_series.loc[np.logical_and(hotel_position_series['position'] > 15, hotel_position_series['position'] <= 20), 'dummy_top1520_avg'] = 1
hotel_position_series.loc[~np.logical_and(hotel_position_series['position'] > 10, hotel_position_series['position'] <= 15), 'dummy_top1015_avg'] = 0
hotel_position_series.loc[np.logical_and(hotel_position_series['position'] > 10, hotel_position_series['position'] <= 15), 'dummy_top1015_avg'] = 1
hotel_position_series.loc[~np.logical_and(hotel_position_series['position'] <= 10, hotel_position_series['position'] > 5), 'dummy_top510_avg'] = 0
hotel_position_series.loc[np.logical_and(hotel_position_series['position'] <= 10, hotel_position_series['position'] > 5), 'dummy_top510_avg'] = 1
hotel_position_series.loc[~(hotel_position_series['position'] <= 5), 'dummy_top5_avg'] = 0
hotel_position_series.loc[(hotel_position_series['position'] <= 5), 'dummy_top5_avg'] = 1

train = train.set_index('prop_id')
train['dummy_top5_avg'] = hotel_position_series.dummy_top5_avg
train['dummy_top510_avg'] = hotel_position_series.dummy_top510_avg
train['dummy_top1015_avg'] = hotel_position_series.dummy_top1015_avg
train['dummy_top1520_avg'] = hotel_position_series.dummy_top1520_avg
train['dummy_top2050_avg'] = hotel_position_series.dummy_top2050_avg
train = train.reset_index()
train['hotel_position_avg'] = hotel_position_series.position
train['hotel_position_avg'] = train['hotel_position_avg'].fillna(-1)

test = test.set_index('prop_id')
test['dummy_top5_avg'] = 0
test['dummy_top510_avg'] = 0
test['dummy_top1015_avg'] = 0
test['dummy_top1520_avg'] = 0
test['dummy_top2050_avg'] = 0
test['dummy_top5_avg'] = hotel_position_series.dummy_top5_avg
test['dummy_top510_avg'] = hotel_position_series.dummy_top510_avg
test['dummy_top1015_avg'] = hotel_position_series.dummy_top1015_avg
test['dummy_top1520_avg'] = hotel_position_series.dummy_top1520_avg
test['dummy_top2050_avg'] = hotel_position_series.dummy_top2050_avg
test['dummy_top5_avg'] = test['dummy_top5_avg'].fillna(0)
test['dummy_top510_avg'] = test['dummy_top510_avg'].fillna(0)
test['dummy_top1015_avg'] = test['dummy_top1015_avg'].fillna(0)
test['dummy_top1520_avg'] = test['dummy_top1520_avg'].fillna(0)
test['dummy_top2050_avg'] = test['dummy_top2050_avg'].fillna(0)
test = test.reset_index()

#########################################################
#HOTEL QUALITY VARIABLES TO ADD TO TEST SET
#########################################################
# may be able to test hashed ones with other vars
# hotel_quality_add = ['hotel_quality_booking', \
# 'hotel_quality_booking_visitor_location_country_idstd', \
# 'hotel_quality_click_srch_length_of_staystd', \
# 'hotel_quality_booking_srch_destination_idstd', \
# 'hotel_quality_click', \
# 'hotel_quality_click_visitor_location_country_idstd', \
# 'hotel_quality_booking_srch_length_of_staystd', \
# 'hotel_quality_click_srch_destination_idstd']

hotel_quality_booking = train.groupby(['prop_id'])['hotel_quality_booking'].apply(lambda x: x.iloc[0])
hotel_quality_click = train.groupby(['prop_id'])['hotel_quality_click'].apply(lambda x: x.iloc[0])
hotel_quality_click_srch_length_of_staystd = train.groupby(['prop_id'])['hotel_quality_click_srch_length_of_staystd'].apply(lambda x: x.iloc[0])
hotel_quality_booking_srch_destination_idstd = train.groupby(['prop_id'])['hotel_quality_booking_srch_destination_idstd'].apply(lambda x: x.iloc[0])
hotel_quality_booking_srch_length_of_staystd = train.groupby(['prop_id'])['hotel_quality_booking_srch_length_of_staystd'].apply(lambda x: x.iloc[0])
hotel_quality_click_srch_destination_idstd = train.groupby(['prop_id'])['hotel_quality_click_srch_destination_idstd'].apply(lambda x: x.iloc[0])
hotel_position_avg_per_prop_id = train.groupby(['prop_id']).apply(lambda x: np.mean(x.hotel_position_avg))
test = test.set_index('prop_id')
train = train.set_index('prop_id')
# the first min assignments here are to take care of the prop_ids which are in test and not train
test['hotel_quality_booking'] = min(train.hotel_quality_booking)
test['hotel_quality_booking'] = hotel_quality_booking
test['hotel_quality_booking'] = test['hotel_quality_booking'].fillna(min(test['hotel_quality_booking']))

test['hotel_quality_click'] = min(train.hotel_quality_click)
test['hotel_quality_click'] = hotel_quality_click
test['hotel_quality_click'] = test['hotel_quality_click'].fillna(min(test['hotel_quality_click']))

test['hotel_quality_click_srch_length_of_staystd'] = min(train.hotel_quality_click_srch_length_of_staystd)
test['hotel_quality_click_srch_length_of_staystd'] = hotel_quality_click_srch_length_of_staystd
test['hotel_quality_click_srch_length_of_staystd'] = test['hotel_quality_click_srch_length_of_staystd'].fillna(min(test['hotel_quality_click_srch_length_of_staystd']))

test['hotel_quality_booking_srch_destination_idstd'] = min(train.hotel_quality_booking_srch_destination_idstd)
test['hotel_quality_booking_srch_destination_idstd'] = hotel_quality_booking_srch_destination_idstd
test['hotel_quality_booking_srch_destination_idstd'] = test['hotel_quality_booking_srch_destination_idstd'].fillna(min(test['hotel_quality_booking_srch_destination_idstd']))

test['hotel_quality_booking_srch_length_of_staystd'] = min(train.hotel_quality_booking_srch_length_of_staystd)
test['hotel_quality_booking_srch_length_of_staystd'] = hotel_quality_booking_srch_length_of_staystd
test['hotel_quality_booking_srch_length_of_staystd'] = test['hotel_quality_booking_srch_length_of_staystd'].fillna(min(test['hotel_quality_booking_srch_length_of_staystd']))


test['hotel_quality_click_srch_destination_idstd'] = min(train.hotel_quality_click_srch_destination_idstd)
test['hotel_quality_click_srch_destination_idstd'] = hotel_quality_click_srch_destination_idstd
test['hotel_quality_click_srch_destination_idstd'] = test['hotel_quality_click_srch_destination_idstd'].fillna(min(test['hotel_quality_click_srch_destination_idstd']))

train['hotel_position_avg_per_prop_id'] = min(hotel_position_avg_per_prop_id)
train['hotel_position_avg_per_prop_id'] = hotel_position_avg_per_prop_id
test['hotel_position_avg_per_prop_id'] = min(hotel_position_avg_per_prop_id)
test['hotel_position_avg_per_prop_id'] = hotel_position_avg_per_prop_id
train['hotel_position_avg_per_prop_id'] = scale(train['hotel_position_avg_per_prop_id'])
test['hotel_position_avg_per_prop_id'] = test['hotel_position_avg_per_prop_id'].fillna(min(test['hotel_position_avg_per_prop_id']))
test['hotel_position_avg_per_prop_id'] = scale(test['hotel_position_avg_per_prop_id'])
train = train.reset_index()
test = test.reset_index()




#########################################################

#Different variable selections
# train['booking_click'] = train.booking_bool #Define a variable that is one if click or booked
# train.loc[train['booking_click'] == 0, 'booking_click'] = train.loc[train['booking_click'] == 0, 'click_bool']

#TRY GIVING MORE WEIGHT TO A BOOKING OF A HOTEL
train['booking_click'] = train.booking_bool * 5 #Define a variable that is one if click or booked
train.loc[train['booking_click'] == 0, 'booking_click'] = train.loc[train['booking_click'] == 0, 'click_bool']
train['booking_click'] = scale(train['booking_click'])

m1_rnn = ['hotel_quality_click_srch_length_of_staystd',
       'hotel_quality_booking_srch_length_of_staystd', 'hotel_quality_click_prop_country_idstd',
       'hotel_quality_click_srch_destination_idstd',
       'hotel_quality_booking_srch_destination_idstd','hotel_quality_booking_prop_country_idstd',
       'hotel_quality_click_visitor_location_country_idstd',
       'hotel_quality_booking_visitor_location_country_idstd','hotel_quality_click_monthstd',
       'hotel_quality_booking_monthstd']  #Diversification only on key

m2_rnn= ['price_rank', 'star_rank','price_difference_rank',
        # 'hotel_quality_click_monthstd',
       'prop_location_score2_visitor_location_country_idstd',
       'price_usd_visitor_location_country_idstd',
       'orig_destination_distance_visitor_location_country_idstd',
       'price_difference_visitor_location_country_idstd',
       'prop_starrating_srch_destination_idstd',
       'prop_review_score_srch_destination_idstd',
       'prop_location_score1_srch_destination_idstd',
       'prop_location_score2_srch_destination_idstd',
       'price_usd_srch_destination_idstd',
       'orig_destination_distance_srch_destination_idstd',
       'price_difference_srch_destination_idstd',
       'hotel_quality_click_srch_destination_idstd',
       'hotel_quality_booking_srch_destination_idstd',
       'srch_length_of_stay_srch_destination_idstd',
       'prop_location_score2_srch_length_of_staystd',
       'price_usd_srch_length_of_staystd',
       'price_difference_srch_length_of_staystd',
       'dummy_top5_avg', 'dummy_top510_avg', 'dummy_top1015_avg',
       'dummy_top1520_avg', 'dummy_top2050_avg',
       'hotel_quality_click',
       'hotel_quality_booking',
       'hotel_quality_click_srch_destination_idstd',
       'hotel_quality_booking_srch_destination_idstd',
       'hotel_quality_click_srch_length_of_staystd',
       'hotel_quality_booking_srch_length_of_staystd',
       'hotel_position_avg_per_prop_id'] #Diversification only on explanatory variables

m2_rnn_test = ['price_rank', 'star_rank', 'price_difference_rank',
       'prop_location_score2_visitor_location_country_idstd',
       'price_usd_visitor_location_country_idstd',
       'orig_destination_distance_visitor_location_country_idstd',
       'price_difference_visitor_location_country_idstd',
       'prop_starrating_srch_destination_idstd',
       'prop_review_score_srch_destination_idstd',
       'prop_location_score1_srch_destination_idstd',
       'prop_location_score2_srch_destination_idstd',
       'price_usd_srch_destination_idstd',
       'orig_destination_distance_srch_destination_idstd',
       'price_difference_srch_destination_idstd',
       'srch_length_of_stay_srch_destination_idstd',
       'prop_location_score2_srch_length_of_staystd',
       'price_usd_srch_length_of_staystd',
       'price_difference_srch_length_of_staystd']

m3_rnn= ['prop_brand_bool', 'promotion_flag', 'price_rank', 'star_rank', 'price_difference_rank',
         'prop_starrating_monotonic',  'dummy_starrating_diff_low',
         'dummy_usd_diff_high', 'dummy_usd_diff_low',
         'hotel_quality_click_monthstd',
         'hotel_quality_booking_monthstd', 'hotel_position_avg_srch_destination_idstd',
         'prop_location_score1_monthstd', 'price_difference_prop_country_idstd',
         'prop_location_score2_monthstd', 'price_usd_srch_destination_idstd',
         'hotel_position_avg_monthstd','price_usd_srch_destination_idstd'] #Diversification only on explanatory variables

'Recurrent Neural Network'

from MachineLearning import NeuralNetwork
from MachineLearning import NeuronLayer
from sklearn.neural_network import MLPClassifier

np.random.seed(10)

#Define and train neural network
neurons1 = 6
neurons2 = 1

layer1 = NeuronLayer(neurons1, len(train[m2_rnn].columns))
layer2 = NeuronLayer(neurons2, neurons1)

activation_function = 'Sigmoid'

neural_network = NeuralNetwork(layer1, layer2, activation_function)
neural_network.train(train[m2_rnn], np.matrix(train['booking_click']).T, 20)
hidden_state, output = neural_network.think(train[m2_rnn])

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,1), random_state=1, activation = 'logistic')
clf.fit(train[m2_rnn], np.array(train['booking_click']).T)
output_ = clf.predict(train[m2_rnn])

#Insample Fit is very bad. This imply the neural network is not an appropate model to train this time of dataset
result = train[['srch_id','booking_bool', 'click_bool']]
result = result.assign(pred = output_)
in_nn_ndcg = neural_network.ndcg(result)



'RandomForestClassifier'
#####################################################
#RUN THESE FIRST
#####################################################
from sklearn.ensemble import *
from sklearn.gaussian_process import GaussianProcess
from sklearn import svm
from sklearn import model_selection
#####################################################
#
#####################################################

n_trees=350
n_jobs=50
# max_depth=100

rf0 = RandomForestRegressor(n_estimators=n_trees,
                            oob_score=True,
                            verbose=2,
                            n_jobs=n_jobs,
                            random_state=1)

# rf0 = ExtraTreesRegressor(n_estimators=n_trees,
#                           bootstrap=True,
#                           oob_score=True,
#                           verbose=2,
#                           n_jobs=n_jobs,
#                           random_state=1)


#INSAMPLE Fit is almost perfect. Hence, will analyze performance out-of-sample.
rf0.fit(train[m2_rnn], train['booking_click'])

output_rf = rf0.predict(train[m2_rnn])
result = train[['srch_id','booking_bool', 'click_bool']]
result = result.assign(pred = output_rf)
in_nn_ndcg = neural_network.ndcg(result)

# Out-of-Sample performance
np.random.seed(10)

train = train.drop(['date_time', 'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1'], axis=1)
not_in_test_cols = ['booking_click', 'booking_bool', 'click_bool', 'position', \
'prop_id', \
'srch_id', \
'gross_bookings_usd',
'hotel_position_avg', \
'hotel_position_avg_visitor_location_country_idstd',\
'hotel_position_avg_srch_length_of_staystd',\
'hotel_position_avg_srch_destination_idstd', \
'hotel_quality_booking_visitor_location_country_idstd', \
'hotel_quality_click_visitor_location_country_idstd']
train_cols = train.loc[:, train.columns.difference(not_in_test_cols)].columns.values

train_srch_id = np.random.choice(a = train.srch_id.unique(), size = round(len(train.srch_id.unique())*0.80), replace = False)
train_ = train[pd.Series(train.srch_id).isin(train_srch_id)] #Get train set
test_ =  train[~pd.Series(train.srch_id).isin(train_srch_id)] #Get test set

# Train random forest regressor
rf0.fit(train_[train_cols], train_['booking_click'])

# Use properties of random forests to determine feature importance
importances = pd.DataFrame({'feature':train_[m2_rnn].columns,'importance':np.round(rf0.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.plot.bar()

# Good performance: 0.44 NDCG
output_rf_out = rf0.predict(test_[train_cols])
result = test_[['prop_id', 'srch_id','booking_bool', 'click_bool']]
result = result.assign(pred = output_rf_out)
ndcg_test = ndcg(result)

# Combined model

# Define random forest regressor models for booking_bool and click_bool
rf1 = RandomForestRegressor(n_estimators=n_trees, verbose=2, n_jobs=n_jobs, random_state=1)
rf2 = RandomForestRegressor(n_estimators=n_trees, verbose=2, n_jobs=n_jobs, random_state=1)

#Train random forests for booking_bool and click_bool
# rf1.fit(train_[m2_rnn], train_['booking_click'])
rf1.fit(train_[train_cols], train_['booking_click'])
rf2.fit(train_[train_cols], train_['booking_bool'])
# rf2.fit(train_[m2_rnn], train_['booking_bool'])

#Generate predictions with each model
output_rf1_out = rf1.predict(test_[train_cols])
output_rf2_out = rf2.predict(test_[train_cols])

#HARD CODE THE COMBINATIONS TO BE SURE
output_rf1rf2_0 = output_rf1_out*0 + output_rf2_out*1
output_rf1rf2_01 = output_rf1_out*0.1 + output_rf2_out*0.9
output_rf1rf2_02 = output_rf1_out*0.2 + output_rf2_out*0.8
output_rf1rf2_03 = output_rf1_out*0.3 + output_rf2_out*0.7
output_rf1rf2_04 = output_rf1_out*0.4 + output_rf2_out*0.6
output_rf1rf2_05 = output_rf1_out*0.5 + output_rf2_out*0.5
output_rf1rf2_06 = output_rf1_out*0.6 + output_rf2_out*0.4
output_rf1rf2_07 = output_rf1_out*0.7 + output_rf2_out*0.3
output_rf1rf2_08 = output_rf1_out*0.8 + output_rf2_out*0.2
output_rf1rf2_09 = output_rf1_out*0.9 + output_rf2_out*0.1
output_rf1rf2_1 = output_rf1_out*1 + output_rf2_out*0

result_0 = test_[['prop_id', 'srch_id','booking_bool', 'click_bool']]
result_0 = result_0.assign(pred= output_rf1rf2_0)
result_01 = test_[['prop_id', 'srch_id','booking_bool', 'click_bool']]
result_01 = result_01.assign(pred= output_rf1rf2_01)
result_02 = test_[['prop_id', 'srch_id','booking_bool', 'click_bool']]
result_02 = result_02.assign(pred= output_rf1rf2_02)
result_03 = test_[['prop_id', 'srch_id','booking_bool', 'click_bool']]
result_03 = result_03.assign(pred= output_rf1rf2_03)
result_04 = test_[['prop_id', 'srch_id','booking_bool', 'click_bool']]
result_04 = result_04.assign(pred= output_rf1rf2_04)
result_05 = test_[['prop_id', 'srch_id','booking_bool', 'click_bool']]
result_05 = result_05.assign(pred= output_rf1rf2_05)
result_06 = test_[['prop_id', 'srch_id','booking_bool', 'click_bool']]
result_06 = result_06.assign(pred= output_rf1rf2_06)
result_07 = test_[['prop_id', 'srch_id','booking_bool', 'click_bool']]
result_07 = result_07.assign(pred= output_rf1rf2_07)
result_08 = test_[['prop_id', 'srch_id','booking_bool', 'click_bool']]
result_08 = result_08.assign(pred= output_rf1rf2_08)
result_09 = test_[['prop_id', 'srch_id','booking_bool', 'click_bool']]
result_09 = result_09.assign(pred= output_rf1rf2_09)
result_1 = test_[['prop_id', 'srch_id','booking_bool', 'click_bool']]
result_1 = result_1.assign(pred= output_rf1rf2_1)

ndcg_test_0 = ndcg(result_0)
ndcg_test_01 = ndcg(result_01)
ndcg_test_02 = ndcg(result_02)
ndcg_test_03 = ndcg(result_03)
ndcg_test_04 = ndcg(result_04)
ndcg_test_05 = ndcg(result_05)
ndcg_test_06 = ndcg(result_06)
ndcg_test_07 = ndcg(result_07)
ndcg_test_08 = ndcg(result_08)
ndcg_test_09 = ndcg(result_09)
ndcg_test_1 = ndcg(result_1)

print(ndcg_test_0)
print(ndcg_test_01)
print(ndcg_test_02)
print(ndcg_test_03)
print(ndcg_test_04)
print(ndcg_test_05)
print(ndcg_test_06)
print(ndcg_test_07)
print(ndcg_test_08)
print(ndcg_test_09)
print(ndcg_test_1)

# ENSEMBLE METHODS
from sklearn.ensemble import *
from sklearn.gaussian_process import GaussianProcess
from sklearn import svm

n_jobs=50

rf0 = ExtraTreesRegressor(n_estimators=100,
                            bootstrap=True,
                            oob_score=True,
                            verbose=2,
                            n_jobs=n_jobs,
                            random_state=1)

train_srch_id = np.random.choice(a = train.srch_id.unique(), size = round(len(train.srch_id.unique())*0.80), replace = False)
train_ = train[pd.Series(train.srch_id).isin(train_srch_id)]
test_ =  train[~pd.Series(train.srch_id).isin(train_srch_id)]

rf0.fit(train_[m2_rnn], train_['booking_click'])
#re-fit again with more trees
_ = rf0.set_params(n_estimators=200, warm_start=True)
_ = rf0.fit(train_[m2_rnn], train_['booking_click'])
#and again...
_ = rf0.set_params(n_estimators=300, warm_start=True)
_ = rf0.fit(train_[m2_rnn], train_['booking_click'])
#and again...
_ = rf0.set_params(n_estimators=400, warm_start=True)
_ = rf0.fit(train_[m2_rnn], train_['booking_click'])
#and again...
_ = rf0.set_params(n_estimators=500, warm_start=True)
_ = rf0.fit(train_[m2_rnn], train_['booking_click'])
#and again...
_ = rf0.set_params(n_estimators=600, warm_start=True)
_ = rf0.fit(train_[m2_rnn], train_['booking_click'])
#and again...
_ = rf0.set_params(n_estimators=700, warm_start=True)
_ = rf0.fit(train_[m2_rnn], train_['booking_click'])
#and again...
_ = rf0.set_params(n_estimators=800, warm_start=True)
_ = rf0.fit(train_[m2_rnn], train_['booking_click'])
#and again...
_ = rf0.set_params(n_estimators=900, warm_start=True)
_ = rf0.fit(train_[m2_rnn], train_['booking_click'])
#and again...
_ = rf0.set_params(n_estimators=1000, warm_start=True)
_ = rf0.fit(train_[m2_rnn], train_['booking_click'])

output_rf_out = rf0.predict(test_[m2_rnn])
result = test_[['prop_id', 'srch_id','booking_bool', 'click_bool']]
result = result.assign(pred = output_rf_out)
ndcg_test = ndcg(result)

with open('results/ndcg_test_first_ensemble.pkl', 'wb') as handle:
    pickle.dump(ndcg_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

#####################################################
#MORE ENSEMBLE METHODS
#####################################################
train_srch_id = np.random.choice(a = train.srch_id.unique(), size = round(len(train.srch_id.unique())*0.80), replace = False)
train_ = train[pd.Series(train.srch_id).isin(train_srch_id)]
test_ =  train[~pd.Series(train.srch_id).isin(train_srch_id)]

seed=10
kfold = model_selection.KFold(n_splits=2, random_state=seed)

rf0 = ExtraTreesRegressor(n_estimators=100,
                            bootstrap=True,
                            # oob_score=True,
                            verbose=2,
                            n_jobs=n_jobs,
                            random_state=1)

results = model_selection.cross_val_score(rf0, train_[m2_rnn], train_['booking_click'], cv=kfold)



'Logistic Regression'

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(train_[m2_rnn], train_['booking_click'])

#Bad Performance: 0.24 NDCG
output_lr_out = logreg.predict(test[m2_rnn])
result = test[['srch_id','booking_bool', 'click_bool']]
result = result.assign(pred = output_lr_out)



'Decision Tree'
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_[m2_rnn], train_['booking_click'])

#Better than Logistic, worse than randome forests 0.3625 NDCG
output_dt_out = decision_tree.predict(test[m2_rnn])
result = test[['srch_id','booking_bool', 'click_bool']]
result = result.assign(pred = output_dt_out)



'LambdaMart'
from pyltr.models import LambdaMART
from pyltr.metrics import NDCG
from pyltr.models.monitors import ValidationMonitor

train_['rel'] = train_['booking_bool']*5 + train_['click_bool'] #should have 5 as maximum
train_.loc[train_['rel'] == 6, 'rel'] = 5 # 6 can only occure if click and bool are 1. Hence, cap at 5.
#add the srch_id as the index to the new train values
m2_rnn.append('srch_id')
LambdaMART_train = train_[m2_rnn].set_index('srch_id').sort_index()
qids_val = LambdaMART_train.index

metric = NDCG(k=100)
monitor = ValidationMonitor(LambdaMART_train, train_['rel'], qids_val.values, metric=metric)
model = LambdaMART(
        metric=metric,
        max_depth = 10,
        n_estimators=1000,
        learning_rate=.1,
        verbose=1,
        max_features=0.5,
        query_subsample=0.5,
        max_leaf_nodes=10,
        min_samples_leaf=64
        )
model.fit(train_[m2_rnn], train_['rel'], qids_val, monitor=monitor)

#creating the data to test the models with
LambdaMART_testX = test_[m2_rnn].set_index('srch_id').sort_index()
LambdaMART_testY = test_[['srch_id','booking_bool', 'click_bool']]
LambdaMART_test_qids = LambdaMART_testX.index

#LambdaMART predictions
Lambda_pred = model.predict(LambdaMART_testX)
print('Random ranking: ', metric.calc_mean_random(LambdaMART_test_qids, LambdaMART_testY))
print('Our model: ', metric.calc_mean(LambdaMART_test_qids, LambdaMART_testY, Lambda_pred))

#old way, probably not correct for LambdaMART
output_LM_out = model.predict(test[m2_rnn])
result = test[['srch_id','booking_bool', 'click_bool']]
result = result.assign(pred = output_LM_out)
