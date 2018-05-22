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
import seaborn as sns

plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import scale


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

#Scale price_rank and star_rank as they where mistakenly not scaled
train['prince_rank'] = scale(train['price_rank'])
train['star_rank'] = scale(train['star_rank'])
test['prince_rank'] = scale(test['price_rank'])
test['star_rank'] = scale(test['star_rank']) 

#Define all variables that are not used for training each model


'Recurrent Neural Network'

from MachineLearning import NeuralNetwork
from MachineLearning import NeuronLayer
from sklearn.neural_network import MLPClassifier

#Different variable selections

train['booking_click'] = train.booking_bool #Define a variable that is one if click or booked
train.loc[train['booking_click'] == 0, 'booking_click'] = train.loc[train['booking_click'] == 0, 'click_bool']


m1_rnn = ['hotel_quality_click_srch_length_of_staystd',
       'hotel_quality_booking_srch_length_of_staystd', 'hotel_quality_click_prop_country_idstd',
       'hotel_quality_click_srch_destination_idstd', 
       'hotel_quality_booking_srch_destination_idstd','hotel_quality_booking_prop_country_idstd',
       'hotel_quality_click_visitor_location_country_idstd',
       'hotel_quality_booking_visitor_location_country_idstd','hotel_quality_click_monthstd',
       'hotel_quality_booking_monthstd']  #Diversification only on key

m2_rnn= ['price_rank', 'star_rank','price_difference_rank',
         'hotel_quality_click_monthstd',          
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
       'dummy_top1520_avg', 'dummy_top2050_avg'] #Diversification only on explanatory variables

m3_rnn= ['prop_brand_bool', 'promotion_flag', 'price_rank', 'star_rank', 'price_difference_rank', 
         'prop_starrating_monotonic',  'dummy_starrating_diff_low', 
         'dummy_usd_diff_high', 'dummy_usd_diff_low',
         'hotel_quality_click_monthstd',
         'hotel_quality_booking_monthstd', 'hotel_position_avg_srch_destination_idstd',
         'prop_location_score1_monthstd', 'price_difference_prop_country_idstd',
         'prop_location_score2_monthstd', 'price_usd_srch_destination_idstd',
         'hotel_position_avg_monthstd','price_usd_srch_destination_idstd'] #Diversification only on explanatory variables

np.random.seed(10)

#Define and train neural network
neurons1 = 6
neurons2 = 1

layer1 = NeuronLayer(neurons1, len(train[m3_rnn].columns))
layer2 = NeuronLayer(neurons2, neurons1)

activation_function = 'Sigmoid'

neural_network = NeuralNetwork(layer1, layer2, activation_function)
neural_network.train(train[m3_rnn], np.matrix(train['booking_click']).T, 20)
hidden_state, output = neural_network.think(train[m3_rnn])

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,1), random_state=1, activation = 'logistic')
clf.fit(train[m3_rnn], np.array(train['booking_click']).T) 
output_ = clf.predict(train[m3_rnn])

#Insample Fit is very bad. This imply the neural network is not an appropate model to train this time of dataset
#Performance: 0.3 to 0.36 NDCG by optimizing parameters and incorporating variables with highest importance.
result = train[['srch_id','booking_bool', 'click_bool']]
result = result.assign(pred = output_)
in_nn_ndcg = neural_network.ndcg(result)



'Random Forest Regressor'
from sklearn.ensemble import *
from sklearn.gaussian_process import GaussianProcess
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

n_trees=150
n_jobs=-1
#max_depth=120

rf0 = RandomForestRegressor(n_estimators=n_trees, verbose=2, n_jobs=n_jobs, random_state=1)

rf0.fit(train[m3_rnn], train['booking_click'])

#Insample Fit is almost perfect. Hence, will analyze performance out-of-sample.
output_rf = rf0.predict(train[m3_rnn])
result = result.assign(pred = output_rf)
in_nn_ndcg = neural_network.ndcg(result)

#Out-of-Sample performance
np.random.seed(10)

train_srch_id = np.random.choice(a = train.srch_id.unique(), size = round(len(train.srch_id.unique())*0.80), replace = False)
train_ = train[pd.Series(train.srch_id).isin(train_srch_id)] #Get train set
test =  train[~pd.Series(train.srch_id).isin(train_srch_id)] #Get test set

#Train random forest regressor
rf0.fit(train_[m2_rnn], train_['booking_click'])

#Use properties of random forests to determine feature importance
importances = pd.DataFrame({'feature':train_[m2_rnn].columns,'importance':np.round(rf0.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.plot.bar()

#Good performance: from 0.41 to 0.475 NDCG by optimizing parameters and removing variables with low importance
#Performance as a function of n_trees: (0.40 NDCG, 5 trees), (0.475, 50 trees), (0.48, 150 trees), (0.51, 300 trees)
output_rf_out = rf0.predict(test[m2_rnn])
result = test[['srch_id','booking_bool', 'click_bool']]
result = result.assign(pred = output_rf_out)

#Alternative to using booking_click. One can train two models, using click and booking as dependent variables respectivley.
rf1 = RandomForestRegressor(n_estimators=n_trees, verbose=2, n_jobs=n_jobs, random_state=1)
rf1.fit(train_[m2_rnn], train_['booking_bool'])
rf2 = RandomForestRegressor(n_estimators=n_trees, verbose=2, n_jobs=n_jobs, random_state=1)
rf2.fit(train_[m2_rnn], train_['click_bool'])
#The predictions can be linearly combined to generate our prediction out-of-sample
out_rf1_out =  rf1.predict(test[m2_rnn])
out_rf2_out =  rf1.predict(test[m2_rnn])
for w in range(0,1, 0.1):
    result[output_rf_out + w] = out_rf1_out*w + out_rf2_out*(1-w)
#We can determine the NCDG for each linear combination and select the optimal w to generate the forecast for the test submission file.
    


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
qids_val = train_['srch_id']

metric = NDCG(k=40)
monitor = ValidationMonitor(train_[m2_rnn], train_['rel'], qids_val.values, metric=metric, stop_after=50)
model = LambdaMART(metric=metric, max_depth = 6, n_estimators=100, learning_rate=.1, verbose=1)
model.fit(train_[m2_rnn], train_['rel'], qids_val, monitor=monitor)

output_LM_out = model.predict(test[m2_rnn])
result = test[['srch_id','booking_bool', 'click_bool']]
result = result.assign(pred = output_LM_out)