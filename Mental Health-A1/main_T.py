# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 22:11:05 2018

@author: Felix Farias Fueyo
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

#Missing Value Analysis
dAgg_df = dAgg[:,:,:].unstack(level = 1)
#Binnary Variables
#Observation 1: Users can start at differnt times
bin_data = ['sms', "call"]
dAgg_df['call'] = dAgg_df['call'].fillna(value = 0)
dAgg_df['sms'] = dAgg_df['sms'].fillna(value = 0)
#Numeric Variables -> Given A1, any NaN as of 02-17 is a missing value of a given variable
#Observation 1: Duration variables are only recoded if user enters app.
#   Hence, if user does not enter app NaN is obtained. However, this is equivalent
#   to a duration of 0. Hence, duration NaNs are replaced by 0.
num_data = dAgg_df.columns
duration_data = list([s for s in num_data if "app" in s])
duration_data.append('screen')
for i in duration_data:
    dAgg_df[i] = dAgg_df[i].fillna(value = 0)
num_data = num_data[np.logical_and('sms' != num_data,'call' != num_data)]
miss = dAgg_df.isnull().sum()/len(dAgg_df)
#Visualization
miss = miss[miss > 0]
miss.sort_values(inplace=True)
miss = miss.to_frame()
miss.columns = ['count']
miss.index.names = ['Name']
miss['Name'] = miss.index
sns.set(style="whitegrid", color_codes=True)
sns.barplot(x = 'Name', y = 'count', data=miss)
plt.xticks(rotation = 90)
sns.plt.show()
#Interpretation: Too many missing values -> Must take actions
#Observation 1: Trade-off between time window and missing values. 
#Observation 2: Most missing values at the begining of the observation period.
#Action: Start observation period as of date 2014-03-27
temp = round(dAgg_df.index.get_level_values('time').unique().values.size/12) #All dates
all_dates = dAgg_df.index.get_level_values('time').unique().values
all_dates.sort()
train_dates = all_dates[30:9*temp] 
test_dates = all_dates[9*temp:12*temp]
del temp
dAgg_df['mood_MA'] = pd.rolling_mean(dAgg_df['mood'],14,1)
train_df = dAgg_df.iloc[np.logical_and(train_dates.max()>=dAgg_df.index.get_level_values('time'),
                        dAgg_df.index.get_level_values('time')>=train_dates.min())] #training set
#Observation 1: There are some users that stop usage before end of training date and 
#               some that start after begin of training date.
#Observation 2: Keeping these users in dataset creates missing values in the future.
#Observation 3: Also test set would consist of missing values. 
#Action: Remove users that do not have complete observations during training period.
co_users = list()
for i in list(train_df.index.get_level_values('id').unique().values):
    temp = train_df.iloc[train_df.index.get_level_values('id') == i] #look for users
    if temp.index.get_level_values('time').values.max() == train_dates.max() and temp.index.get_level_values('time').values.min() == train_dates.min():
        co_users.append(i)
train_df = train_df.loc[co_users]
#Observation 1: Users in test set can drop out during testing period
#Action: Test the strategy for the remaining users separetely! Performance of the 
#       calibration should be determined by the average RMSE accross users.    
test_df = dAgg_df.iloc[dAgg_df.index.get_level_values('time')>=test_dates.min()] #training set
test_df = test_df.loc[co_users]
miss = train_df.isnull().sum()/len(train_df)
#Visualization
miss.sort_values(inplace=True)
miss = miss.to_frame()
miss.columns = ['count']
miss.index.names = ['Name']
miss['Name'] = miss.index
miss1 = miss[miss > 0]
sns.set(style="whitegrid", color_codes=True)
sns.barplot(x = 'Name', y = 'count', data=miss1)
plt.xticks(rotation = 90)
sns.plt.show()
#Interpretation: Significant improvement. 
#Observation 1: Some variables still show significant amount of missing values.
#Action: Observe correlations and decide weather or not to remove a variable.
corr = train_df.corr()
temp = corr.iloc['mood' == corr.index.get_level_values('variable')]
corr_mood = pd.DataFrame(temp.values).transpose()
corr_mood.columns = ['correlation']
corr_mood.index.names = ['Name']
corr_mood['Name'] = temp.columns
corr_mood = corr_mood.sort_values(by = 'correlation')
corr_mood = corr_mood[np.logical_and('sms'!=corr_mood['Name'],
                           'call'!=corr_mood['Name'])]
corr_mood = corr_mood['mood'!=corr_mood['Name']]
sns.set(style="whitegrid", color_codes=True)
sns.barplot(x = 'Name', y = 'correlation', data=corr_mood)
plt.xticks(rotation = 90)
sns.plt.show()
#Interpretation: Two different cases, c1 = high missing + no correlation (<|0.1|)
#and c2 = high missing + some correlation. 
#c1 variables are Office, Utilities, Unknown, Travel, Finance, Game
#c2 variables are Weather
#Observation 1: Variables with missing values have low correlation with 'mood'
#Observation 2: Only exceptions is Weather (neg. corr.)
#Observation 3: Weather is the variable with the highest amount of NAs
#Action 1: Remove c1 type variables. (Note: Removed variables depends on threshold (<|0.1|)
#Action 2: Remove c2. (This should be discussed with the TA! Potentially important, but 
#          extremely many NAs, recommendations?)
miss_var = miss[miss['count']<0.60]['Name'].values.tolist()
corr_var = corr_mood[abs(corr_mood['correlation'])>0.2]['Name'].values.tolist()
train_variables = list(set(corr_var).intersection(miss_var))
train_variables.insert(len(train_variables),'sms')
train_variables.insert(len(train_variables),'call')
train_variables.insert(len(train_variables),'mood')
train_df = train_df[train_variables]
test_df = test_df[train_variables]
#Visualize
nd = pd.melt(train_df, value_vars = train_variables)
n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)
n1 = n1.map(sns.distplot, 'value')
n1
#Remaining Missing Values
#Approach 1: fill all missing values of the remaining Vs. with corresponding medians
for i in list(set(corr_var).intersection(miss_var)):
    train_df[i].fillna(np.nanmedian(train_df[i].values), inplace=True)
    
train_df['mood'].fillna(np.nanmedian(train_df['mood'].values), inplace=True)
#Note: test set missing values are not filled. Forecast each user separately.

#Binary Data Analysis
#Pivot tables: to provide argument in favor of any behavior
call_pivot = train_df.pivot_table(index='call', values='mood', aggfunc=np.median)
sms_pivot = train_df.pivot_table(index='sms', values='mood', aggfunc=np.median)
#Interpretation: median value same for categories. Hence, no explanatory power.
#Action: Do ANOVA analysis to reinforce conclusion
cat = ['sms','call']
bin_data = train_df[['mood','sms','call']]
def anova(frame):
    anv = pd.DataFrame()
    anv['features'] = cat
    pvals = []
    for c in cat:
           samples = []
           for cls in frame[c].unique():
                  s = frame[frame[c] == cls]['mood'].values
                  samples.append(s)
           pval = stats.f_oneway(*samples)[1]
           pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')
k = anova(bin_data)
#Observation 1: Variables are not significant at the 0.01 or 0.05 confidence level.
#Action 1: Remove sms and call from data set.
if k['pval'][1] > 0.05 and k['pval'][0] > 0.05:
    train_variables = list(set(corr_var).intersection(miss_var))
    train_variables.insert(len(train_variables),'mood')
    train_df = train_df[train_variables]
    test_df = test_df[train_variables]


'''Transform Numeric Variables'''
#Standardize
train_variables1 = train_variables
train_variables1.remove('mood')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_df[train_variables1])
scaled = scaler.transform(train_df[train_variables1])
for i, col in enumerate(train_variables1):
    train_df[col] = scaled[:,i]
    
for i in list(set(corr_var).intersection(miss_var)):
    test_df[i].fillna(np.nanmedian(test_df[i].values), inplace=True)    
    test_df['mood'].fillna(np.nanmedian(test_df['mood'].values), inplace=True)

scaler.fit(test_df[train_variables1])
scaled = scaler.transform(test_df[train_variables1])
for i, col in enumerate(train_variables1):
    test_df[col] = scaled[:,i]

'''Model Training & Evaluation'''
#from sklearn.metrics import mean_squared_error
#from sklearn.preprocessing import StandardScaler

#Predict for each user indipendently. Example for AS14.33
from MachineLearning import NeuralNetwork
from MachineLearning import NeuronLayer
np.random.seed(1)
#Define and train neural network
neurons1 = 3
neurons2 = 1
layer1 = NeuronLayer(neurons1, len(train_df.columns)-1)
layer2 = NeuronLayer(neurons2, neurons1)
#Observation 1: ReLu's gradient drives the network's weights to explode. 
#Action: Use Sigmoid, which has much ower gradient values.
activation_function = 'ReLu'
#print("Stage 1) Random starting synaptic weights: ")
#neural_network.print_weights()

#Initialize matrix to store performance out of sample
performance_rmse = np.zeros((len(test_df.index.get_level_values('id').unique()),2))
performance_rmse = pd.DataFrame(performance_rmse)
performance_rmse.columns = ['Neural Network', 'Benchmark']
performance_rmse.index.names = ['Patient']
performance_rmse['Patient'] = test_df.index.get_level_values('id').unique()
performance_mad = np.zeros((len(test_df.index.get_level_values('id').unique()),2))
performance_mad = pd.DataFrame(performance_mad)
performance_mad.columns = ['Neural Network', 'Benchmark']
performance_mad.index.names = ['Patient']
performance_mad['Patient'] = test_df.index.get_level_values('id').unique()

for i in test_df.index.get_level_values('id').unique():
    train_ex = train_df.iloc[train_df.index.get_level_values('id') == i]
    #Train network to predict one day ahead of time fitting aX_t = y_{t+1}
    train_ex['mood'] = train_ex['mood'].shift(-1)
    train_ex = train_ex.drop(train_ex.index[len(train_ex)-1])
    neural_network = NeuralNetwork(layer1, layer2, activation_function)
    neural_network.train(train_ex.drop('mood', 1), np.matrix(train_ex['mood']).T, 30000)

    #print("Stage 2) Trained synaptic weights: ")
    #neural_network.print_weights()

    test_ex = test_df.iloc[test_df.index.get_level_values('id') == i]
    #Drop last row, as there is no value of mood available at T+1
    test_ex['mood'] = test_ex['mood'].shift(-1)
    test_ex = test_ex.drop(test_ex.index[len(test_ex)-1])
    hidden_state, output = neural_network.think(test_ex.drop('mood', 1))

    #Performance out-of-sample
    #RMSE
    nn_rmse = neural_network.rmse(output, test_ex['mood'])
    performance_rmse.loc[performance_rmse['Patient'] == i, 'Neural Network'] = nn_rmse
    #MAD
    nn_mad = neural_network.mad(output, test_ex['mood'])
    performance_mad.loc[performance_mad['Patient'] == i, 'Neural Network'] = nn_mad

    #Benchmark to outperfrom: median of historical values 
    benchmark = np.ones((len(test_ex['mood']),1)) * train_ex['mood'].median()
    #RMSE
    ben_rmse = neural_network.rmse(benchmark, test_ex['mood'])
    performance_rmse.loc[performance_rmse['Patient'] == i, 'Benchmark'] = ben_rmse
    #MAD
    ben_mad = neural_network.mad(benchmark, test_ex['mood'])
    performance_mad.loc[performance_mad['Patient'] == i, 'Benchmark'] = ben_mad

sum(performance_rmse['Neural Network'] > performance_rmse['Benchmark'])/len(performance_rmse)
sum(performance_mad['Neural Network'] > performance_mad['Benchmark'])/len(performance_mad)
