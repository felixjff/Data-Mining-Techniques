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
import xgboost as xgb

plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm

#loading data
dataset = pd.read_csv("Data/dataset_mood_smartphone.csv")
#dataset['time'] = pd.to_datetime(dataset['time'], format = '%Y%m%D', errors = 'coerce') #convert dates to datetime object (or a series), seems to miscreate 'datetime'
dataset['time'] = pd.to_datetime(dataset['time'])
variables = dataset['variable'].unique()

#Missing values? 
dataset['variable'].isnull().any()

#Aggregation to daily average 
dataset = pd.DataFrame(dataset)
dAgg = dataset.groupby(['id', 'variable', dataset['time'].dt.date])['value'].mean() #hierarchical grouping: id > time > variable, averaging the values of the variable. Seems to lose the exact date information
print(dAgg.head(3000)) #to check grouping and values

#Distribution: target variable (assuming no missing values)
sns.distplot(dAgg[:,'mood',:]) #Left-skewed distribution.

#Missing Value Analysis
dAgg_df = dAgg[:,:,:].unstack(level = 1)
#Binnary Variables -> Assumption 1: phones tracked as of 02-17. Hence, no missing variables
#for 'call' or 'sms', i.e. if NaN, then set to 0. 
bin_data = ['sms', "call"]
dAgg_df['call'] = dAgg_df['call'].fillna(value = 0)
dAgg_df['sms'] = dAgg_df['sms'].fillna(value = 0)
#Numeric Variables -> Given A1, any NaN as of 02-17 is a missing value of a given variable
num_data = dAgg_df.columns
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
temp = round(dAgg_df.index.get_level_values('time').unique().values.size/10) #All dates
all_dates = dAgg_df.index.get_level_values('time').unique().values
all_dates.sort()
train_dates = all_dates[38:9*temp] 
test_dates = all_dates[9*temp:10*temp]
del temp
train_df = dAgg_df.iloc[np.logical_and(train_dates.max()>=dAgg_df.index.get_level_values('time'),
                        dAgg_df.index.get_level_values('time')>=train_dates.min())] #training set
test_df = dAgg_df.iloc[dAgg_df.index.get_level_values('time')>=test_dates.min()] #training set
miss = train_df.isnull().sum()/len(train_df)
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
corr_var = corr_mood[abs(corr_mood['correlation'])>0.05]['Name'].values.tolist()
train_variables = list(set(corr_var).intersection(miss_var))
train_variables.insert(len(train_variables),'sms')
train_variables.insert(len(train_variables),'call')
train_variables.insert(len(train_variables),'mood')
train_df = train_df[train_variables]
test_df = test_df[train_variables]
#Remaining Missing Values
#Approach 1: fill all missing values of the remaining Vs. with corresponding medians
for i in list(set(corr_var).intersection(miss_var)):
    train_df[i].fillna(np.nanmedian(train_df[i].values), inplace=True)
    test_df[i].fillna(np.nanmedian(test_df[i].values), inplace=True)
    
train_df['mood'].fillna(np.nanmedian(train_df['mood'].values), inplace=True)
test_df['mood'].fillna(np.nanmedian(test_df['mood'].values), inplace=True)

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
train_variables = list(set(corr_var).intersection(miss_var))
train_variables.insert(len(train_variables),'mood')
train_df = train_df[train_variables]
test_df = test_df[train_variables]

###Transform Numeric Variables###
#Reduce length of tails: Check skewnes using log(x + 1)
from scipy.stats import skew
skewed = train_df.apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > 0.75]
skewed = skewed.index
train_df[skewed] = np.log1p(train_df[skewed])
test_df[skewed] = np.log1p(test_df[skewed])
del test_df['mood']

#Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_df)
scaled = scaler.transform(train_df)
for i, col in enumerate(train_df.columns):
       train_df[col] = scaled[:,i]

numeric_features = list(train_df.columns)
numeric_features.remove('mood')
scaled = scaler.fit_transform(test_df[numeric_features])
for i, col in enumerate(numeric_features):
      test_df[col] = scaled[:,i]       
       

###Model Training & Evaluation###
#create a label set (for later)
label_df = pd.DataFrame(index = train_df.index, columns = ['mood'])
label_df['mood'] = np.log(train_df['mood'])
#Improve parameter selection by performing cross-validation!
regr = xgb.XGBRegressor(colsample_bytree=0.2,
                       gamma=0.0,
                       learning_rate=0.05,
                       max_depth=6,
                       min_child_weight=1.5,
                       n_estimators=7200,
                       reg_alpha=0.9,
                       reg_lambda=0.6,
                       subsample=0.2,
                       seed=42,
                       silent=1)

regr.fit(train_df, label_df)      