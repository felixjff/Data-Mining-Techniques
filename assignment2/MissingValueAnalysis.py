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
import math

plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm

#loading data
dataset = pd.read_csv('Data/training_set_VU_DM_2014.csv')

''' Missing Value Analysis '''
## Start: Visitor History ##
dataset['starrating_diff'] = 0
dataset['usd_diff'] = 0
dataset['starrating_diff'] = abs(dataset['visitor_hist_starrating'] - dataset['prop_starrating'])
dataset.loc[dataset['visitor_hist_adr_usd'] == 0, 'visitor_hist_adr_usd'] = 0
dataset['usd_diff'] = abs(np.log10(dataset['visitor_hist_adr_usd']) - np.log10(dataset['price_usd']))

pct_usd = pd.DataFrame(np.zeros((11,2))) #10 categories plus NAs
pct_usd.columns = ['Booking', 'Click']
pct_star = pd.DataFrame(np.zeros((6,2)))
pct_star.columns = ['Booking', 'Click']

for i in range(10):
    dataset.loc[np.logical_and( i/10 < dataset['usd_diff'], dataset['usd_diff'] <= (i+1)/10 ), 'usd_diff'] = i+1
    pct_usd.iloc[i] = dict(Booking = dataset.loc[dataset['usd_diff'] == i, 'booking_bool'].sum()/len(dataset.loc[dataset['usd_diff'] == i]),
                Click = dataset.loc[dataset['usd_diff'] == i, 'click_bool'].sum()/len(dataset.loc[dataset['usd_diff'] == i]))

pct_usd.iloc[10] = dict(Booking = dataset.loc[dataset['usd_diff'].isnull(), 'booking_bool'].sum()/dataset['usd_diff'].isnull().sum(),
                Click = dataset.loc[dataset['usd_diff'].isnull(), 'click_bool'].sum()/dataset['usd_diff'].isnull().sum())

for i in range(5):
    dataset.loc[np.logical_and( i < dataset['starrating_diff'], dataset['starrating_diff'] <= (i+1) ), 'starrating_diff'] = i+1
    pct_star.iloc[i] = dict(Booking = dataset.loc[dataset['starrating_diff'] == i, 'booking_bool'].sum()/len(dataset.loc[dataset['starrating_diff'] == i]),
                 Click = dataset.loc[dataset['starrating_diff'] == i, 'click_bool'].sum()/len(dataset.loc[dataset['starrating_diff'] == i]))

pct_star.iloc[5] = dict(Booking = dataset.loc[dataset['starrating_diff'].isnull(), 'booking_bool'].sum()/dataset['starrating_diff'].isnull().sum(),
                 Click = dataset.loc[dataset['starrating_diff'].isnull(), 'click_bool'].sum()/dataset['starrating_diff'].isnull().sum())

pct_usd.index.names = ['Category']
pct_usd['Category'] = pct_usd.index
pct_star.index.names = ['Category']
pct_star['Category'] = pct_star.index

pct_star[['Booking', 'Click']] = pct_star[['Booking', 'Click']]*100
pct_usd[['Booking', 'Click']] = pct_usd[['Booking', 'Click']]*100
pct_star.columns = ['% of Hotels being Booked', '% of Hotels being Click', 'Category']
pct_usd.columns = ['% of Hotels being Booked', '% of Hotels being Click', 'Category']
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

dataset['dummy_starrating_diff_high'] = 0
dataset['dummy_starrating_diff_low'] = 0
dataset['dummy_usd_diff_high'] = 0
dataset['dummy_usd_diff_low'] = 0

for i in usd_diff_low:
    dataset.loc[dataset['usd_diff'] == i, 'dummy_usd_diff_low'] = 1 
for i in usd_diff_high:
    dataset.loc[dataset['usd_diff'] == i, 'dummy_usd_diff_high'] = 1 
for i in star_diff_low:
    dataset.loc[dataset['starrating_diff'] == i, 'dummy_starrating_diff_low'] = 1 
for i in star_diff_high:
    dataset.loc[dataset['starrating_diff'] == i, 'dummy_starrating_diff_high'] = 1
    
dataset = dataset.drop(['usd_diff', 'starrating_diff', 'visitor_hist_adr_usd', 'visitor_hist_starrating'], axis = 1)    
## Visitor History ##

## Hotel Descriptions ## (prop_review_score & prop_location_score2 & srch_query_affinity_score)
hotel_vars = ['prop_review_score', 'prop_location_score2', 'srch_query_affinity_score']

pct_booking_NAs = pd.DataFrame(np.zeros((len(hotel_vars), 2)))
pct_booking_NAs.columns = ['NA', 'Not NA']
pct_booking_NAs.index.names = ['Variables']
pct_booking_NAs['Variables'] = hotel_vars
pct_click_NAs = pd.DataFrame(np.zeros((len(hotel_vars), 2)))
pct_click_NAs.columns = ['NA', 'Not NA']
pct_click_NAs.index.names = ['Variables']
pct_click_NAs['Variables'] = hotel_vars

for i in hotel_vars:
    pct_booking_NAs.loc[pct_booking_NAs['Variables'] == i, 'NA'] =  dataset.loc[dataset[i].isnull(), 'booking_bool'].sum()/dataset[i].isnull().sum()
    pct_booking_NAs.loc[pct_booking_NAs['Variables'] == i, 'Not NA'] = dataset.loc[~dataset[i].isnull(), 'booking_bool'].sum()/sum(~dataset[i].isnull())
    pct_click_NAs.loc[pct_click_NAs['Variables'] == i, 'NA'] = dataset.loc[dataset[i].isnull(), 'click_bool'].sum()/dataset[i].isnull().sum()
    pct_click_NAs.loc[pct_click_NAs['Variables'] == i, 'Not NA'] = dataset.loc[~dataset[i].isnull(), 'click_bool'].sum()/sum(~dataset[i].isnull())

pct_booking_NAs[['NA', 'Not NA']] = pct_booking_NAs[['NA', 'Not NA']]*100
pct_booking_NAs = pd.melt(pct_booking_NAs, id_vars="Variables", var_name="Value", value_name="% of Hotels being Booked")
sns.set(style="whitegrid", color_codes=True)
sns.factorplot(x='Variables', y='% of Hotels being Booked', hue='Value', data= pct_booking_NAs, kind='bar')
plt.xticks(rotation = 45)

pct_click_NAs[['NA', 'Not NA']] = pct_click_NAs[['NA', 'Not NA']]*100
pct_click_NAs = pd.melt(pct_click_NAs, id_vars="Variables", var_name="Value", value_name="% of Hotels being Clicked")
sns.set(style="whitegrid", color_codes=True)
sns.factorplot(x='Variables', y='% of Hotels being Clicked', hue='Value', data= pct_click_NAs, kind='bar')
plt.xticks(rotation = 45)

#Bar plots show user do not like to book and click hotels with missing values. Missing values for these variables are imputted
#with worse case scenario. Case for affinity is not straightforward though. Hence, a different approach to imput missing values 
#for affinity could be considered. 
dataset.loc[dataset['prop_review_score'].isnull(), 'prop_review_score'] = 1  #lowest. Not zero such that it does not overlap with missing rating case
dataset.loc[dataset['prop_location_score2'].isnull(), 'prop_location_score2'] = 0.69 #sample minimum
dataset.loc[dataset['srch_query_affinity_score'].isnull(), 'srch_query_affinity_score'] = np.log(0.001) #99.9% probability it won't be clicked
## Hotel Descriptions ##

## Comparison Variables ##
comp_vars = ['comp1_rate','comp2_rate','comp3_rate','comp4_rate','comp5_rate','comp6_rate','comp7_rate','comp8_rate']

pct_comp_booking = pd.DataFrame(np.zeros((len(comp_vars), 4)))
pct_comp_booking.columns = ['Higher', 'Same' , 'Lower', 'NA']
pct_comp_booking.index.names = ['Variables']
pct_comp_booking['Variables'] = comp_vars
pct_comp_click = pd.DataFrame(np.zeros((len(comp_vars), 4)))
pct_comp_click.columns = ['Higher', 'Same' , 'Lower', 'NA']
pct_comp_click.index.names = ['Variables']
pct_comp_click['Variables'] = comp_vars

for i in comp_vars:
    pct_comp_booking.loc[pct_comp_booking['Variables'] == i, 'Higher'] =  dataset.loc[dataset[i] == -1, 'booking_bool'].sum()/len(dataset.loc[dataset[i] == 1])
    pct_comp_booking.loc[pct_comp_booking['Variables'] == i, 'Same'] = dataset.loc[dataset[i] == 0, 'booking_bool'].sum()/len(dataset.loc[dataset[i] == 0])
    pct_comp_booking.loc[pct_comp_booking['Variables'] == i, 'Lower'] = dataset.loc[dataset[i] == 1, 'booking_bool'].sum()/len(dataset.loc[dataset[i] == -1])
    pct_comp_booking.loc[pct_comp_booking['Variables'] == i, 'NA'] = dataset.loc[dataset[i].isnull(), 'booking_bool'].sum()/len(dataset.loc[dataset[i].isnull()])    
    pct_comp_click.loc[pct_comp_click['Variables'] == i, 'Higher'] =  dataset.loc[dataset[i] == -1, 'click_bool'].sum()/len(dataset.loc[dataset[i] == 1])
    pct_comp_click.loc[pct_comp_click['Variables'] == i, 'Same'] = dataset.loc[dataset[i] == 0, 'click_bool'].sum()/len(dataset.loc[dataset[i] == 0])
    pct_comp_click.loc[pct_comp_click['Variables'] == i, 'Lower'] = dataset.loc[dataset[i] == 1, 'click_bool'].sum()/len(dataset.loc[dataset[i] == -1])
    pct_comp_click.loc[pct_comp_click['Variables'] == i, 'NA'] = dataset.loc[dataset[i].isnull(), 'click_bool'].sum()/len(dataset.loc[dataset[i].isnull()])    

pct_comp_booking[['Higher', 'Same' , 'Lower', 'NA']] = pct_comp_booking[['Higher', 'Same' , 'Lower', 'NA']]*100
pct_comp_click[['Higher', 'Same' , 'Lower', 'NA']] = pct_comp_click[['Higher', 'Same' , 'Lower', 'NA']]*100

#Table shows some tendency of users to book in expedia if the prices are lower. Hence, one should take this into account. 
#The easy approach is to assume there is no difference in prices across competitors. Thus, setting all missing values equal to zero. 
for i in comp_vars:
    dataset.loc[dataset[i].isnull(), i] = 0    
## Comparison Variables ##

## Other Variables ##
dataset.loc[dataset['promotion_flag'].isnull(), 'promotion_flag'] = 0
dataset.loc[dataset['orig_destination_distance'].isnull(), 'orig_destination_distance'] = -1
dataset.loc[dataset['gross_bookings_usd'].isnull(), 'gross_bookings_usd'] = 0
## Other Variables ##

#Check for missing values
missing_values = []
for i in dataset.columns.values: 
    missing_values.append([i, len(dataset.loc[dataset[i] == math.nan,i])])

dataset.to_csv('data/MissingValueAnalysis/training_set_VU_DM_2014.csv')