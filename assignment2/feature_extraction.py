import numpy as np
import pandas as pd

testing = 0
if testing == 0:
    with open('data/training_set_VU_DM_2014.csv', 'r') as csvfile:
        sample = pd.read_csv(csvfile)
else:
    with open('data/test_set_VU_DM_2014.csv', 'r') as csvfile:
        sample = pd.read_csv(csvfile)

##Decrease Sample Size##
np.random.seed(10)

train_srch_id = np.random.choice(a = sample.srch_id.unique(), size = round(len(sample.srch_id.unique())*0.25), replace = False)
train = sample[pd.Series(sample.srch_id).isin(train_srch_id)]
del sample
del train_srch_id

##PRICE DIFFERENCE##
#difference in current search price to previous search price
price_difference_series = train.set_index(['date_time']).sort_index().groupby(['prop_id']).apply(lambda x: x.price_usd.diff()).reset_index()
train = train.sort_values(['prop_id','date_time']).reset_index(drop=True)
train['price_difference'] = price_difference_series.price_usd
train['price_difference'] = train['price_difference'].fillna(0)

##HOTEL QUALITY##
#number of times each prop_id has been booked
booking_series = train.groupby(['prop_id']).sum().booking_bool
booking_series = booking_series + 1

#number of times each prop_id has been clicked
click_series = train.groupby(['prop_id']).sum().click_bool
click_series = click_series + 1

#number of times each prop_id has appeared in all searches
count_series = train.groupby(['prop_id']).count().srch_id

hotel_quality_booking = booking_series.divide(count_series)
hotel_quality_click = click_series.divide(count_series)

#append the hotel quality to the train dataframe
train = train.set_index(['prop_id']).sort_index()
train['hotel_quality_booking'] = hotel_quality_booking
train['hotel_quality_click'] = hotel_quality_click

#reset the index back to normal
train = train.reset_index()

##HOTEL POSITION##
#position of the hotel in the same destination in previous and next searches
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

#train = train.sort_values(['prop_id','date_time']).reset_index(drop=True)
train['hotel_position_avg'] = hotel_position_series.position
#train['hotel_position_avg'] = train['hotel_position_avg'].fillna(-1)

##PRICE RANK##
#order of the price within srch_id
train['price_rank'] = train.groupby(['srch_id'])['price_usd'].rank()

##STAR RANK##
#order of the star rating within srch_id
train['star_rank'] = train.groupby(['srch_id'])['prop_starrating'].rank()

##PRICE DIFFERENCE RANK##
#difference in price, negative difference ranked higher than positive difference,
#I.e. if a property reduces in price between searches this is ranked high
train['price_difference_rank'] = train.groupby(['prop_id'])['price_difference'].rank()

##Monotonic Property Star Rating
train['prop_starrating_monotonic'] = abs(train.prop_starrating - np.mean(train.loc[train['booking_bool'] == 1].prop_starrating))

if testing == 0:
    train.to_csv('data/feature_extraction/training_set_VU_DM_2014.csv')
else:
    train.to_csv('data/feature_extraction/test_set_VU_DM_2014.csv')
