# Author: Jacob Dawson
#
# In this file, we will build a model, train and predict, and output results.
# Maybe we'll do this with an sklearn learner, maybe we'll use a Keras deep
# learner, but our output will be things like: temperature, precipitation,
# and cloud cover. To make our task even easier, perhaps we will limit our
# predictions to just one local area--that is, given all kinds of weather data
# for the entire US, what is the temperature/precipitation/cloud cover/pressure
# in New York on a certain day?

import pandas as pd
import numpy as np
from IPython.display import display
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
#from sklearn.preprocessing import OrdinalEncoder
seed = 3

print('Loading data')

list_of_files = {}
for (dirpath, dirnames, filenames) in os.walk('hourly_weather'):
    for filename in filenames:
        if filename.endswith('.csv'):
            list_of_files[filename] = os.sep.join([dirpath, filename])

df = pd.read_csv('hourly_weather/temperature.csv')
#display(df)

y = df['New York'] # let's try to predict the temps in New York!
y = pd.to_numeric(y)
y = y.fillna(value=-1)
#print(y)

i=0
for k,v in list_of_files.items():
    if k=='temperature.csv':
        continue
    if k=='city_attributes.csv':
        continue

    # I want to encode this but it's become a headache. We'll circle back.
    if k=='weather_description.csv':
        continue
    '''# we need to handle strings and floats differently
    if k=='weather_description.csv':
        i+=1
        string_df = pd.read_csv(v)
        enc = OrdinalEncoder().set_params(encoded_missing_value=-1)
        df_without_datetime = string_df.drop(['datetime'], axis=1)
        df_without_datetime = enc.fit_transform(df_without_datetime)
        for column in df_without_datetime:
            string_df[column] = df_without_datetime[column]
        df = df.merge(string_df, on='datetime', suffixes=(None,str(i)))
    else:
        i+=1
        df = df.merge(pd.read_csv(v), on='datetime', suffixes=(None,str(i)))'''
    i+=1
    
    df = df.merge(pd.read_csv(v), on='datetime', suffixes=(None,str(i)))

print('Preprocessing data')

# the idea here is that we want an algorithm which can predict the temperature
# of new york at some hour in the future, given previous recorded values from
# all over the world. As such, we'll leave in the recorded values for new york,
# even though this seems a bit strange--a real weather station would have that
# data (obs) if they were making a prediction

# we don't want to let it cheat by looking at the date:
df = df.drop(['datetime'], axis=1)
# and encode all as numeric:
df = df.apply(pd.to_numeric, errors='raise', downcast='float')
#display(df)
df = df.fillna(-1)
#display(df)

'''
# let's do our splitting first, before we mess up our test set!
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=seed, shuffle=False)
'''

#print(df.shape)

df = df.shift(periods=-24, fill_value=-1)
df = df.iloc[0:(df.shape[0]//(7*24))*24*7]
dfs = np.array(np.array_split(df, (df.shape[0]//24)//7)) # break into weeks
dfs = np.reshape(dfs, (-1, dfs.shape[1] * dfs.shape[2])) # painful! I hate to do it!
y = y[0:(y.size//(7*24))*24*7]
y = np.reshape(np.array(y), (-1,7*24))

X_train, X_test, y_train, y_test = train_test_split(dfs, y[:,1], test_size=0.25, random_state=seed, shuffle=True)
regr = RandomForestRegressor(max_depth=10, random_state=seed, n_estimators=25)

print('Fitting model')
regr.fit(X_train, y_train)

print('Predicting')

y_pred = regr.predict(X_test)

print('Scoring')

score = mean_absolute_error(y_test, y_pred)
r2Score = r2_score(y_test, y_pred)

print('Score', score)
print('r2Score', r2Score)
