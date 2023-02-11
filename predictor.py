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
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OrdinalEncoder
seed = 3

list_of_files = {}
for (dirpath, dirnames, filenames) in os.walk('hourly_weather'):
    for filename in filenames:
        if filename.endswith('.csv'):
            list_of_files[filename] = os.sep.join([dirpath, filename])

df = pd.read_csv('hourly_weather/temperature.csv')
#display(df)

y = df['New York'] # let's try to predict the temps in New York!
y = pd.to_numeric(y)
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
display(df)

'''
# let's do our splitting first, before we mess up our test set!
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=seed, shuffle=False)
'''

#print(df.shape)

df = df.shift(periods=-24, fill_value=-1)
dfs = np.array_split(df, (df.shape[0]//24)//7) # break into weeks
print(dfs)
