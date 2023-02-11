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
from IPython.display import display
import os

list_of_files = {}
for (dirpath, dirnames, filenames) in os.walk('hourly_weather'):
    for filename in filenames:
        if filename.endswith('.csv'):
            list_of_files[filename] = os.sep.join([dirpath, filename])

df = pd.read_csv('hourly_weather/temperature.csv')
#display(df)

Y = df['New York'] # let's try to predict the temps in New York!

i=0
for k,v in list_of_files.items():
    if k=='temperature.csv':
        continue
    if k=='city_attributes.csv':
        continue
    i+=1
    df = df.merge(pd.read_csv(v), on='datetime', suffixes=(None,str(i)))

#display(df)
