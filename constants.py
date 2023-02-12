# Author: Jacob Dawson
#
# This file just collects a few constants and functions we use repeatedly in
# other scripts

import os
import pandas as pd

seed = 3

def loadData():
    print('Loading data')

    list_of_files = {}
    for (dirpath, dirnames, filenames) in os.walk('hourly_weather'):
        for filename in filenames:
            if filename.endswith('.csv'):
                list_of_files[filename] = os.sep.join([dirpath, filename])

    df = pd.read_csv('hourly_weather/temperature.csv')

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
    
    return df
