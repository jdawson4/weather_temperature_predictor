# Author: Jacob Dawson
#
# This file just collects a few constants and functions we use repeatedly in
# other scripts

import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np

seed = 3
chunked_length = 24 * 7 # the number of hours our model can see for prediction
batch_size = 128
epoch_interval = 2
learnRate = 0.00001 # adam's default is 0.001


def loadData():
    print("Loading data")

    list_of_files = {}
    for (dirpath, dirnames, filenames) in os.walk("hourly_weather"):
        for filename in filenames:
            if filename.endswith(".csv"):
                list_of_files[filename] = os.sep.join([dirpath, filename])

    df = pd.read_csv("hourly_weather/temperature.csv")

    i = 0
    for k, v in list_of_files.items():
        if k == "temperature.csv":
            continue
        if k == "city_attributes.csv":
            continue

        # I want to encode this but it's become a headache. We'll circle back.
        if k == "weather_description.csv":
            continue
        """# we need to handle strings and floats differently
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
            df = df.merge(pd.read_csv(v), on='datetime', suffixes=(None,str(i)))"""

        i += 1
        df = df.merge(pd.read_csv(v), on="datetime", suffixes=(None, str(i)))

    return df

def chunk(X, y):
    '''
    Given some X,y with the shape (timesteps, features), we return a chunked X
    with the shape (num_examples,chunked_length,features) and some y with the
    shape (num_examples,features)
    '''
    newX = []
    newy = []
    for i in range(0, len(X)-chunked_length, 1):
        newX.append(X[i:i+chunked_length,:])
        newy.append(y[i+chunked_length,:])
    newX = np.array(newX)
    newy = np.array(newy)
    return newX, newy

def lstmArchitecture():
    '''
    This will return a model using an LSTM-based architecture
    '''
    init = keras.initializers.RandomNormal(seed=seed)
    input = keras.layers.Input(shape=(chunked_length,180), dtype=tf.float16)

    output = keras.layers.LSTM(128, activation='selu', return_sequences=True, kernel_initializer=init)(input)
    #output = keras.layers.BatchNormalization()(output)
    #output = keras.layers.Dropout(0.25)(output)
    output = keras.layers.LSTM(64, activation='selu', return_sequences=True, kernel_initializer=init)(output)
    #output = keras.layers.BatchNormalization()(output)
    #output = keras.layers.Dropout(0.25)(output)
    output = keras.layers.LSTM(32, activation='selu', return_sequences=True, kernel_initializer=init)(output)
    #output = keras.layers.BatchNormalization()(output)
    #output = keras.layers.LSTM(180, activation=None, return_sequences=False, kernel_initializer=init)(output)
    output = keras.layers.Flatten()(output)
    output = keras.layers.Dense(180, activation=None)(output)

    return keras.Model(inputs=input, outputs=output, name='predictor')

if __name__=='__main__':
    #df = loadData()
    #print(df.shape)
    arch = lstmArchitecture()
    arch.summary()
