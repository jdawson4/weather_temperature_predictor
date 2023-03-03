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
chunked_length = 24 * 3  # the number of hours our model can see for prediction
batch_size = 256
epoch_interval = 5
learnRate = 0.001  # adam's default is 0.001
epochs = 50
internal_layers = 128
dropout = 0.5


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
    """
    Given some X,y with the shape (timesteps, features), we return a chunked X
    with the shape (num_examples,chunked_length,features) and some y with the
    shape (num_examples,features)
    """
    newX = []
    newy = []
    for i in range(0, len(X) - chunked_length, 1):
        newX.append(X[i : i + chunked_length, :])
        newy.append(y[i + chunked_length, :])
        # newy.append(y[i+chunked_length])
    newX = np.array(newX)
    newy = np.array(newy)
    return newX, newy


def lstmArchitecture():
    """
    This will return a model using an LSTM-based architecture
    """
    init = keras.initializers.RandomNormal(seed=seed)
    input = keras.layers.Input(shape=(chunked_length, 180), dtype=tf.float16)
    output = keras.layers.Dropout(dropout)(input)
    output = keras.layers.LSTM(
        internal_layers,
        activation="selu",
        return_sequences=True,
        kernel_initializer=init,
    )(output)
    output = keras.layers.Dropout(dropout)(output)
    output = keras.layers.LSTM(
        internal_layers,
        activation="selu",
        return_sequences=True,
        kernel_initializer=init,
    )(output)
    output = keras.layers.LSTM(
        180, activation=None, return_sequences=False, kernel_initializer=init
    )(output)


def attnLayer(input, kernelInit, heads=4, kDim=64, out_shape=128):
    output = keras.layers.MultiHeadAttention(
        num_heads=heads,
        key_dim=kDim,
        output_shape=out_shape,
        kernel_initializer=kernelInit,
    )(input, input)
    output = keras.layers.LayerNormalization()(output)
    output = keras.layers.Dense(
        units=out_shape, activation=None, kernel_initializer=kernelInit
    )(output)
    output = keras.layers.Activation("selu")(output)
    output = keras.layers.Dropout(dropout)(output)
    return output


def attentionArchitecture():
    """
    This will return a model using an attention-based architecture
    """
    init = keras.initializers.RandomNormal(seed=seed)
    input = keras.layers.Input(shape=(chunked_length, 180), dtype=tf.float16)
    a1 = attnLayer(input=input, kernelInit=init)
    a2 = attnLayer(input=a1, kernelInit=init)
    a3 = attnLayer(input=keras.layers.Add()([a1, a2]), kernelInit=init)
    a4 = attnLayer(input=keras.layers.Add()([a1, a2, a3]), kernelInit=init)
    a5 = attnLayer(input=keras.layers.Add()([a1, a2, a3, a4]), kernelInit=init)
    a6 = attnLayer(input=keras.layers.Add()([a1, a2, a3, a4, a5]), kernelInit=init)
    a7 = attnLayer(input=keras.layers.Add()([a1, a2, a3, a4, a5, a6]), kernelInit=init)
    a8 = attnLayer(
        input=keras.layers.Add()([a1, a2, a3, a4, a5, a6, a7]), kernelInit=init
    )
    a9 = attnLayer(
        input=keras.layers.Add()([a1, a2, a3, a4, a5, a6, a7, a8]), kernelInit=init
    )
    a10 = attnLayer(
        input=keras.layers.Add()([a1, a2, a3, a4, a5, a6, a7, a8, a9]), kernelInit=init
    )
    output = keras.layers.GlobalAveragePooling1D()(a10)
    output = keras.layers.Dense(units=180, activation=None, kernel_initializer=init)(
        output
    )
    # output shape needs to be [None, 180] btw

    return keras.Model(inputs=input, outputs=output, name="predictor")


if __name__ == "__main__":
    # df = loadData()
    # print(df.shape)
    # arch = lstmArchitecture()
    arch = attentionArchitecture()
    arch.summary()
