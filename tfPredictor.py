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

# import numpy as np
from constants import *

# from IPython.display import display
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import maxabs_scale

df = loadData()
print("Preprocessing data")

# we don't want to let it cheat by looking at the date:
df = df.drop(["datetime"], axis=1)
# and encode all as numeric:
df = df.apply(pd.to_numeric, errors="raise", downcast="float")
df = df.fillna(-1)

y = df.shift(periods=24, fill_value=-1) # we try to predict 24 hours ahead
#y = np.array(y["New York"])

scaler = StandardScaler().fit(df)
df = maxabs_scale(scaler.transform(df))
y = maxabs_scale(scaler.transform(y))

X, y = chunk(df, y)
trainX = X[:int(len(X)*0.75)]
testX = X[int(len(X)*0.75):]

trainy = y[:int(len(y)*0.75)]
testy = y[int(len(y)*0.75):]

#print(trainX)
#print(trainy)
#print(testX)
#print(testy)

print("X contains nans:",(np.isnan(trainX).any()))
print("y contains nans:",(np.isnan(trainy).any()))


print('Making model')
model = lstmArchitecture()
model.summary()

class EveryKCallback(keras.callbacks.Callback):
    def __init__(self,epoch_interval=epoch_interval):
        self.epoch_interval = epoch_interval
    def on_epoch_begin(self,epoch,logs=None):
        if ((epoch % self.epoch_interval)==0):
            self.model.save_weights("ckpts/ckpt"+str(epoch), overwrite=True, save_format='h5')
            #self.model.save('network',overwrite=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learnRate),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError(), 'accuracy'],
    #run_eagerly=True,
)

print('Training model')
history = model.fit(
    x=trainX,
    y=trainy,
    batch_size=batch_size,
    epochs=10,
    callbacks=[EveryKCallback()],
    validation_data=(testX, testy),
    shuffle=True,
)

# when we're ready to make some predictions, this code will be waiting for us:
"""
score = mean_absolute_error(y_test, y_pred)
r2Score = r2_score(y_test, y_pred)

print('Score', score)
print('r2Score', r2Score)
"""
