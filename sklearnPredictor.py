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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

df = loadData()
y = df["New York"]  # let's try to predict the temps in New York!
y = pd.to_numeric(y)
y = y.fillna(value=-1)

print("Preprocessing data")

# we don't want to let it cheat by looking at the date:
df = df.drop(["datetime"], axis=1)
# and encode all as numeric:
df = df.apply(pd.to_numeric, errors="raise", downcast="float")
df = df.fillna(-1)

df = df.shift(periods=-24, fill_value=-1)

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.25, random_state=seed, shuffle=True
)
regr = RandomForestRegressor(max_depth=10, random_state=seed, n_estimators=25)

print("Fitting model")
regr.fit(X_train, y_train)

print("Predicting")

y_pred = regr.predict(X_test)

print("Scoring")

score = mean_absolute_error(y_test, y_pred)
r2Score = r2_score(y_test, y_pred)

print("Score", score)
print("r2Score", r2Score)
