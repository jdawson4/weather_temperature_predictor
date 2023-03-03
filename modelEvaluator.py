# Author: Jacob Dawson
#
# We're going to do the evaluation here on the off chance that our batch size
# is too large during training and it causes issues lol

# import matplotlib.pyplot as plt
from constants import *

trainX, trainy, testX, testy = preprocess(loadData())

model = tf.keras.models.load_model("network")

print("VAL SET:")
model.evaluate(
    x=testX,
    y=testy,
    batch_size=batch_size // 2,
)

print("TRAIN SET:")
model.evaluate(
    x=trainX,
    y=trainy,
    batch_size=batch_size // 2,
)

# MORE CODE BENEATH HERE!
# I really want to test the obvious thing: given some graphs of temperature
# and pressure, let's say, can our AI predict what tomorrow's temperature?
