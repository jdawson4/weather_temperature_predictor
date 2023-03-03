# Author: Jacob Dawson
#
# In this file, we will build a model, train and predict, and output results.
# Maybe we'll do this with an sklearn learner, maybe we'll use a Keras deep
# learner, but our output will be things like: temperature, precipitation,
# and cloud cover. To make our task even easier, perhaps we will limit our
# predictions to just one local area--that is, given all kinds of weather data
# for the entire US, what is the temperature/precipitation/cloud cover/pressure
# in New York on a certain day?

# import pandas as pd

# import numpy as np
import matplotlib.pyplot as plt
from constants import *

# from IPython.display import display
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score

# TF_GPU_ALLOCATOR=cuda_malloc_async

# all our data handling goes here:
trainX, trainy, testX, testy = preprocess(loadData())

print("Making model")
# model = lstmArchitecture()
model = attentionArchitecture()
model.summary()


class EveryKCallback(keras.callbacks.Callback):
    def __init__(self, epoch_interval=epoch_interval):
        self.epoch_interval = epoch_interval

    def on_epoch_begin(self, epoch, logs=None):
        if (epoch % self.epoch_interval) == 0:
            # self.model.save_weights("ckpts/ckpt"+str(epoch), overwrite=True, save_format='h5')
            self.model.save("network", overwrite=True)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learnRate),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError(), "accuracy"],
)

print("Training model")
history = model.fit(
    x=trainX,
    y=trainy,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[EveryKCallback()],
    validation_data=(testX, testy),
    shuffle=True,
)

model.save("network", overwrite=True)

# found this code in a tutorial forever ago, might be cool to use here:
def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


visualize_loss(history, "Training and Validation Loss")
