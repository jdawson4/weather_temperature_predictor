# Author: Jacob Dawson
#
# We're going to do the evaluation here on the off chance that our batch size
# is too large during training and it causes issues lol

import matplotlib.pyplot as plt
from constants import *
import random
#random.seed(seed)

trainX, trainy, testX, testy, _, _ = preprocess(loadData())

model = tf.keras.models.load_model("network")

'''print("VAL SET:")
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
)'''

# I really want to test the obvious thing: given some graphs of temperature
# and pressure, let's say, can our AI predict what tomorrow's temperature?
columnToShow = 27
for i in range(5):
    # choose 5 random spots:
    j = random.randint(0, len(testX)-1)
    thisX = testX[j]
    thisy = testy[j]

    predictedY = np.array(model(tf.expand_dims(thisX, axis=0)))[0]

    groundTruthTemps = thisX[:,columnToShow]
    groundTruthTemps = np.concatenate((groundTruthTemps, thisy[:,columnToShow]))

    predictedTemps = np.empty((thisX.shape[0],))
    predictedTemps[:] = np.nan
    predictedTemps = np.concatenate((predictedTemps, predictedY[:,columnToShow]))

    plt.plot(groundTruthTemps, color='red', label='Ground Truth')
    plt.plot(predictedTemps, color='blue', label='Predictions')
    plt.legend(loc="upper left")
    plt.show()
    #break
