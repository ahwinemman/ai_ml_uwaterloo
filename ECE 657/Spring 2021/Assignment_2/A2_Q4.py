import SimpSOM as sps
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom


# Training inputs for RGB colors. List of 24 colours
colors = [[0.863, 0.078, 0.235],
      [1, 0, 0],
      [1, 0.388, 0.278],
      [0.980, 0.502, 0.447],
      [1, 0.271, 0],
      [1, 1, 0],
      [0.604, 0.804, 0.196],
      [0.333, 0.419, 0.184], [0, 0.502, 0], [0.486, 0.988, 0], [0, 0.392, 0], [0, 1, 0.498], [0.275, 0.509, 0.706],
      [0.392, 0.584, 0.929], [0, 0.749, 1], [0.118, 0.565, 1], [0.529, 0.808, 0.922], [0, 0,0.804], [0, 0, 1],
      [1, 0.078, 0.576], [1, 0.412, 0.706], [1, 0.753, 0.796], [0, 0.502, 0.502], [1, 0.843, 0]]


# initialize a SOM network (100 x 100). Learning rate can be changed with the learning_rate parameter.
som = MiniSom(100, 100, 3, sigma=3.,
              learning_rate=10,
              neighborhood_function='gaussian')

# show the output of the untrained model
plt.imshow(abs(som.get_weights()), interpolation='none')
# the second parameter is the number of epochs. Can be changed here.
som.train(colors, 1000, random_order=True, verbose=True)
# show the output of the trained model
plt.imshow(abs(som.get_weights()), interpolation='none')



