# This code will solve the MNIST Kaggle competition, https://www.kaggle.com/c/digit-recognizer

import numpy as np

# We are going to use deep neural network with tensorflow.
import tensorflow as tf

# we are going to use pandas, http://pandas.pydata.org/pandas-docs/stable/
import pandas as pd

txt = pd.read_csv('train.csv', sep = ',', header = 0)

#print txt

v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print txt['0']
