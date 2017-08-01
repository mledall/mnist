# This code will solve the MNIST Kaggle competition, https://www.kaggle.com/c/digit-recognizer

import numpy as np

# We are going to use deep neural network with tensorflow.
import tensorflow as tf

# we are going to use pandas, http://pandas.pydata.org/pandas-docs/stable/
import pandas as pd


def load_train_data():
	file_path = 'train.csv'
	txt = pd.read_csv(file_path, sep = ',', header = 0)
	X = txt.values.copy()
	np.random.shuffle(X)	# randomize the input arrays
	labels = X[:,0]
	train = X[:,1:]		# Use a StandardScaler() on the pixel data? Is there any advantage?
	print 'training data input size: ', np.shape(train)
	print 'labels input size: ', np.shape(labels)
	return train, labels

load_train_data()


