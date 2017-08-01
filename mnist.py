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
	print '--> loaded training data'
	return train, labels

def load_test_data():
	file_path = 'test.csv'
	txt = pd.read_csv(file_path, sep = ',', header = 0)		# test data only consists of the pixel data. No id labels, will have to create our own.
	X = txt.values.copy()
	test = X		# Use a StandardScaler() on the pixel data? Is there any advantage?
	print 'test data input size: ', np.shape(test)
	print '--> loaded test data'
	return test


load_test_data()


