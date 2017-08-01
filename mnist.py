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

def tf_session():		# defines the tensorflow classifier
	train, labels = load_train_data()
	sess = tf.session()

	# placeholders can be used for the inputs and labels since they do not change.
	x = tf.placeholder(tf.float32, [None, 784])	# the number of input will vary, hence 'None', and the number of pixels len(train[0]) = 784 is fixed
	y_ = tf.placeholder(tf.float32, [None, 10])				# the number of labels is 10, 0-9, and the number of inputs will vary, None'

	# Our model consists of defining the weights and biases. Since these are going to be learnt, they need to be movable, hence we use variables.
	W = tf.variables(tf.zeros([784, 10]))	# weights is matrix with dimensions #pixels x #labels, all initialized to 0, there are 10 neurons
	b = tf.variables(tf.zeros([10]))		# there is one bias for each neuron
	y = tf.matmul(x,W)+b					# matmul is provided by tf for matrix multiplication

	# Evaluation, cross entropy for now, but can change that. This is for local evaluation, since kaggle uses their own.
	



train, labels = load_train_data()
print len(train)


