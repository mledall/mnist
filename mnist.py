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
	x = tf.placeholder(tf.float32, [None, len(train[0])])	# the number of input will vary, hence 'None', and the number of pixels len(train[0]) = 784 is fixed
	y_ = tf.placeholder(tf.float32, [None, 10])				# the number of labels is 10, 0-9, and the number of inputs will vary, None'

	# Our model: simple one layer
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	y = tf.matmul(x,W)+b	# Note the order of x and W for the dimensions to match

	# Initialization
	init = tf.global_variables_initializer()
	sess.run(init)

	# Evaluation: cross entropy
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))

	# Training of the model
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
	train = optimizer.minimize(cross_entropy)



train, labels = load_train_data()
print len(train)


