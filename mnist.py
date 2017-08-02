# This code will solve the MNIST Kaggle competition, https://www.kaggle.com/c/digit-recognizer

import numpy as np

# We are going to use deep neural network with tensorflow.
import tensorflow as tf

# we are going to use pandas, http://pandas.pydata.org/pandas-docs/stable/
import pandas as pd

def cycle(arr):		# cycles the columns of a matrix
	temp = arr[:,[0]]
	arr[:,:-1] = arr[:,1:]
	arr[:,[-1]] = temp
	return arr


def load_train_data():
	file_path = 'train.csv'
	txt = pd.read_csv(file_path, sep = ',', header = 0)
	X = txt.values.copy()
#	X = cycle(X)			# puts the label as the last column
	np.random.shuffle(X)	# randomize the input arrays: first element is label, remaining is pixel data
#	train_label = X[:,0]
#	train_pixel = X[:,1:]		# Use a StandardScaler() on the pixel data? Is there any advantage?
	print '--> loaded training data'
	x = X[:,1:]
	y = X[:,0]
	return x,y

def load_test_data():
	file_path = 'test.csv'
	txt = pd.read_csv(file_path, sep = ',', header = 0)		# test data only consists of the pixel data. No id labels, will have to create our own.
	X = txt.values.copy()
	test = X		# Use a StandardScaler() on the pixel data? Is there any advantage?
	print 'test data input size: ', np.shape(test)
	print '--> loaded test data'
	return test


def NN_model():
	train_data = load_train_data()
	
	# placeholders can be used for the inputs and labels since they do not change.
	x = tf.placeholder(tf.float32, [None, 784])	# the number of input will vary, hence 'None', and the number of pixels len(train[0]) = 784 is fixed
	y_ = tf.placeholder(tf.float32, [None, 10])				# the number of labels is 10, 0-9, and the number of inputs will vary, None'

	# Our model consists of defining the weights and biases. Since these are going to be learnt, they need to be movable, hence we use variables.
	W = tf.Variable(tf.zeros([784, 10]))	# weights is matrix with dimensions #pixels x #labels, all initialized to 0, there are 10 neurons
	b = tf.Variable(tf.zeros([10]))		# there is one bias for each neuron
	y = tf.matmul(x,W)+b					# matmul is provided by tf for matrix multiplication. This is just one layer.

	# Evaluation, cross entropy for now, but can change that. This is for local evaluation, since kaggle uses their own.
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))

	# Training of the model
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
	train = optimizer.minimize(cross_entropy)

	sess = tf.InteractiveSession()	# The command InteractiveSession() allows to evaluate the model directly. If we used tf.Session() instead, we would have to explicitly open a session with the command with tf.Session(): ....  https://www.tensorflow.org/api_docs/python/tf/InteractiveSession
	optimizer, train = NN_model()
	sess.run(train, feed_dict = {x: train_data[0], y_: train_data[1]})

	# Evaluate the model
#	correct_prediction = tf.equal(tf.argmax(y,1 ), tf.argmax(y_,1 ))
#	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#	print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))	


NN_model()




