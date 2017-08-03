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

def one_hot(value):	# creates a one hot vector out of a number between 0-9
	arr = np.zeros(10)
	arr[value] = 1
	return arr

def one_hot_array(arr):	# creates an array of one hot vectors out of an array of numnbers
	new_array = np.zeros((len(arr),10))
	for i in range(len(arr)):
		new_array[i,:] = one_hot(arr[i])	
	return new_array


def load_train_data(eval_r):	# eval_r gives the ratio of the training data is used for training, and 1-eval_r used for evaluation
	file_path = 'train.csv'
	txt = pd.read_csv(file_path, sep = ',', header = 0)
	X_train = txt.values.copy()[:,1:]
	X_label = txt.values.copy()[:,0]
#	X = cycle(X)			# puts the label as the last column
	np.random.shuffle(X_train)	# randomize the input arrays: first element is label, remaining is pixel data
	print '--> loaded training data'
	L = int(eval_r*len(X_train))
	L_eval = int((1-eval_r) * len(X_train))
	x, y = X_train[:L], X_label[:L]
	x_eval, y_eval = X_train[L:], X_label[L:]
	y = one_hot_array(y)	# Turns labels into an array of one_hot vectors.
	y_eval = one_hot_array(y_eval)
	return x, y, x_eval, y_eval		# x_eval and y_eval are subsets of training data used for evaluation


def load_test_data():
	file_path = 'test.csv'
	txt = pd.read_csv(file_path, sep = ',', header = 0)		# test data only consists of the pixel data. No id labels, will have to create our own.
	X = txt.values.copy()
	test = X		# Use a StandardScaler() on the pixel data? Is there any advantage?
	print 'test data input size: ', np.shape(test)
	print '--> loaded test data'
	return test


def NN_model():
	sess = tf.InteractiveSession()	# The command InteractiveSession() allows to evaluate the model directly. If we used tf.Session() instead, we would have to explicitly open a session with the command with tf.Session(): ....  https://www.tensorflow.org/api_docs/python/tf/InteractiveSession
	eval_r = 0.8
	X, Y, X_eval, Y_eval = load_train_data(eval_r)
	
	# placeholders can be used for the inputs and labels since they do not change.
	x = tf.placeholder(tf.float32, [None, 784])	# the number of input will vary, hence 'None', and the number of pixels len(train[0]) = 784 is fixed
	y_ = tf.placeholder(tf.float32, [None, 10])				# the number of labels is 10, 0-9

	# Our model consists of defining the weights and biases. Since these are going to be learnt, they need to be movable, hence we use variables.
	W = tf.Variable(tf.zeros([784, 10]))	# weights is matrix with dimensions #pixels x #labels, all initialized to 0, there are 10 neurons
	b = tf.Variable(tf.zeros([10]))		# there is one bias for each neuron
	y = tf.matmul(x,W)+b					# matmul is provided by tf for matrix multiplication. This is just one layer.

	# Initialization
	init = tf.global_variables_initializer()
	sess.run(init)

	# Evaluation, cross entropy for now, but can change that. This is for local evaluation, since kaggle uses their own.
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))

	# Training of the model
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
	train = optimizer.minimize(cross_entropy)

	sess.run(train, feed_dict = {x: X, y_: Y})
	
#	batch_size = 100
#	count = 1
#	for i in xrange(0, len(X), batch_size):
#		print 'batch number: %d' % count
#		sess.run(train, feed_dict = {x: X[i:i+batch_size], y_: Y[i:i+batch_size]})
#		count = count +1

	# Evaluate the model
#	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#	print(sess.run(accuracy, feed_dict = {x: X_eval, y_: Y_eval}))

NN_model()


#v = [i for i in xrange(100)]
#batch_size = 4
#count = 1
#for i in xrange(0, len(v), batch_size):
#	print 'batch number %d' %count
#	print v[i:i+batch_size]
#	count = count + 1
#print len(v)/batch_size





