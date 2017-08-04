# This code will solve the MNIST Kaggle competition, https://www.kaggle.com/c/digit-recognizer
# This code will implement a convolutional neural network to solve the MNIST competition
# The input and output data looks the same, only the model will have to be changed.
import sys
import numpy as np
import time as time


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

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def load_train_data(eval_r):	# eval_r gives the ratio of the training data is used for training, and 1-eval_r used for evaluation
	file_path = 'train.csv'
	txt = pd.read_csv(file_path, sep = ',', header = 0)
	X = txt.values.copy()
	np.random.shuffle(X)	# randomize the input arrays: first element is label, remaining is pixel data
	X_train = X[:,1:]
	X_label = X[:,0]
#	X = cycle(X)			# puts the label as the last column
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
	return X


def NN_model(eval_r):
	sess = tf.InteractiveSession()	# The command InteractiveSession() allows to evaluate the model directly. If we used tf.Session() instead, we would have to explicitly open a session with the command with tf.Session(): ....  https://www.tensorflow.org/api_docs/python/tf/InteractiveSession
	print '--> loaded training data'
	X, Y, X_eval, Y_eval = load_train_data(eval_r)
	X, Y, X_eval, Y_eval = X[:1000], Y[:1000], X_eval[:1000], Y_eval[:1000]	# Defines a subset of the training set for a quick evalutation of the network
	# placeholders can be used for the inputs and labels since they do not change.
	x = tf.placeholder(tf.float32, [None, 784])				# the number of input will vary, hence 'None', and the number of pixels len(train[0]) = 784 is fixed
	y_ = tf.placeholder(tf.float32, [None, 10])				# the number of labels is 10, 0-9

	def conv2d(x,W):	# convolutional layer indigenous to tensorflow
		return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

	def max_pool_2x2(x):
		return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

	W_conv1 = weight_variable([5,5,1,32])	# initialization of the weights of the first convolution layer
	b_conv1 = bias_variable([32])	# initialization of the biases

	x_image = tf.reshape(x, [-1,28,28,1])


# The following defines the model: number of layers, neurons, shaping and processing of the images.


	h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)	# This is the first convolution layer, with a ReLU output.
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)	# here you see the input of this layer is the output h_conv1 of the previous layer
	h_pool2 = max_pool_2x2(h_conv2)	# and then you pool again.

	# In the following step, we are going to fully connect the layer to train over all of the picture

	W_fc1 = weight_variable([7 * 7 * 64, 1024])		# I do not quite understand all of these functions, and dimensions, why 7*7, why 1024... 
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # same here, I do not quite know what all of those numbers mean.
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # Here we perform the training on the fully connected layer.

	keep_prob = tf.placeholder(tf.float32)	# we create a placeholder which will be used to define the probability that a given neuron will be used during training. We use dropout during training, but not during testing.
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)	# This is the part where we apply the dropout, which avoids overfitting.

	W_fc2 = weight_variable([1024, 10])	# This defines the readout layer
	b_fc2 = bias_variable([10])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv))

	# Training of the model
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.05)
	train_step = optimizer.minimize(cross_entropy)

	# Evaluation of the model, this is for local evaluation since Kaggle uses their own measure.
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Initialization of model
	init = tf.global_variables_initializer()
	sess.run(init)

	initial = time.time()

	print 'Training the network'
	train_step.run(feed_dict={x: X, y_: Y, keep_prob: 0.5})
	train_accuracy = accuracy.eval(feed_dict={x: X_eval, y_: Y_eval, keep_prob: 1.0})
	print 'accuracy of the model: %f' %train_accuracy
	print 'time to train: %fs' %(time.time()-initial)


	# Feeds the network with one test image
#	test_data = load_test_data()
#	feed_dict = {x: np.reshape(test_data[0],(1,784))}
#	classification = sess.run(y, feed_dict)
	
	# Feeds the network with test images to classify
#	print 'loading test data'
#	test_data = load_test_data()
#	feed_dict = {x: test_data}
#	print 'classifying test images'
#	classification = sess.run(y, feed_dict)
#	return classification

NN_model(0.8)

# The output vector is a 10D vector, whose entries are the "scores" that each neurons corresponding to the one_hot vector obtained. Thus some will be positive, some will be negative, and will also not be between 0-9. For instance, one result might look like [9886.63183594, -10975.38085938, 12687.75488281,-410.18963623,-3160.11547852,-7059.89794922,4049.5065918,-5421.63183594,3557.73974609,-3154.41796875] . We need to convert this back into a one_hot vector, and for that we take the largest positive value as 1, and all others as 0.

def one_hot_transf(arr):	# returns the digit associated to a vector of scores (the highest entry corresponds to the correct digit)
	index = np.where(arr == arr.max())[0][0]
	return index	# This is an integer


def submission_file(name = 'mnist_submission_file.csv'):
	classification = NN_model(0.99)
	class_array = np.zeros(len(classification))
	id_array = [0 for _ in range(len(classification))]
	with open(name, 'w') as f:
		f.write('ImageId,')
		f.write('Label')
		f.write('\n')
		print 'writing results in file...'
		for i in xrange(len(classification)):
			class_array[i] = one_hot_transf(classification[i])
			id_array[i] = i+1
			f.write('%d,' %id_array[i])
			f.write('%d' %class_array[i])
			f.write('\n')
	print("Wrote submission to file {}.".format(name))

#submission_file()

#print 'python version: %s' %str(sys.version[:5])
#print 'tensforflow version: %s' %str(tf.__version__)





