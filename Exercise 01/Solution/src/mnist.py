import tensorflow as tf
from optparse import OptionParser
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

# Command line options
parser = OptionParser()

parser.add_option("--imageWidth", action="store", type="int", dest="imageWidth", default=28, help="Image width for feeding into the network")
parser.add_option("--imageHeight", action="store", type="int", dest="imageHeight", default=28, help="Image height for feeding into the network")
parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=1, help="Number of channels in the image")

parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=10000, help="Batch size")
parser.add_option("--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=10, help="Training epochs")
parser.add_option("--learningRate", action="store", type="float", dest="learningRate", default=1e-3, help="Learning Rate")
parser.add_option("--numClasses", action="store", type="int", dest="numClasses", default=10, help="Number of classes")

parser.add_option("--numTrainingInstances", action="store", type="int", dest="numTrainingInstances", default=60000, help="Training instances")
parser.add_option("--numTestInstances", action="store", type="int", dest="numTestInstances", default=10000, help="Test instances")

# Parse command line options
(options, args) = parser.parse_args()
print (options)

# Import dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Create placeholder
inputPlaceholder = tf.placeholder(tf.float32, shape=[None, int(options.imageHeight * options.imageWidth * options.imageChannels)])
labelsPlaceholder = tf.placeholder(tf.int32, shape=[None, options.numClasses])

def createNetwork(inputPlaceholder):
	inputPlaceholder = tf.reshape(inputPlaceholder, [-1, options.imageHeight, options.imageWidth, options.imageChannels]) # Output shape: 28 x 28

	net = tf.layers.conv2d(inputs=inputPlaceholder, filters=64, kernel_size=(3, 3), strides=(2, 2), padding='VALID', activation=tf.nn.relu, name='conv1') # Output shape: 14 x 14
	net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=(2, 2), strides=(1, 1), padding='SAME', activation=tf.nn.relu, name='conv2') # Output shape: 14 x 14
	net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(1, 1), name='pool2') # Output shape: 7 x 7
	net = tf.layers.dropout(inputs=net, rate=0.35, name='pool2_drop')

	# FC-layer
	net = tf.contrib.layers.flatten(inputs=net)
	net = tf.layers.dense(inputs=net, units=256, activation=tf.nn.tanh, name='fc1')
	net = tf.layers.dropout(inputs=net, rate=0.5, name='fc1_drop')

	net = tf.layers.dense(inputs=net, units=options.numClasses, activation=None, name='logits')
	return net

with tf.name_scope('Model'):
	# Create the graph
	logits = createNetwork(inputPlaceholder)

with tf.name_scope('Loss'):
	# Add the logits to the loss function
	cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=labelsPlaceholder, logits=logits)

with tf.name_scope('Optimizer'):
	# Define Optimizer
	sgdTrainOp = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(cross_entropy_loss)
	adaDeltaTrainOp = tf.train.AdadeltaOptimizer(learning_rate=3e-1).minimize(cross_entropy_loss)
	adamTrainOp = tf.train.AdamOptimizer(learning_rate=options.learningRate).minimize(cross_entropy_loss)

initOp = tf.global_variables_initializer()

# GPU config
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

lossDict = {}
with tf.Session(config=config) as sess:
	# Iterate over the different optimizers
	for optimizer in ["SGD", "AdaDelta", "Adam"]:
		print ("Selected optimizer: %s" % optimizer)
		lossDict[optimizer] = {"train" : [], "test": []}
		sess.run(initOp) # Reinitialize the weights randomly

		for epoch in range(options.trainingEpochs):
			print ("Starting training for epoch # %d" % (epoch))
			numSteps = int(options.numTrainingInstances / options.batchSize)
			for step in range(numSteps):
				# Compute test loss first since the error reported after optimization will be lower than the train error
				lossTest = sess.run(cross_entropy_loss, feed_dict={inputPlaceholder: mnist.test.images, labelsPlaceholder: mnist.test.labels})

				batch = mnist.train.next_batch(options.batchSize)
				if optimizer == "SGD":
					loss, _ = sess.run([cross_entropy_loss, sgdTrainOp], feed_dict={inputPlaceholder: batch[0], labelsPlaceholder: batch[1]})
				elif optimizer == "AdaDelta":
					loss, _ = sess.run([cross_entropy_loss, adaDeltaTrainOp], feed_dict={inputPlaceholder: batch[0], labelsPlaceholder: batch[1]})
				elif optimizer == "Adam":
					loss, _ = sess.run([cross_entropy_loss, adamTrainOp], feed_dict={inputPlaceholder: batch[0], labelsPlaceholder: batch[1]})
				else:
					print ("Error: Optimizer not defined")
				
				print ("Step: %d | Train Loss: %f | Test Loss: %f" % (step, loss, lossTest))
				lossDict[optimizer]["train"].append(loss)
				lossDict[optimizer]["test"].append(lossTest)

# Plot the loss curves using Matplotlib
for optim in lossDict:
	mpl.style.use('seaborn')

	fig, ax = plt.subplots()
	ax.set_title('Optimizer: {!r}'.format(optim), color='C0')

	x = np.arange(0, len(lossDict[optim]["train"]))
	ax.plot(x, lossDict[optim]["train"], 'C0', label='Train', linewidth=2.0)
	ax.plot(x, lossDict[optim]["test"], 'C1', label='Test', linewidth=2.0)
	ax.legend()

	plt.savefig('./loss_curve_' + optim + ' .png', dpi=300)
	# plt.show()
	plt.close('all')