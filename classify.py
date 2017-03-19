# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import argparse
# display plots in this notebook

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
def run( inimage ):
	caffe_root = '/home/aishwarya/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
	sys.path.insert(0, caffe_root + 'python')

	import caffe		# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
	import os
	if os.path.isfile(caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'):
		print 'GoogleNet found.'
	else:
		print 'GoogleNet not found'
		#!../scripts/download_model_binary.py ../models/bvlc_reference_caffenet

	caffe.set_mode_cpu()

	model_def = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
	model_weights = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'

	net = caffe.Net(model_def,      # defines the structure of the model
		            model_weights,  # contains the trained weights
		            caffe.TEST)     # use test mode (e.g., don't perform dropout)

	# load the mean ImageNet image (as distributed with Caffe) for subtraction
	mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
	mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
	print 'mean-subtracted values:', zip('BGR', mu)

	# create transformer for the input called 'data'
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

	transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
	transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
	transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
	transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

	# load ImageNet labels
	labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
	#if not os.path.exists(labels_file):
		#get_ipython().system(u'../data/ilsvrc12/get_ilsvrc_aux.sh')
		
	labels = np.loadtxt(labels_file, str, delimiter='\t')

	#load image 
	#ap = argparse.ArgumentParser()
	#ap.add_argument("-i", "--image", required=True, help="path to base image")
	#args = ap.parse_args()
	image = caffe.io.load_image(inimage)
	

	net.blobs['data'].data[...] = transformer.preprocess('data', image)

	# perform classification
	net.forward()

	# obtain the output probabilities
	output_prob = net.blobs['prob'].data[0]

	# sort top five predictions from softmax output
	top_inds = output_prob.argsort()[::-1][:5]

	#print 'probabilities and labels:'
	#print zip(output_prob[top_inds], labels[top_inds])

	#print label with max probability
	#print labels[output_prob.argmax()]
	return labels[output_prob.argmax()]
