from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import time
import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from keras.layers import Input
from build_model import ImageModel 
from load_data import ImageData, split_data
import pickle as pkl
from keras.utils import to_categorical
import scipy
from ml_loo import generate_ml_loo_features
from art.utils import load_cifar10


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset_name', type = str, 
		choices = ['cifar10'], 
		default = 'cifar10')

	parser.add_argument('--model_name', type = str, 
		choices = ['resnet'], 
		default = 'resnet') 

	parser.add_argument('--data_sample', type = str, 
		choices = ['x_train', 'x_val', 'x_val200'], 
		default = 'x_val200') 

	parser.add_argument(
		'--attack', 
		type = str, 
		choices = ['CW', 'FGSM', 'PGD'], 
		default = 'CW'
	)

	parser.add_argument(
		'--det', 
		type = str, 
		choices = ['ml_loo'], 
		default = 'ml_loo'
	)

	args = parser.parse_args()
	dict_a = vars(args) 
	data_model = args.dataset_name + args.model_name

	print('Loading dataset...') 
	dataset = ImageData(args.dataset_name)
	model = ImageModel(args.model_name, args.dataset_name, train = False, load = True)

	###########################################################
	# Loading original, adversarial and noisy samples
	###########################################################
	
	print('Loading original, adversarial and noisy samples...')
	assert os.path.exists('../GAT/cifar10/GAT-CIFAR10/AES/hamming_succ_0.9_{}_AE.npy'.format(args.attack)), 'test D^2 first'
	(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
	idxs = np.load('../GAT/cifar10/GAT-CIFAR10/AES/{}_AEs_idxs.npy'.format(args.attack))
	test_mask = np.load('../GAT/cifar10/GAT-CIFAR10/AES/hamming_succ_0.9_{}_AE.npy'.format(args.attack))
	train_mask = np.random.permutation(np.arange(len(test_mask))[1-test_mask])[:1000]

	X_adv = np.load('../GAT/cifar10/GAT-CIFAR10/AES/{}_AEs.npy'.format(args.attack))[chosen_idxs]
	X_test_adv = X_adv[test_mask]
	X_test = x_test[idxs[test_idxs]]
	X_train_adv = X_adv[train_mask]
	X_train = x_test[idxs[train_mask]]
	# X_test = np.load('{}/data/{}_{}_{}.npy'.format(data_model, args.data_sample, args.attack, 'ori'))

	# X_test_adv = np.load('{}/data/{}_adv_{}_{}.npy'.format(data_model, args.data_sample, args.attack, 'ori'))

	# X_train = np.load('{}/data/{}_train_{}_{}.npy'.format(data_model, args.data_sample, args.attack, 'ori'))

	# X_train_adv = np.load('{}/data/{}_train_adv_{}_{}.npy'.format(data_model, args.data_sample, args.attack, 'ori'))

	Y_test = model.predict(X_test)
	print("X_test_adv: ", X_test_adv.shape)

	
	x = {
		'train': {
			'original': X_train, 
			'adv': X_train_adv, 
			},
		'test': {
			'original': X_test, 
			'adv': X_test_adv, 
			},
	}
	#################################################################
	# Extracting features for original, adversarial and noisy samples
	#################################################################
	cat = {'original':'ori', 'adv':'adv', 'noisy':'noisy'}
	dt = {'train':'train', 'test':'test'}

	if args.det in ['ml_loo']:
		if args.model_name == 'resnet':
			interested_layers = [14,24,35,45,56,67,70]

		print('extracting layers ', interested_layers)
		reference = - dataset.x_train_mean

		combined_features = generate_ml_loo_features(args, data_model, reference, model, x, interested_layers)

		for data_type in ['test', 'train']:
			for category in ['original', 'adv']:
				np.save('{}/data/{}_{}_{}_{}_{}.npy'.format(
					data_model,
					args.data_sample,
					dt[data_type],
					cat[category],
					args.attack, 
					args.det), 
					combined_features[data_type][category])

	
