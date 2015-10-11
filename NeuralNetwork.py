import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import ExtractingBitTags
import csv
from theano import *
import theano.tensor as T
from keras.utils import np_utils 
import Recsys
import time
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

ratings_filename = 'user_item_ratings_user.csv'
user_item_matrix_full_filename = 'test.csv'
nb_classes = 7

#This function won't work with multiprocessing. If using this, add train_fn and comput_prediction as arguments for Neural_Network and delete all lines in Neural_Network till <alpha = 10.0>
def make_functions(num_features):
	W1_shape = (num_features/4, num_features)
	b1_shape = num_features/4
	W2_shape = (nb_classes, num_features/4)
	b2_shape = nb_classes
	
	W1 = shared(np.random.random(W1_shape) - 0.5, name = "W1")
	b1 = shared(np.random.random(b1_shape) - 0.5, name = "b1")
	W2 = shared(np.random.random(W2_shape) - 0.5, name = "W2")
	b2 = shared(np.random.random(b2_shape) - 0.5, name = "b2")


	x = T.dmatrix("x")
	labels = T.dmatrix("labels")

	hidden = T.nnet.sigmoid(x.dot(W1.transpose())+b1)
	output = T.nnet.softmax(hidden.dot(W2.transpose()) + b2)
	prediction = T.argmax(output, axis=1)

	reg_lambda = 0.0001
	regularization = reg_lambda * ((W1 * W1).sum() + (W2 * W2).sum() + (b1 * b1).sum() + (b2 * b2).sum())
	
	cost = T.nnet.binary_crossentropy(output, labels).mean() + regularization

	compute_prediction = function([x], prediction)

	alpha = T.dscalar("alpha")
	weights = [W1, W2, b1, b2]
	updates = [(w, w-alpha * grad(cost, w)) for w in weights]
	train_nn = function([x, labels, alpha],
	                    cost,
	                    updates = updates)
	return train_nn, compute_prediction

def accuracy(predicted, actual):
    total = 0.0
    correct = 0.0
    for p, a in zip(predicted, actual):
        total+=1
        if p==a:
            correct+=1
    return correct/total

def Neural_Network(X_train, y_train, X_test, nb_classes, num_features, X_val = None, y_val = None):
	W1_shape = (num_features/4, num_features)
	b1_shape = num_features/4
	W2_shape = (nb_classes, num_features/4)
	b2_shape = nb_classes
	
	W1 = shared(np.random.random(W1_shape) - 0.5, name = "W1")
	b1 = shared(np.random.random(b1_shape) - 0.5, name = "b1")
	W2 = shared(np.random.random(W2_shape) - 0.5, name = "W2")
	b2 = shared(np.random.random(b2_shape) - 0.5, name = "b2")
	
	x = T.dmatrix("x")
	labels = T.dmatrix("labels")

	hidden = T.nnet.sigmoid(x.dot(W1.transpose())+b1)
	output = T.nnet.softmax(hidden.dot(W2.transpose()) + b2)
	prediction = T.argmax(output, axis=1)

	#Can experiment with changing this to see if it has an effect on accuracy
	reg_lambda = 0.0001
	regularization = reg_lambda * ((W1 * W1).sum() + (W2 * W2).sum() + (b1 * b1).sum() + (b2 * b2).sum())
	cost = T.nnet.binary_crossentropy(output, labels).mean() + regularization

	compute_prediction = function([x], prediction)

	alpha = T.dscalar("alpha")
	weights = [W1, W2, b1, b2]
	updates = [(w, w-alpha * grad(cost, w)) for w in weights]
	train_fn = function([x, labels, alpha],
	                    cost,
	                    updates = updates)
	
	#Can experiment with changing this to see if it has an effect on accuracy
	alpha = 10.0

	costs = []
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	while True:
		costs.append(float(train_fn(X_train, Y_train, alpha)))
		if len(costs) > 2 and costs[-2] - costs[-1] < 0.0001:
			if alpha < 0.2:
				break
			else:
				alpha = alpha/1.5
	if X_val is not None:
		validation = compute_prediction(X_val)
		acc = accuracy(validation, y_val)
		print 'accuracy ', acc

	prediction = compute_prediction(X_test)
	return prediction


#Following functions are for logistic regression. Takes longer and is less accurate.
def make_functions_reg(num_features):
	W_shape = (nb_classes, num_features)
	b_shape = nb_classes
	W = shared(np.random.random(W_shape) - 0.5, name = "W")
	b = shared(np.random.random(b_shape) - 0.5, name="b")
	x = T.dmatrix("x")
	labels = T.dmatrix("labels")
	output = T.nnet.softmax(x.dot(W.transpose()) + b)
	prediction = T.argmax(output, axis = 1)
	cost = T.nnet.binary_crossentropy(output,labels).mean()
	compute_prediction = function([x], prediction)
	compute_cost = function([x, labels], cost)
	grad_W = grad(cost, W)
	grad_b = grad(cost, b)
	alpha = T.dscalar("alpha")
	updates = [(W, W-alpha*grad_W),(b, b-alpha*grad_b)]
	train = function([x, labels, alpha], cost, updates = updates)
	return train, compute_prediction

def Logistic_Regression(X_train, y_train, X_test, nb_classes, num_features, X_val = None, y_val = None):
	W_shape = (nb_classes, num_features)
	b_shape = nb_classes
	W = shared(np.random.random(W_shape) - 0.5, name = "W")
	b = shared(np.random.random(b_shape) - 0.5, name="b")
	x = T.dmatrix("x")
	labels = T.dmatrix("labels")
	output = T.nnet.softmax(x.dot(W.transpose()) + b)
	prediction = T.argmax(output, axis = 1)
	cost = T.nnet.binary_crossentropy(output,labels).mean()
	compute_prediction = function([x], prediction)
	compute_cost = function([x, labels], cost)
	grad_W = grad(cost, W)
	grad_b = grad(cost, b)
	alpha = T.dscalar("alpha")
	updates = [(W, W-alpha*grad_W),(b, b-alpha*grad_b)]
	train = function([x, labels, alpha], cost, updates = updates)

	alpha = 10.0
	costs =[]
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	while True:
		costs.append(float(train_fn(X_train, Y_train, alpha)))
		if len(costs)>2 and costs[-2] - costs[-1] < 0.0001:
			if alpha<0.2:
				break
			elif len(costs) > 10000:
				break
			else:
				alpha = alpha/1.5
		elif len(costs) > 10000:
				break
	if X_val is not None:
		validation = compute_prediction(X_val)
		acc = accuracy(validation, y_val)
		print 'accuracy ', acc
	prediction = compute_prediction(X_test)
	return prediction
