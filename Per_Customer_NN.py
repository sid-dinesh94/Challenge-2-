import Recsys
import ExtractingBitTags
import NeuralNetwork
from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import csv
import time 


ratings_filename = 'user_item_ratings_user.csv'

#dense user_item_matrix is saved in this file and used as input to Recsys.
user_item_matrix_full_filename = 'test.csv'

#Scoring can be 0-6
nb_classes = 7

user_item_matrix, userIDs, itemIDs = Recsys.make_matrix(ratings_filename, want_matrix = True, only_bits = True)

bits_global, bitIDs = ExtractingBitTags.make_bit_tags(make_csv = False)

bitTags =[]
for bit in bits_global:
	bitTags.append(bits_global[int(bit)])

#Makes a neural network for a particular user, predicts ratings for unrated bits and returns the dense user_bit_vector.
def per_user(user_item_vector):
	bits = bits_global
	ratings = []
	bitTagsRated = []
	bitTagsUnrated = []
	i=0
	for bit in itemIDs:
		if user_item_vector[i] != -1:
			bitTagsRated.append(bits[int(bit)])
			ratings.append(user_item_vector[i])
		else:
			bitTagsUnrated.append(bits[int(bit)])
		i+=1
	vectorizer_copy = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None)
	X_train = vectorizer_copy.fit_transform(bitTagsRated)
	X_test = vectorizer_copy.transform(bitTagsUnrated)
	X_test = X_test.toarray()
	X_train = X_train.toarray()
	num_features = X_train.shape[1]
	y_train = np.asarray(ratings)
	num_rated = len(ratings)
	num_unrated = len(bitTagsUnrated)

	#Uncomment the next few lines to check accuracy
	'''#Validity set for accuracy calculations
	split = 5*(num_rated/6)
	X = X_train
	X_val = X[split:]
	X_train = X[:split]
	y = y_train
	y_val = y[split:]
	y_train = y[:split]
	'''

	predictions = NeuralNetwork.Neural_Network(X_train, y_train, X_test, nb_classes, num_features, X_val = None, y_val = None)
	rowdata = []
	j=0
	for i in range(len(itemIDs)):
		if user_item_vector[i] == -1:
			rowdata.append(predictions[j])
			j+=1
		else:
			rowdata.append(user_item_vector[i])
	return rowdata

#Uncomment the next few lines if the code is leaving zombie processes after completion.
#Add <import psutil> on top if using this
'''
def kill_proc_tree(pid, including_parent=True):
	parent = psutil.Process(pid)
	for child in parent.get_children(recursive=True):
		child.kill()
	if including_parent:
		parent.kill()
'''

def make_results():
	print "Making dense user-item matrix"
	start = time.clock()

	'''
	#Without multiprocessing
	with open(user_item_matrix_full_filename, 'w') as f:
		writer = csv.writer(f, delimiter = ',')
		rowheader = ['\\']
		rowheader = rowheader + itemIDs
		writer.writerow(rowheader)
		for i in range(len(userIDs)):
				if i%100==0:
					print 'On user ', i
				rowdata = per_user(i, user_item_matrix, bits, userIDs, itemIDs, vectorizer, num_features, train_fn, compute_prediction)
				writer.writerow(rowdata)
	'''
	#With multiprocessing
	with open(user_item_matrix_full_filename, 'w') as f:
		writer = csv.writer(f, delimiter = ',')
		rowheader = ['\\']
		rowheader = rowheader + itemIDs
		writer.writerow(rowheader)
		total = len(userIDs)
		print total
		rows = []
		for_acc =0
		for user in user_item_matrix:
			rows.append(user)
			#Uncomment the next few lines if testing accuracy
			'''
			if for_acc > 5:
				break
			for_acc+=1
			'''

		with ProcessPoolExecutor(max_workers = 10) as pool:
			futures = pool.map(per_user, rows)
			i =0
			for f in futures:
				if i%100 ==0:
					print 'On user ',i
				rowdata = [userIDs[i]]
				rowdata = rowdata + f
				i+=1
				writer.writerow(rowdata)
			pool.shutdown()
	
	end = time.clock()
	print "Time: " ,(end-start)
	print "...done"


def main():
	make_results()

if __name__ == '__main__':
	main()
