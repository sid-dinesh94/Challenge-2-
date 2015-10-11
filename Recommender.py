import pandas as pd
import numpy as np
from annoy import AnnoyIndex
import operator
import sys
from multiprocessing.dummy import Pool as ThreadPool
from concurrent.futures import ThreadPoolExecutor
import time

class recommender:
	def __init__(matrix):
		self.matrix = matrix
		self.customerIDs = matrix.index
		self.merchantIDs = matrix.columns
		self.max_trees = max_trees
		self.num_neighbors = num_neighbors
		self.nearest = dict()
		self.recommendations = dict()

	def find_nearest(self):
		ann = AnnoyIndex(num_merchants)
		for customer in self.customers:
			customer_vector = list(matrix.loc[[customer]])
			ann.add_item(customer, customer_vector)
			if customer%200 == 0:
				print 'Adding '+ str(customer)
		print "Building"
		if len(self.merchantIDs) > max_trees:
			ann.build(max_trees)
		else:
			ann.build(len(self.merchantIDs))
		print "...done"
		for customer in self.customers:
			neighbors = ann.get_nns_by_item(customer, num_neighbors)
			if customer%200 == 0:
				print "Found neighbors for " + str(customer)
			self.nearest[customer] = []
			for neighbor in neighbors:
				if neighbor != customer:
					self.nearest[customer].append((neighbor, ann.get_distance(neighbor, customer)))

	def recommend_per_customer(self, customer):
		self.recommendations[customer] = []
		customer_recommendations = {}
		customer_vector = pd.DataFrame(self.matrix.ix[customer])
		not_transacted = customer_vector[customer_vector[customer]==-1].index
		totalDistance = 0.0
		for i in self.nearest[customer]:
			totalDistance += i[1]
		for i in self.nearest[customer]:
			weight = i[1] / totalDistance
			neighbor = i[0]
			neighbor_vector = self.matrix.ix[neighbor]
			neighbor_table = pd.DataFrame(self.matrix.ix[neighbor])
			transacted = neighbor_vector[neighbor_vector[neighbor]>0].index
			for merchant in not_transacted:
				if merchant in transacted:
					if merchant not in customer_recommendations:
						customer_recommendations[merchant] = (neighbor_vector[merchant]*weight)
					else:
						customer_recommendations[merchant] = (customer_recommendations[merchant]+neighbor_vector[merchant]*weight)
		#self.recommendations[customer] = list(customer_recommendations.items())
		#self.recommendations[customer].sort(key=lambda artistTuple: artistTuple[1],reverse = True)
		self.recommendations[customer] = [x[0] for x in sorted(customer_recommendations.items(), key=operator.itemgetter(1), reverse=True)[:10]]

	def recommend(self):
		self.find_nearest()
		for customer in self.customers:
			self.recommend(customer)
		return pd.from_dict(self.recommendations)

def make_matrix():
	start = time.clock()
	data = pd.read_csv("DataScienceChallenge_Data/DataScienceChallenge_Training.csv", sep = ",")		
	customers = data['Cust_map'].astype(int).unique()
	customers = customers[:1000]
	def map_function(customer):
		data_customer = data[data['Cust_map']==customer]
		df_customer = pd.DataFrame(columns = ['Customer', 'Merchants', 'nTransactions'])
		df_customer['Merchants'] = data_customer['Merch_Map_final'].unique()
		df_customer['nTransactions'] = list(data_customer.groupby('Merch_Map_final').sum()['NumTrans'])
		df_customer['Customer'] = customer
		return df_customer
	with ThreadPoolExecutor(max_workers = 10) as pool:
		per_customer = pool.map(map_function, customers)
	end = time.clock()
	print "Time: " ,(end-start) 
	new_data = pd.concat(per_customer)
	new_data = new_data.pivot(index='Customer', columns='Merchants', values='nTransactions')
	return new_data
	#pool = ThreadPool(10) 
	#per_customer = pool.map(map_function, customers)

def main():
	matrix = make_matrix()
	rec = recommender(matrix)
	solutions = rec.recommend()
	solutions.to_csv('solutions.csv', sep=',')

if __name__ == '__main__':
	main()

