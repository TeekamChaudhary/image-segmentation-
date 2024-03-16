"""*****************************************************************************************
IIIT Delhi License
Copyright (c) 2023 Supratim Shit
*****************************************************************************************"""
import sys
sys.path.append("C:/Users/Ravi Ranjan Kumar/Downloads/wkpp.py")

import wkpp 

from sklearn.datasets import fetch_kddcup99
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

import numpy as np
import random


# Real data input
dataset = fetch_kddcup99()								# Fetch kddcup99 
data = dataset.data										# Load data
data = np.delete(data,[0,1,2,3],1) 						# Preprocess
data = data.astype(float)								# Preprocess
data = StandardScaler().fit_transform(data)				# Preprocess

n = np.size(data,0)										# Number of points in the dataset
d = np.size(data,1)										# Number of dimension/features in the dataset.
k = 17													# Number of clusters (say k = 17)
Sample_size = 100										# Desired coreset size (say m = 100)


def D2(data,k):											# D2-Sampling function.


	#----------Your code for Algo-1----------#
    def D2(data, k):
    # Initialize an empty list to store the selected centers
    centers = []
    
    # Randomly select the first center from the dataset
    center_index = random.randint(0, len(data) - 1)
    centers.append(data[center_index])
    
    # Calculate distances from the first center to all other points
    distances = cdist(data, [centers[0]])
    
    # Repeat the process until k centers are selected
    for _ in range(1, k):
        # Calculate the probability distribution based on the distances
        prob_dist = np.min(distances, axis=1) ** 2
        prob_dist /= np.sum(prob_dist)
        
        # Choose the next center with probability proportional to its distance squared
        center_index = np.random.choice(len(data), p=prob_dist)
        centers.append(data[center_index])
        
        # Update distances for the newly selected center
        new_distances = cdist(data, [centers[-1]])
        distances = np.minimum(distances, new_distances)
    
    return np.array(centers)

# Example usage:
# centers = D2(data, k)


									# Returns B from Algo-1.

centers = D2(data,k)									# Call D2-Sampling (D2())

def Sampling(data,k,centers,Sample_size):				# Coreset construction function.


	#----------Your code for Algo-2----------#
    def CoresetConstruction(data, k, centers, sample_size):
    # Initialize arrays to store coreset points and weights
    coreset = np.zeros((sample_size, data.shape[1]))
    weights = np.zeros(sample_size)
    
    # Calculate distances from each data point to the selected centers
    distances = cdist(data, centers)
    
    # Assign each data point to the nearest center and compute the weight
    nearest_center_indices = np.argmin(distances, axis=1)
    for i in range(k):
        points_indices = np.where(nearest_center_indices == i)[0]
        num_points = len(points_indices)
        if num_points > 0:
            # Sample a subset of points from the cluster proportional to their squared distances
            prob_dist = np.min(distances[points_indices], axis=1) ** 2
            prob_dist /= np.sum(prob_dist)
            sampled_indices = np.random.choice(points_indices, size=int(sample_size * num_points / len(data)), p=prob_dist)
            # Add sampled points to the coreset
            coreset[i * int(sample_size * num_points / len(data)):(i + 1) * int(sample_size * num_points / len(data))] = data[sampled_indices]
            # Compute weights based on cluster size
            weights[i * int(sample_size * num_points / len(data)):(i + 1) * int(sample_size * num_points / len(data))] = num_points / sample_size
    
    return coreset, weights

# Example usage:
# coreset, weights = CoresetConstruction(data, k, centers, Sample_size)



								# Return coreset points and its weights.

coreset, weight = Sampling(data,k,centers,Sample_size)	# Call coreset construction algorithm (Sampling())

#---Running KMean Clustering---#
fkmeans = KMeans(n_clusters=k,init='k-means++')
fkmeans.fit_predict(data)

#----Practical Coresets performance----# 	
Coreset_centers, _ = wkpp.kmeans_plusplus_w(coreset, k, w=weight, n_local_trials=100)						# Run weighted kMeans++ on coreset points
wt_kmeansclus = KMeans(n_clusters=k, init=Coreset_centers, max_iter=10).fit(coreset,sample_weight = weight)	# Run weighted KMeans on the coreset, using the inital centers from the above line.
Coreset_centers = wt_kmeansclus.cluster_centers_															# Compute cluster centers
coreset_cost = np.sum(np.min(cdist(data,Coreset_centers)**2,axis=1))										# Compute clustering cost from the above centers
reative_error_practicalCoreset = abs(coreset_cost - fkmeans.inertia_)/fkmeans.inertia_						# Computing relative error from practical coreset, here fkmeans.inertia_ is the optimal cost on the complete data.

#-----Uniform Sampling based Coreset-----#
tmp = np.random.choice(range(n),size=Sample_size,replace=False)		
sample = data[tmp][:]																						# Uniform sampling
sweight = n*np.ones(Sample_size)/Sample_size 																# Maintain appropriate weight
sweight = sweight/np.sum(sweight)																			# Normalize weight to define a distribution

#-----Uniform Samling based Coreset performance-----# 	
wt_kmeansclus = KMeans(n_clusters=k, init='k-means++', max_iter=10).fit(sample,sample_weight = sweight)		# Run KMeans on the random coreset
Uniform_centers = wt_kmeansclus.cluster_centers_															# Compute cluster centers
uniform_cost = np.sum(np.min(cdist(data,Uniform_centers)**2,axis=1))										# Compute clustering cost from the above centers
reative_error_unifromCoreset = abs(uniform_cost - fkmeans.inertia_)/fkmeans.inertia_						# Computing relative error from random coreset, here fkmeans.inertia_ is the optimal cost on the full data.
	

print("Relative error from Practical Coreset is",reative_error_practicalCoreset)
print("Relative error from Uniformly random Coreset is",reative_error_unifromCoreset)