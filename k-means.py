# Dakota Crowder
# k-means clustering on Activity sensor data
# requires preprocess.py

import random as rng
import numpy as np
import math

# used for the k = 3 labels
sedentary = [1, 2, 3, 4, 7, 8]
active = [5, 6, 9, 10, 11]
exercising = [12, 13, 14, 15, 16, 17, 18, 19]
k_3_label = [sedentary, active, exercising]


# picks k random points to be the initial centroids
def random_centroids(array, k):
    return rng.sample(array, k)


# calculates a new centroid via the mean instance of the cluster points method
def update_centroids(clusters, centroids):
    centroid_mean = []
    for j in range(len(centroids)):
        feature_mean = []
        for i in range(len(centroids[0])):
            feature_mean.append((np.sum(clusters[:, i])) / len(centroids))
        centroid_mean.append(feature_mean)
    return centroid_mean


# Calculates the euclidean distance for a single x for each centroid
def calculate_euclidean_distance(x, centroids):
    distance = []
    x_minus_c = 0
    for i in range(len(x)):
        x_minus_c += x[i] - centroids[i]
    distance.append(math.sqrt(x_minus_c**2))
    return np.asarray(distance)

