# Dakota Crowder
# k-means clustering on Activity sensor data
# requires scipy and numpy

import random as rng
import numpy as np
from scipy.spatial import distance as dis

# used for the k = 3 labels
sedentary = [1, 2, 3, 4, 7, 8]
active = [5, 6, 9, 10, 11]
exercising = [12, 13, 14, 15, 16, 17, 18, 19]
k_3_label = [sedentary, active, exercising]


# picks k random points to be the initial centroids
# array is a 2d array, (instance, feature)
def random_centroids(array, k):
    return np.asarray(rng.sample(list(array), k))


# calculates a new centroid via the mean instance of the cluster points method
# clusters is a 3d array, (cluster, instance, feature)
# centroids is a 2d array, (cluster, features)
def update_centroids(clusters, centroids):
    centroid_mean = []
    for i in range(len(centroids)):
        feature_mean = []
        for j in range(len(centroids[0])):
            feature_mean.append((np.sum(clusters[i, :, j])) / len(clusters[i]))
        centroid_mean.append(feature_mean)
    return centroid_mean


# Calculates the euclidean distance for a single x for each centroid
# x being a single instance 1-d, (feature)
# centroids being a 2d array, (centroid, features)
def calculate_euclidean_distance(x, centroids):
    return np.asarray([dis.euclidean(x, c) for c in centroids])


# Calculates the cosine distance for a single x for each centroid
# x being a single instance 1-d, (feature)
# centroids being a 2d array, (centroid, features)
def calculate_cosine_distance(x, centroids):
    return np.asarray([(1 - (dis.cosine(x, c))) for c in centroids])


# Calculates the centroid coherence, clusters is a 3d array, (cluster, instance, feature)
# centroids being a 2d array, (centroid, features)
def centroid_coherence(clusters, centroids):
    k = len(centroids)
    coherence = 0
    for centroid in centroids:
        cluster_difference = 0
        for cluster in clusters:
            cluster_difference += np.sum((cluster - centroid) ** 2)
        coherence += cluster_difference
    return (1/k) * coherence


# Calculates the Entropy of labels
def entropy(labels, clusters):

