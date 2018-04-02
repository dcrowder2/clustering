# Dakota Crowder
# k-means clustering on Activity sensor data
# requires scipy and numpy

import random as rng
import numpy as np
from scipy.spatial import distance as dis

# used for the k = 3 labels
sedentary = [0, 1, 2, 3, 6, 7]
active = [4, 5, 8, 9, 10]
exercising = [11, 12, 13, 14, 15, 16, 17, 18]

# flag for cosine similarity
cosine = False


# picks k random points to be the initial centroids
# array is a 2d array, (instance, feature)
def random_centroids(array, k):
    return np.asarray(rng.sample(list(array), k))


# calculates a new centroid based on the mean of the cluster
def update_centroid(cluster, centroid):
    centroid_mean = []
    if np.size(cluster) is not 0:
        for i in range(len(cluster[0])):
            if i is not 0:
                centroid_mean.append((np.sum(cluster[:, i])) / np.size(cluster, 0))
    else:
        return centroid
    return centroid_mean


# Calculates the euclidean distance for a single x for each centroid
# x being a single instance 1-d, (feature)
# centroids being a 2d array, (centroid, features)
def calculate_euclidean_distance(x, centroids):
    return np.asarray([dis.euclidean(x, c) for c in centroids])


# Calculates the cosine distance for a single x for each centroid
# x being a single instance 1-d, (feature)
# centroids being a 2d array, (centroid, features)
def calculate_cosine_similarity(x, centroids):
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
# clusters is a 3d array, (label, instance, features)
def entropy(clusters):
    k = np.size(clusters, 0)
    sigma = np.sum([np.size(clusters[i], 0) for i in range(k)])
    m = k * sigma
    c_i_over_m = [len(c_i) / m for c_i in clusters]
    counts = np.zeros((k, k))  # will be a k^2 element array of the counts for each label in the
    #                       in the given cluster, for k = 3, 0 = sedentary
    #                       1 = active, 2 = exercising
    #                       (cluster, label_count)
    for i in range(k):
        for instance in clusters[i]:
            if k == 3:
                if instance[0] in sedentary:
                    counts[i, 0] += 1
                elif instance[0] in active:
                    counts[i, 1] += 1
                else:
                    counts[i, 2] += 1
            else:

                counts[i, int(instance[0])] += 1
    p_i_j = np.full((k, k), 1.)
    for cluster in range(k):
        for label in range(k):
            if np.size(clusters[cluster]) is not 0:
                p_i_j[cluster][label] = ((counts[cluster][label]) / float(len(clusters[cluster]))) + .1
    label_sum = np.sum(p_i_j * np.log2(p_i_j), axis=1)
    return np.sum(c_i_over_m * (-label_sum))


# takes in instances and centroids and uses the k-means to cluster
# returns a 3d array, (cluster, instance, feature)
# instances is a 2d array, (instance, feature)
# centroids is a 2d array, (cluster, feature)
def clustering(instances, centroids):
    k = np.size(centroids, 0)
    clusters = [[] for not_used in range(k)]
    if not cosine:
        for instance in instances:
            clusters[int(np.argmin(calculate_euclidean_distance(instance[1:], centroids)))].append(instance)
    else:
        for instance in instances:
            clusters[int(np.argmax(calculate_cosine_similarity(instance[1:], centroids)))].append(instance)
    return clusters


# takes in two centroid arrays, the old one and the new one, which will see if there is a big enough change to
# continue the clustering
def mean_change(old, new):
    sigma = abs(np.sum(abs(new) - abs(old)))
    if sigma < .01:  # Arbitrary number, if the old - new is small enough to be consider no change
        return False
    else:
        return True


def run(instances, k):
    centroids = random_centroids(instances[:, 1:], k)
    old_centroids = np.zeros((k, len(instances[0, 1:])))
    clusters = np.asarray(clustering(instances, centroids))
    print("Starting entropy: ")
    print(entropy(clusters))
    print("Starting k-means")
    while mean_change(old_centroids, centroids):
        old_centroids = centroids
        centroids = np.asarray([update_centroid(np.asarray(clusters[i]), centroids[i]) for i in range(np.size(centroids, 0))])
        clusters = np.asarray(clustering(instances, centroids))
        print("Entropy:")
        print(entropy(clusters))


if __name__ == '__main__':
    data = np.genfromtxt('formatted.csv', delimiter=',')
    if input("Use cosine similarity?(y/N)") == 'y':
        cosine = True
    run(data, int(input("How many Clusters?")))
