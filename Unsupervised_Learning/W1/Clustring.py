import numpy as np
import matplotlib.pyplot as plt
from utils import *
### Implementation of K-Means ####
### K-means is an iterative procedure that starts by guessing the initial centroiod and then refined this by:
# repeatedly assigning examples to their closest centroids, and then recomputing the centroid based on the assignments


def find_closest_centroids(X, centroids):
    """
    Computes the centroids memberships for every example
    :param X: ndarray, (m,n) Input values
    :param centroids: ndarray, (K,n) centroids -> locations of all the values
    :return: array_like: (m,) closest centroids
    """
    K = centroids.shape[0]

    m, n  = X.shape
    idx = np.zeros(m,dtype=int)

    for i in range(m):
        distances = np.sum((X[i] - centroids)**2, axis= 1) #((X[i][0] - centroid[0][0])**2) + ((X[i][1] - centroid[
        # 0][1])**2)
        idx[i] = np.argmin(distances)  # Find the index of the centroid with the minimum distance

    return idx



def compute_centroids(X, idx, K):
    """
    Return the new centriods by computing the means of the data points assigned to each centroid.

    :param X: ndarray (m, n ) Data points
    :param idx: (m, ) Array containing index of the closes centroid for each example in X.
    Conceretely , idx[i] contains the index of the centriod closes to example i
    :param K: number of centroids
    :return: (K,n) New centroids computed
    """
    m,n = X.shape
    centroids = np.zeros((K,n))
    for i in range(K):
        centroids[i] = sum(X[idx==i]) / len(X[idx==i])

    return centroids

####Loading an example dataset that we will use it
X = load_data()
initial_centroids = np.array([[5,3], [6,3], [8,5], [4,4]])
idx = find_closest_centroids(X, initial_centroids)
K = 3
centroids = compute_centroids(X, idx, K)
print(centroids)
'''
 ######K-means on a sample dataset¶#######

After completed the two functions (find_closest_centroids and compute_centroids) above, 
the next step is to run the K-means algorithm on a toy 2D dataset to help  understand how K-means works
'''

def run_kMeans(X, initial_centroids, max_iters = 10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X is a single example
    :param X: ndarray (m, n) initial data set
    :param initial_centroids: (K,n) inital centroids for initialization
    :param max_iters: num of max iterations
    :param plot_progress: to see the progress
    :return: centroids, idx

    """

