import numpy as np
import matplotlib.pyplot as plt
from utils import *
# Load an image of a bird
original_img = plt.imread('bird_small.png')

# Visualizing the image
plt.imshow(original_img)

print("Shape of original_img is:", original_img.shape)
# Divide by 255 so that all values are in the range 0 - 1 (not needed for PNG files)
# original_img = original_img / 255

# Reshape the image into an m x 3 matrix where m = number of pixels
# (in this case m = 128 x 128 = 16384)
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X_img that we will use K-Means on.

X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

def kMeans_init_centroids(X_img, K):

    #Select Randomly X_img values

    rand_idx = np.random.permutation(X_img.shape[0])
    centroids = X_img[rand_idx[:K]]

    return centroids
def compute_centroids(X, idx, K):

    m, n = X.shape
    centroids = np.zeros((K,n))

    for i in range(K):
        centroids[i] = sum(X[idx==i])/ len(X[idx==i])

    return centroids
def find_closest_centroids(X, centroids):
    K = centroids.shape[0]

    m, n = X.shape
    init_centroids = centroids
    idx = np.zeros(m, dtype=int)

    for i in range(m):
        distance = np.sum((X[i] - centroids)**2, axis = 1)
        idx [i] = np.argmin(distance)
    return idx

def run_kMeans(X_img, initial_centroids, max_iters):

    m,n = X_img.shape
    centroids = initial_centroids
    K = initial_centroids.shape[0]

    for i in range(max_iters):
        idx = find_closest_centroids(X_img, centroids)
        centroids = compute_centroids(X_img, idx, K)

    return centroids,idx

K = 16
max_iters = 10

init_centroids = kMeans_init_centroids(X_img, K)

centroids, idx = run_kMeans(X_img, init_centroids, max_iters)
plot_kMeans_RGB(X_img, centroids, idx, K)

show_centroid_colors(centroids)
idx = find_closest_centroids(X_img, centroids)
X_recovered = centroids[idx,:]
X_recovered = np.reshape(X_recovered, original_img.shape)