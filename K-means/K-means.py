from random import random
from tkinter.ttk import Progressbar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

def findClosestCentroids(X, centroids):
    """
    X: dataset of size (m,n) m datapoints each having n features
    centroids: K means centroids (K,n)
    -------
    Returns
    idx : array_like
        A vector of size (m, ) which holds the centroids assignment for each example (row) in the dataset X.
    
    """
    K = centroids.shape[0] 
    idx = np.zeros(X.shape[0]) #m sized vector
    m=X.shape[0]

    for i in range(m):
        d = np.zeros(K) # store distance from each of K-means of one datapoint
        for j in range(K):
            d[j] = np.dot(X[i]-centroids[j], X[i]-centroids[j])
        idx[i] = np.where(d==np.min(d))[0][0]

    return idx

def computeCentroids(X, idx, K):
    """
    Parameters
    ----------
    X : dataset of size (m,n) m datapoints each having n features
    
    idx : array_like
        A vector of size (m, ) which holds the centroids assignment for each example (row) in the dataset X.
    
    Returns
    -------
    centroids : array_like
        A matrix of size (K, n) where each row is the mean of the data 
        points assigned to it.
    """
    m, n = X.shape
    centroids = np.zeros((K,n))

    for i in range(K):
        temp = np.where(idx==i)[0]
        for j in temp:
            centroids[i] += X[j]
        if(np.size(temp)!=0):
            centroids[i] = centroids[i]/np.size(temp)
        else:
            centroids[i] = X[np.random.randint(0,m)]

    return centroids

def runKmeans(X, centroids, max_iters):
    K = centroids.shape[0]
    idx = None

    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, K)
        print("progress",i)

    return centroids, idx

def randomInitCentroids(X, K):
    m,n =X.shape
    centroids = np.zeros((K,n))

    randidx =np.random.permutation(X.shape[0])
    centroids = X[randidx[:K],:]
    return centroids


# X = np.array([[0,0],[0,1],[1,0],[10,10],[10,9],[9,10]])
# K = 2
# centroids = randomInitCentroids(X,K)
# final_centroids,idx = runKmeans(X,centroids, 1000)
# print(final_centroids)
# print(idx)

K = 10
max_iters = 5

A = image.imread('bird_small.png') # shape 128x128x3
X = A.reshape(-1,4) # shape 16384x3

initial_centroids = randomInitCentroids(X, K)

final_centroids, idx =runKmeans(X, initial_centroids, max_iters)

compressed_X = np.zeros(X.shape)

for i in range(idx.shape[0]):
    compressed_X[i] = final_centroids[int(idx[i])]


A_compressed = compressed_X.reshape(A.shape)
plt.imsave('compressed.png', A_compressed)
