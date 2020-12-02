
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
# Load image
import imageio


# Set random seed so output is all same
np.random.seed(1)


class KMeans(object):

    def __init__(self):  # No need to implement
        pass

    def pairwise_dist(self, x, y):  # [5 pts]
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between 
                x[i, :] and y[j, :]
                """
        # l2 norm, ord=2
        dist = np.linalg.norm(x[:, np.newaxis, :] - y, ord=2, axis=2)
        return dist
        # raise NotImplementedError

    def _init_centers(self, points, K, **kwargs):  # [5 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers.
        """
        # K random indices from points
        indices = np.random.choice(points.shape[0], size=K, replace=False)
        # form the numpy array with these indices
        centers = points[indices, :]
        return centers
        # raise NotImplementedError

    def _update_assignment(self, centers, points):  # [10 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            points: NxD numpy array, the observations
        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point

        Hint: You could call pairwise_dist() function.
        """
        # distances between points and all cluster centers
        distances = self.pairwise_dist(points, centers)
        # index of minimum distance for each row
        cluster_idx = np.argmin(distances, axis=1)
        return cluster_idx
        # raise NotImplementedError

    def _update_centers(self, old_centers, cluster_idx, points):  # [10 pts]
        """
        Args:
            old_centers: old centers KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            centers: new centers, K x D numpy array, where K is the number of clusters, and D is the dimension.
        """
        K, D = old_centers.shape[0], old_centers.shape[1]
        # intialize centers as zero array
        centers = np.zeros((K, D))
        for i in range(K):
            # find mean of all points having i as cluster idx
            centers[i] = np.mean(points[cluster_idx == i, :], axis=0)
        return centers
        # raise NotImplementedError

    def _get_loss(self, centers, cluster_idx, points):  # [5 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans. 
        """
        # find squared distances between all points and cluster centers
        distances = np.linalg.norm(points[:, np.newaxis, :] - centers, ord=2, axis=2) ** 2
        # select distance from cluster center
        distance_from_cluster_center = distances[np.arange(len(distances)), cluster_idx]
        # loss is sum of all these distances
        loss = np.sum(distance_from_cluster_center)
        return loss
        # raise NotImplementedError

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            verbose: boolean to set whether method should print loss (Hint: helpful for debugging)
            kwargs: any additional arguments you want
        Return:
            cluster assignments: Nx1 int numpy array
            cluster centers: K x D numpy array, the centers
            loss: final loss value of the objective function of KMeans
        """
        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        return cluster_idx, centers, loss

    def find_optimal_num_clusters(self, data, max_K=15):  # [10 pts]
        """Plots loss values for different number of clusters in K-Means

        Args:
            image: input image of shape(H, W, 3)
            max_K: number of clusters
        Return:
            None (plot loss values against number of clusters)
        """
        losses = []
        # for each value of K, find the loss and append to array
        for i in range(1, max_K + 1):
            cluster_idx, centers, loss = self.__call__(data, i)
            losses.append(loss)
        return losses

        # raise NotImplementedError


def intra_cluster_dist(cluster_idx, data, labels):  # [4 pts]
    """
    Calculates the average distance from a point to other points within the same cluster

    Args:
        cluster_idx: the cluster index (label) for which we want to find the intra cluster distance
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        intra_dist_cluster: 1D array where the i_th entry denotes the average distance from point i 
                            in cluster denoted by cluster_idx to other points within the same cluster
    """
    # indices of rows with cluster==cluster_idx
    indices = [i for i in range(len(labels)) if labels[i] == cluster_idx]
    # find rows
    rows_of_interest = data[indices]
    # initialize intra_dist_cluster
    intra_dist_cluster = np.zeros((rows_of_interest.shape[0]))

    for i in range(intra_dist_cluster.shape[0]):
        x = np.concatenate((rows_of_interest[:i, :], rows_of_interest[i + 1:, :]))
        y = rows_of_interest[i, :]
        intra_dist_cluster[i] = np.mean(np.sqrt(np.sum((x - y[None, :]) ** 2, axis=1)))

    return intra_dist_cluster
    # raise NotImplementedError


def inter_cluster_dist(cluster_idx, data, labels):  # [4 pts]
    """
    Calculates the average distance from one cluster to the nearest cluster
    Args:
        cluster_idx: the cluster index (label) for which we want to find the intra cluster distance
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        inter_dist_cluster: 1D array where the i-th entry denotes the average distance from point i in cluster
                            denoted by cluster_idx to the nearest neighboring cluster
    """
    # find number of distinct clusters
    num_clusters = len(np.unique(labels))
    # separate data points by cluster
    separate_data = []
    for j in range(num_clusters):
        indices = [i for i in range(len(labels)) if labels[i] == j]
        separate_data.append(data[indices])

    # indices of rows with cluster==cluster_idx
    indices = [i for i in range(len(labels)) if labels[i] == cluster_idx]
    # find rows
    rows_of_interest = data[indices]

    # initialize inter_dist_cluster
    inter_dist_cluster = np.zeros((rows_of_interest.shape[0]))

    # find min. average distance from each point to nearest cluster
    for i in range(rows_of_interest.shape[0]):
        y = rows_of_interest[i, :]
        distances = []
        for j in range(len(separate_data)):
            if j != cluster_idx:
                x = separate_data[j]
                distances.append(np.mean(np.sqrt(np.sum((x - y[None, :]) ** 2, axis=1))))
        inter_dist_cluster[i] = min(distances)
    return inter_dist_cluster
    # raise NotImplementedError


def silhouette_coefficient(data, labels):  # [2 pts]
    """
    Finds the silhouette coefficient of the current cluster assignment

    Args:
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        silhouette_coefficient: Silhouette coefficient of the current cluster assignment
    """
    # unique clusters
    clusters = np.unique(labels)
    # initialize silhouette coefficient
    SC = 0

    # for each value of cluster_idx, find inter and intra cluster distance, add SC(i) to SC
    for cluster_idx in clusters:
        intra_dist_cluster = intra_cluster_dist(cluster_idx, data, labels)
        inter_dist_cluster = inter_cluster_dist(cluster_idx, data, labels)
        # max of inter and intra cluster distance for each point
        max_of_both = np.max(np.array([intra_dist_cluster, inter_dist_cluster]), axis=0)
        # calculate SC of each point
        S = [x / y for x, y in zip(inter_dist_cluster - intra_dist_cluster, max_of_both)]
        # add sum of all SCs for that cluster to SC
        SC += np.sum(S)

    # SC is average of sum of all individual SCs
    SC /= data.shape[0]

    return SC
    # raise NotImplementedError
