import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt


def random_init(xs, k):
    """
    Affect the data vectors to a random cluster

    Params:
        xs (np.ndarray): the data matrix (nfeatures, nsamples)
        k (int): the number of clusters

    Returns:
        np.ndarray: random vector of assignement to clusters (nsamples, )
    """
    z = np.random.choice(np.array(range(0, k)), xs.shape[1])
    return z


def assign_xs(xs, mus):
    """
    Affect the data vectors to the cluster to which centroid they are the closest

    Params:
        xs (np.ndarray): the data matrix (nfeatures, nsamples)
        mus (np.ndarray): the centroids (nfeatures, nclusters)

    Returns:
        np.ndarray: vector of assignement to clusters (nsamples, )
    """
    dists = distance.cdist(xs.T, mus.T)
    z = np.argmin(dists, axis=1)
    return z


def update_mus(xs, z):
    """
    Centroids update for kmeans

    Params:
        xs (np.ndarray): the data matrix (nfeatures, nsamples)
        z (np.ndarray): vector of assignement to clusters (nsamples, )

    Returns:
        np.ndarray: vector of assignement to clusters (nsamples, )
    """
    k = np.unique(z).shape[0]
    mus = np.zeros((xs.shape[0], k))
    for i in range(0, k):
        inds = (z == i).astype(int)
        mus[:, i] = (1 / np.sum(inds)) * np.sum(xs * inds, axis=1)
    return mus


def iterate_kmeans(xs, k, nits=20, epsilon=0.1):
    """
    Iterate kmeans updates (1: centroid updates, 2: reassignement)

    Params:
        xs (np.ndarray): the data matrix (nfeatures, nsamples)
        k (int): n clusters
        nits (int): numbers of iterations to perform

    Returns:
        tuple: mus (centroid matrix), z (assignement vector)
    """
    z = random_init(xs, k)
    objs = [np.inf]
    for i in range(0, nits):
        mus = update_mus(xs, z)
        objs.append(objective_func(xs, mus, z))
        z = assign_xs(xs, mus)
        if np.abs(objs[i+1] - objs[i]) < epsilon:
            return mus, z
        # print("Objective value: " + str(objs[i+1]))
    return mus, z


def objective_func(xs, mus, z):
    dists = distance.cdist(xs.T, mus.T) ** 2
    k = np.unique(z).shape[0]
    obj = 0
    for i in range(0, k):
        inds = (z == i).astype(int)
        obj += np.sum(dists[:, i] * inds)
    return obj


def clustered_table(xs, z):
    """
    Dataset and cluster assignement in a pandas dataframe

    Params:
        xs (np.ndarray): the data matrix (nfeatures, nsamples)
        z (np.ndarray): vector of assignement to clusters (nsamples, )

    Returns:
        pd.core.frame.DataFrame
    """
    xspd = pd.DataFrame(data=xs.T, columns=["x0", "x1"])
    xspd["c"] = z
    return xspd


def plot_clusters(xs, mus, z):
    """
    Plot data in different colors according to their cluster assignement and the centroids

    Params:
        xs (np.ndarray): the data matrix (nfeatures, nsamples)
        mus (np.ndarray): the centroids (nfeatures, nclusters)
        z (np.ndarray): vector of assignement to clusters (nsamples, )

    Returns:
        nonetype: None
    """
    xspd = clustered_table(xs, z)
    k = np.unique(z).shape[0]
    fig, ax = plt.subplots()
    for i in range(0, k):
        ax.scatter(xspd[xspd.c == i].x0, xspd[xspd.c == i].x1)
        ax.scatter(mus[0, i], mus[1, i], c="k", marker="^", s=200)


def compare_centroids(xs, k, nsims, maxit=50, epsilon=0.1):
    mus_dict = {}
    for i in range(0, k):
        mus_dict[i] = np.zeros((xs.shape[0], nsims))
    for j in range(0, nsims):
        mus, z = iterate_kmeans(xs, k, maxit, epsilon)
        print(j)
        for i in range(0, k):
            mus_dict[i][:, j] = mus[:, i]
    return mus_dict


def plot_centroids(mus_dict, k):
    for i in range(0, k):
        plt.scatter(mus_dict[i][0, :], mus_dict[i][1, :])
