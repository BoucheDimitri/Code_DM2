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


def iterate_kmeans(xs, k, nits):
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
    for i in range(0, nits):
        mus = update_mus(xs, z)
        z = assign_xs(xs, mus)
        print(i)
    return mus, z


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


def plot_clusters(xs, z):
    """
    Plot data in different colors according to their cluster assignement

    Params:
        xs (np.ndarray): the data matrix (nfeatures, nsamples)
        z (np.ndarray): vector of assignement to clusters (nsamples, )

    Returns:
        nonetype: None
    """
    xspd = clustered_table(xs, z)
    k = np.unique(z).shape[0]
    fig, ax = plt.subplots()
    for i in range(0, k):
        ax.scatter(xspd[xspd.c == i].x0, xspd[xspd.c == i].x1)
