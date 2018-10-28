import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 5


def cluster_repartition(z):
    nc = np.unique(z).shape[0]
    n = z.shape[0]
    pi = np.zeros((nc, ))
    for c in np.unique(z):
        pi[c] = z[z == c].shape[0] / n
    return pi


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


def clusters_cov(xs, z):
    xspd = clustered_table(xs, z)
    sigmas = []
    d = xs.shape[0]
    for c in np.unique(z):
        cvc = xspd[xspd.c == c].cov().values
        sigmas.append(cvc[:d, :d])
    return sigmas


def clusters_cov_diag(xs, z):
    xspd = clustered_table(xs, z)
    sigmas = []
    d = xs.shape[0]
    for c in np.unique(z):
        cvc = xspd[xspd.c == c].cov().values
        sigmas.append(np.eye(d) * np.mean(np.diag(cvc)))
    return sigmas


def plot_clusters_ellipses(xs, mus, sigmas, nsig, z):
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
    for j in range(0, k):
        ax.scatter(xspd[xspd.c == j].x0, xspd[xspd.c == j].x1)
        ax.scatter(mus[0, j], mus[1, j], c="k", marker="^", s=200)
        lamb, u = np.linalg.eig(sigmas[j])
        lamb = np.sqrt(lamb)
        ell = Ellipse(xy=(mus[0, j], mus[1, j]),
                      width=lamb[0] * nsig * 2,
                      height=lamb[1] * nsig * 2,
                      angle=np.rad2deg(np.arccos(u[0, 0])),
                      linewidth=0,
                      facecolor="none",
                      edgecolor='C' + str(j))
        print(ell.get_lw())
        ax.add_artist(ell)


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
    for j in range(0, k):
        ax.scatter(xspd[xspd.c == j].x0, xspd[xspd.c == j].x1)
        ax.scatter(mus[0, j], mus[1, j], c="k", marker="^", s=200)
