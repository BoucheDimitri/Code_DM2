import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt


def random_init(xs, k):
    z = np.random.choice(np.array(range(0, k)), xs.shape[1])
    return z


def assign_xs(xs, mus):
    dists = distance.cdist(xs.T, mus.T)
    z = np.argmin(dists, axis=1)
    return z


def update_mus(xs, z):
    k = np.unique(z).shape[0]
    mus = np.zeros((xs.shape[0], k))
    for i in range(0, k):
        inds = (z == i).astype(int)
        mus[:, i] = (1 / np.sum(inds)) * np.sum(xs * inds, axis=1)
    return mus


def iterate_kmeans(xs, k, nits):
    z = random_init(xs, k)
    for i in range(0, nits):
        mus = update_mus(xs, z)
        z = assign_xs(xs, mus)
        print(i)
    return mus, z


def clustered_table(xs, z):
    xspd = pd.DataFrame(data=xs.T, columns=["x0", "x1"])
    xspd["c"] = z
    return xspd


def plot_clusters(xs, z):
    xspd = clustered_table(xs, z)
    k = np.unique(z).shape[0]
    fig, ax = plt.subplots()
    for i in range(0, k):
        ax.scatter(xspd[xspd.c == i].x0, xspd[xspd.c == i].x1)
