import numpy as np
from scipy.spatial import distance


def assign(xs, mus):
    dists = distance.cdist(xs.T, mus.T)
    return np.argmax(dists, axis=1)