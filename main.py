import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os

import em
importlib.reload(em)
import kmeans
importlib.reload(kmeans)

path = os.getcwd() + "/data/"

data_train = pd.read_table(path + "EMGaussian.data", header=None, sep=" ")
x = data_train.values.T

mus, z = kmeans.iterate_kmeans(x, 4)
pi = kmeans.cluster_repartition(z)
sigmas = kmeans.clusters_cov(x, z)

pzgx = em.pz_given_x(x, pi, mus, sigmas)