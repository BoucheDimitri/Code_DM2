import pandas as pd
import matplotlib.pyplot as plt
import importlib
import os

import em
import kmeans
import utils

# Plotting parameters
plt.rcParams.update({'font.size': 50})

# Reload module (for developpement)
importlib.reload(em)
importlib.reload(kmeans)
importlib.reload(utils)

# Load the data
path = os.getcwd() + "/data/"
data_train = pd.read_table(path + "EMGaussian.data", header=None, sep=" ")
x = data_train.values.T

# Run k-means
mus, z = kmeans.iterate_kmeans(x, 4, nits=100, epsilon=0.001)
utils.plot_clusters(x, mus, z)

#






mus, z = kmeans.iterate_kmeans(x, 4)
pi = utils.cluster_repartition(z)
sigmas = utils.clusters_cov(x, z)
# sigmas = utils.clusters_cov_diag(x, z)
#
# pi_test, mus_test, sigmas_test = em.m_step(x, pi, mus, sigmas)
# pi_test, mus_test, sigmas_test = em.m_step(x, pi_test, mus_test, sigmas_test)
pi_test, mus_test, sigmas_test = em.m_step_diag(x, pi, mus, sigmas)
pi_test, mus_test, sigmas_test = em.m_step_diag(x, pi_test, mus_test, sigmas_test)

pi_test, mus_test, sigmas_test, qs = em.iterate_em(x, pi, mus, sigmas, 200, 0.00001, diag=False)

z = em.assign_cluster(x, pi_test, mus_test, sigmas_test)

utils.plot_clusters(x, mus_test, sigmas_test, 1.75, z)