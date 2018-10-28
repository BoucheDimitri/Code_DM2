import pandas as pd
import matplotlib.pyplot as plt
import importlib
import os
import numpy as np

import em
import kmeans
import utils

# Plotting parameters
plt.rcParams.update({'font.size': 22})

# Reload module (for developpement)
importlib.reload(em)
importlib.reload(kmeans)
importlib.reload(utils)


# Load the data
path = os.getcwd() + "/data/"
data_train = pd.read_table(path + "EMGaussian.data", header=None, sep=" ")
data_test = pd.read_table(path + "EMGaussian.test", header=None, sep=" ")
x = data_train.values.T
xtest = data_test.values.T
xall = np.concatenate((x, xtest), axis=1)


# Run k-means
k = 4
mus, z = kmeans.iterate_kmeans(x, k, nits=100, epsilon=0.001)
# Plot clusters and centers
fig1, ax1 = plt.subplots()
utils.plot_clusters(x, mus, z, ax1)


# Compare several runs of k-means with different random initializations
centers, objectives = kmeans.compare_several_runs(x, k, nsims=100, nits=100, epsilon=0.001)
# Plot the different centers obtained
kmeans.plot_centroids(centers, k)
# Plot histogram of distorstion values
plt.hist(objectives)


# Run EM with covariance matrices proportional to the identity matrix
# Initialization with kmeans
mus_0, z = kmeans.iterate_kmeans(x, 4)
pi_0 = utils.cluster_repartition(z)
sigmas_0 = utils.clusters_cov(x, z)
# Iterate EM
pi_diag, mus_diag, sigmas_diag, qs_diag = em.iterate_em(x, pi_0, mus_0, sigmas_0, 100, 0.0001, diag=True)
# Predict labels using the parameters learned by EM
ztrain_diag = em.assign_cluster(x, pi_diag, mus_diag, sigmas_diag)
# Plot the results
fig2, axes2 = plt.subplots(1, 2)
utils.plot_clusters_ellipses(x, mus_diag, sigmas_diag, 1.5, ztrain_diag, axes2[0])
# Predict labels using the parameters learned by EM on all data (train + test)
zall_diag = em.assign_cluster(xall, pi_diag, mus_diag, sigmas_diag)
# Plot the results
utils.plot_clusters(xall, mus_diag, zall_diag, axes2[1])


# Run EM with general covariance matrices
pi, mus, sigmas, qs = em.iterate_em(x, pi_0, mus_0, sigmas_0, 100, 0.0001, diag=False)
# Predict labels using the parameters learned by EM
ztrain = em.assign_cluster(x, pi, mus, sigmas)
# Plot the results
fig3, axes3 = plt.subplots(1, 2)
utils.plot_clusters_ellipses(x, mus, sigmas, 1.5, ztrain, axes3[0])
# Predict labels using the parameters learned by EM on all data (train + test)
zall = em.assign_cluster(xall, pi, mus, sigmas)
# Plot the results
utils.plot_clusters(xall, mus, zall, axes3[1])
