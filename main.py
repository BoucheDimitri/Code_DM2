import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

path = os.getcwd() + "/data/"

data_train = pd.read_table(path + "EMGaussian.data", header=None, sep=" ")
x = data_train.values.T


mus_t = np.array([[1, 2], [4, -1]])
pi_t = np.array([0.3, 0.7])
covs_t = []
covs_t.append(0.5*np.eye(2))
covs_t.append(np.eye(2))

mus_tplus1 = np.array([[0, 1], [2, -1]])
pi_tplus1 = np.array([0.5, 0.5])
covs_tplus1 = []
covs_tplus1.append(0.4*np.eye(2))
covs_tplus1.append(0.9*np.eye(2))


pzgx = pz_given_x(x, pi_t, mus_t, covs_t)