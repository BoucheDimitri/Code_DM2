import pandas as pd
import matplotlib.pyplot as plt
import os

path = os.getcwd() + "/data/"

data_train = pd.read_table(path + "EMGaussian.data", header=None, sep=" ")
x = data_train.values.T