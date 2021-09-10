from random import random

from sklearn import cluster
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[1.1], [0.9], [2.1]])
X = np.array([[random() * 6] for _ in range(1000)])
# remove mean, set variance=1
XS = StandardScaler().fit_transform(X)  #computes mean&std and transforms data to (0,1)
print(XS.mean())  # ~0
print(XS.std())   # ~1

# print(XS)
# x = X[:, 0]
# print(x)  # numpy flat array, ~ [1,1,2]

# bw = cluster.estimate_bandwidth(X)
# print(bw)
