import numpy as np
# from numpy import ndarray

data = [[[0.5, 0.], [1.01, 1.], [0.5, 1]], [1, 1, 0]]
X, y = data  # X = [[0.5, 0], [1.01, 1], [0.5, 1]]
X = np.array(X)
# x1 = X[:, 0]
# print(x1)
print(X[:,0])   # get x-es from np.array