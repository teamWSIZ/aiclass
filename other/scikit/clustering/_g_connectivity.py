from sklearn.neighbors import kneighbors_graph
import numpy as np

X = [[0], [3], [1]]

A = kneighbors_graph(X, 2, mode='connectivity', include_self=True)

print(A)
"""
  (0, 0)	1.0
  (0, 2)	1.0
  (1, 1)	1.0 ...
  """
print(type(A))  # <class 'scipy.sparse.csr.csr_matrix'>
print(A.toarray())
print(type(A.toarray()))  # <class 'numpy.ndarray'>

nX = np.array(X)
print(nX)  # column
print(nX.T)  # row
