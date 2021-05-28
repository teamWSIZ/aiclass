import torch as T
import torch.nn as NN
from torch import tensor

"""
Elementy występujące na openai / microscope, 
"""

dtype = T.float

data = [[[0, 0], [1, 1]], [[0, 0], [1, 1]]]
z = tensor(data, dtype=dtype)

sz = T.softmax(z, dim=0)
# dim=1 equalize in each element at level 1 [[[0.2689, 0.2689], [0.7311, 0.7311]],[[0.2689, 0.2689],[0.7311, 0.7311]]]
# dim=2 equalize in each at depth 2, [[[0.5000, 0.5000],[0.5000, 0.5000]],[[0.5000, 0.5000],[0.5000, 0.5000]]]


print(sz)
