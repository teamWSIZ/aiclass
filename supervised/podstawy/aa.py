import torch as T
from torch import tensor
from random import shuffle

type = T.double

z = tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=type, device=T.device('cpu'))
n = z.size()[0]

for t in z:
    t[0] = 0
    t[1] = 0

print(z)
