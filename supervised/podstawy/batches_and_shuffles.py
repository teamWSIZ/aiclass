import torch as T
from torch import tensor
from random import shuffle

type = T.double


def randperm(n):
    # losowa parmutacja liczb od [0,n) ; można też użyć T.randperm(n)
    p = [i for i in range(n)]
    shuffle(p)
    return p


z = tensor([[0, 1], [5, 2], [0, 3], [0, 4], [0, 5], [0, 6]], dtype=type, device=T.device('cpu'))

n = z.size()[0]

p = randperm(n)
z = z[p]

per_torch = T.randperm(n)
# print(per_torch.tolist()) # [0, 3, 5, 4, 1, 2]

print(z)
print(z[per_torch])