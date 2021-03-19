import torch as T
from torch import tensor

z = tensor([[0, 1], [5, 2], [0, 3], [0, 4], [0, 5], [0, 6]],
           dtype=T.double, device=T.device('cpu'))

# print(z.size())
# print(z.tolist())

w = tensor([[[0,1,2],[3,4,5],[6,7,8]], [[0,0,0],[1,1,1],[2,2,2]]], device=T.device('cpu'), dtype=T.float)
print(w)
w_gpu = w.cuda()
print(w_gpu)
