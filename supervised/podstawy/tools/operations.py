import torch as T
import torch.nn as NN
from torch import tensor

dtype = T.double
dev = T.device('cpu')

data = [[0, 1], [5, 2], [0, 3], [0, 4], [0, 5], [0, 6]]
z = tensor(data, dtype=dtype)
print(z.size())

x = z.view(-1, 2, 3)
print(x)
print(x.size())


# print(T.max_pool1d(z, 2).tolist())  # [[1.0], [5.0], [3.0], [4.0], [5.0], [6.0]]
# print(T.sum(z).tolist())  # 26.0
# print(T.flatten(z).tolist())  # [0.0, 1.0, 5.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0]
#
# print('---------')
# # Indeksy dla convolution: [batch][channel][x][y][z]
# # channel ~ color;
# # batch index --> samples in current batch
# data2 = tensor([[[1, 2, 3, 0, 0, 1, 0, 0]]], dtype=dtype)
#
# g = T.max_pool1d(data2, kernel_size=2, stride=2, padding=1)
# print('pool',g.tolist())
#
# weight = tensor([[[1, 2, 0.1]]], dtype=dtype)
# print(T.conv1d(data2, weight, padding=1).tolist())  # [[[2.2, 5.3, 8.0, 3.0, 0.1, 2.0, 1.0, 0.0]]]
#
#
# print(T.avg_pool1d(data2, kernel_size=2, stride=2))
#
#
