import random

import torch
from torch import nn, tensor

dtype = torch.float

# TORCH: clamp
# t = tensor([-2, -1, 0, 1, 2, 3], dtype=dtype)
# print(t.clamp(-1, 1))  # tensor([-1., -1.,  0.,  1.,  1.,  1.])

# TORCH: loss functions
# mse = nn.MSELoss()
# l1smooth = nn.SmoothL1Loss()
# out = tensor([0, 0, 0], dtype=dtype)
# pre = tensor([0, 10, 0], dtype=dtype)
# print('mse     ', mse(out, pre))
# print('l1smooth', l1smooth(out, pre))  # less sensitive to very large loss values..

# PYTHON: random.sample
# w = [0, 1, 2, 3, 4, 5]
# print(random.sample(w, 3))  # 3 elems from `w`... randomly chosen

t = tensor([2, 1, 5, 8, 10])
d = tensor([0, 0, 4])
torch.gather(t, 0, d)
print(d)
