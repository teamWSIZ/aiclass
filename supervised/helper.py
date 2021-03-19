from random import randint
from typing import List

import numpy as np
from torch import tensor
import torch

tensor_type = torch.float


def generate_2x1_anywhere(sz, n_success, n_failure, batch_size):
    in_ = []
    ou_ = []
    total = n_success + n_failure
    while len(in_) < total:
        w = [randint(0, 1) for _ in range(sz)]
        s = np.array(w).sum()
        if s == 2 and n_success > 0:
            in_.append(w)
            ou_.append(1)
            n_success -= 1
        if s != 2 and n_failure > 0:
            in_.append(w)
            ou_.append(0)
            n_failure -= 1
    sample_in, sample_ou = tensor(in_, dtype=tensor_type), tensor(ou_, dtype=tensor_type)
    batches_in = torch.split(sample_in, batch_size)  # todo: comment
    batches_ou = torch.split(sample_ou, batch_size)
    return batches_in, batches_ou


def generate_2x1_dist2(len, count, need_1, batch_size, gpu=False):
    in_ = []
    ou_ = []
    while count > 0:
        w = [randint(0, 1) for _ in range(len)]
        s = w.count(1)

        if s == 2 and dist(w, 1) == 2:
            in_.append(w)
            ou_.append(1)
            need_1 -= 1
            count -= 1
        elif need_1 <= 0:
            in_.append(w)
            ou_.append(0)
            count -= 1

    device = 'cpu' if not gpu else 'cuda'
    sample_in, sample_ou = tensor(in_, dtype=tensor_type, device=device), tensor(ou_, dtype=tensor_type, device=device)
    batches_in = torch.split(sample_in, batch_size)
    batches_ou = torch.split(sample_ou, batch_size)
    # print('**', batches_in)
    return batches_in, batches_ou


def dist(list, elem):
    # Znajduje odległość między pierwszym i ostatnim wystąpieniem "elem" w liście
    n = len(list)
    l = n
    r = 0
    for i in range(n):
        if list[i] == elem:
            l = min(l, i)
            r = max(r, i)
    return r - l


def format_list(l: List[float]) -> str:
    res = '['
    for x in l:
        res += f' {x:5.3f}'
    res += ']'
    return res
