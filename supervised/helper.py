from random import randint
from typing import List, Tuple

import numpy as np
from torch import tensor, Tensor
import torch





"""
Wejście do sieci:
[0,0,0,1,1,1]   o długości "len"
wyjście:
1 jeśli występują tylko 2 jedynki i są oddalone o 2, czyli np. 
[0,0,0,1,1,1] --> 0
[0,1,0,1,0,0] --> 1
[1,0,1,0,0,0] --> 1
[1,0,1,0,0,1] --> 0

"""

"""
Konwersja na tensory:

t_sample = tensor(sample, dtype=torch.double, device='cpu')

Cięcie na batch-es:

b_sample = torch.split(t_sample, batch_size)

"""


def gen_random_trainset_1twice_distance_given(length, count, distance) -> Tuple[List, List]:
    """
    Generuje dwie listy (tensory):
    - próbek (wejść do sieci),
    - spodziewanych wyników.
    """
    samples = []
    outputs = []
    for _ in range(count):
        w = [randint(0, 1) for _ in range(length)]
        s = w.count(1)
        samples.append(w)
        if s == 2 and dist(w, elem=1) == distance:
            outputs.append(1)
        else:
            outputs.append(0)

    return samples, outputs


def gen_all_samples_1_distance_given(length, distance) -> Tuple[List, List]:
    """
    Generuje wszystkie listy długości length z dwoma 1 oddalonymi o distance.
    Note: Wynik będzie listą (length - distance) elementów (list).
    """
    samples = []
    outputs = []
    for st in range(length - distance):
        x = [0] * length
        x[st] = 1
        x[st + distance] = 1
        samples.append(x)
        outputs.append(1)
    return samples, outputs


def dist(lista, elem):
    # Znajduje odległość między pierwszym i ostatnim wystąpieniem "elem" w liście [0,0,1,0,0,0,1] --> 4
    n = len(lista)
    l = n
    r = 0
    for i in range(n):
        if lista[i] == elem:
            l = min(l, i)
            r = max(r, i)
    return r - l


def format_list(l: List[float]) -> str:
    res = '['
    for x in l:
        res += f' {x:5.3f}'
    res += ']'
    return res

def format_list_int(l: List[int]) -> str:
    return format_list([float(x) for x in l])
    # res = '['
    # for x in l:
    #     res += f' {x}'
    # res += ']'
    # return res
