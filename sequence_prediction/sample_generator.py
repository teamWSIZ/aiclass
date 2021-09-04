from math import pi, sin
from random import random
from typing import Tuple, List


def gen_samples(n_samples, history_len, model_function, x_from, x_to, dx) -> Tuple[List, List]:
    """
    Tworzymy próbki składające się z historii jakiejś funkcji (tu sin(x)), oraz z kolejnego punktu tej funkcji.
    Zadanie sieci neuronowej to przewidzieć kolejną wartość tej funkcji
    """
    samples = []
    outputs = []

    for st in range(n_samples):
        start = x_from + random() * (x_to - x_from) # punkt startowy losowany z przedzialu [xfrom,xto]
        x = [model_function(start + i * dx) for i in range(history_len)]
        samples.append(x)
        outputs.append(model_function(start + history_len * dx))
    return samples, outputs
