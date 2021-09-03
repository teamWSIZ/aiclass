from math import pi, sin
from random import random
from typing import Tuple, List

DX = 0.1


def gen_samples(n_samples, input_length) -> Tuple[List, List]:
    """
    Tworzymy próbki składające się z historii jakiejś funkcji (tu sin(x)), oraz z kolejnego punktu tej funkcji.
    Zadanie sieci neuronowej to przewidzieć kolejną wartość tej funkcji
    """
    samples = []
    outputs = []

    for st in range(n_samples):
        start = random() * pi * 2
        x = [1 + sin(start + i * DX) for i in range(input_length)]
        samples.append(x)
        outputs.append(1 + sin(start + input_length * DX))
    return samples, outputs
