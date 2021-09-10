from math import pi, sin
from random import random, randint
from typing import Tuple, List

from sequence_prediction.text_prediction.translate import encode_char, decode_char, encode_with_channels


def gen_samples(n_samples, history_len) -> Tuple[List, List]:
    """
    Tworzymy próbki składające się z historii jakiegoś tekstu.
    Zadanie sieci neuronowej to przewidzieć kolejną literę mając dany kawałek tekstu.

    Każda z próbek ma wymiary n_channels x history_len
        ~ obrazek 1D wielkości history_len, zrobiony z n_channels kolorow

    Całość ma wymiar n_samples x n_channels x history_len

    """
    samples = []
    outputs = []

    text = 'aaa bbb ccc ddd aaa ccc ddd aaa bbb ddd aaa ccc aaa ccc ddd aaa bbb ddd aaa ccc aaa ccc ddd aaa bbb ddd aaa ccc '

    for st in range(n_samples):
        end = randint(history_len, len(text)-1)

        sample_text = text[end - history_len:end]
        output_char = text[end]

        sample_channel_encoded = encode_with_channels(sample_text)
        output_encoded = encode_char(output_char)

        samples.append(sample_channel_encoded)
        outputs.append(output_encoded)

    return samples, outputs


if __name__ == '__main__':
    gg, oo = gen_samples(1, 4)
    print(gg)
