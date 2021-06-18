from typing import List


def encode_state(w: List[int]):
    x = 0
    for i in w:
        x <<= 1
        x |= i
    return x


if __name__ == '__main__':
    enc = encode_state([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    print(bin(enc), enc)
