from typing import List

map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, ' ': 4}
reverse_map = {}
for (k, v) in map.items():
    reverse_map[v] = k


def encode_char(c) -> List[float]:
    out = [0] * 5
    out[map[c]] = 1
    return out


def decode_char(o: List[float]):
    """
    Ważne bo:
    ch0: 0.61
    ch1: 0.01
    ch2: 0.00
    ch3: 0.00
    ch4: 0.25
    """
    mx = max(o)
    for i, f in enumerate(o):
        if f == mx:
            return reverse_map[i]


def encode_with_channels(sample: str, n_channels=5):
    enc = [[] for _ in range(n_channels)]
    for c in sample:
        for i in range(n_channels):
            enc[i].append(1 if map[c] == i else 0)
    return enc


if __name__ == '__main__':
    x = encode_char('c')
    print(decode_char(x))
    print(encode_with_channels('aa bb ccd'))
