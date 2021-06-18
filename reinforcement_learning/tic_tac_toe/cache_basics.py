from random import randint
from typing import List

from cachetools import cached, LRUCache


@cached(cache=LRUCache(maxsize=10))
def foo(x) -> int:
    print('uruchamiam foo')
    return x * x


@cached(cache=LRUCache(maxsize=10))
def goo(a, b, c) -> int:
    return a + b + c


for i in range(300):
    print(foo(randint(0, 10)))
    print(goo(*[1, 1, 8]))
