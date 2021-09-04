from math import sin


def evaluate(x, f):
    return f(x)


def f1(x):
    return sin(x)


def f2(x):
    return x * x

print(evaluate(1, f1))
print(evaluate(0, f1))
print(evaluate(2, f2))