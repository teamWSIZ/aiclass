from math import sin


def model_sinus(x):
    return 1 + sin(x)


def model_lorentz(x):
    return 1 / (1 + (x - 5) ** 2)
