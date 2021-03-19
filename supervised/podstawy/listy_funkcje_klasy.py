w = [1, 1, 0, 1]  # lista
print(w.count(1))  # 3
print(len(w))  # 4
w.append(2)
print(w)  # [1, 1, 0, 1, 2]
print(w[0])  # 1
w[0] = 10
print(w[0])  # 10

g = [[0, 1], [3, 4], [6, 7]]  # lista 2d
print(len(g))  # 3
print(g[0])  # [0,1]
print(g[0][1])  # 1

z = [[[0, 1], [3, 4], [6, 7]], [[7, 7], [7, 7], [8, 8]]]  # lista 3d
print(z[0])  # [[0, 1], [3, 4], [6, 7]]
print(z[0][2])  # [6,7]
print(z[0][2][0])  # 6

t = [[[[1]]]]  # tablica 4d ... [nr_obrazka][kolor][rząd][kolumna]


def add(a: float, b: float) -> float:
    # przykład funkcji
    return a + b


x = 'abc'
print(type(x))

print(add(2, 8))


class Car:
    fuel = 10
    name = 'Xiao'

    def __init__(self, fuel, name):
        self.fuel = fuel
        self.name = name

    def fill_up(self, final_fuel):
        self.fuel = final_fuel

    def drive(self, km):
        self.fuel -= 5 * km


mycar = Car(30, 'Zhong')
mycar.fill_up(final_fuel=50)
