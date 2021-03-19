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



