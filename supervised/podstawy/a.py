w = [1, 2, 1, 0, 0]

print(w.count(0))


def dist(list, elem):
    # Znajduje odległość między pierwszym i ostatnim wystąpieniem "elem" w liście
    n = len(list)
    l = n
    r = 0
    for i in range(n):
        if list[i] == elem:
            l = min(l, i)
            r = max(r, i)
    return r - l


print(dist([0, 1, 0, 0, 1], 1))
