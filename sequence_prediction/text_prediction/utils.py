def compare_lists(a, b, eps=0.05):
    n = len(a) * len(a[0])
    same = 0
    for batch_i in range(len(a)):
        for pos in range(len(a[batch_i])):
            if abs(a[batch_i][pos] - b[batch_i][pos]) <= eps:
                same += 1
    return same, n
