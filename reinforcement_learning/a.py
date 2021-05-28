Q = 200
for i in range(50):
    Q *= 95 / 100
    print(f'{80-i:4}, {Q:4.3f}')
