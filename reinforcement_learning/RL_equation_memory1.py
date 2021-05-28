# wykonać wiele razy...
LR = 0.1
R = 3
Q = 10 ** 9  # startowa wartość "zmiennej Q"

for i in range(300):
    deltaQ = LR * (R - Q)
    Q += deltaQ
    print(f'{i:3} {Q:4.2f}')
