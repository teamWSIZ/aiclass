from datetime import datetime
from random import randint
from time import sleep
from typing import List

N = 20  # ile mamy pól

FORWARD = 0
BACKWARD = 1


def print_q(q: List[List[float]], normalize=False):
    scale = 1
    if normalize:
        scale = max(max(q[0]), max(q[1]), 1)

    for row in q:
        for num in row:
            print(f'{num / scale:7.2f} ', end='')
        print()


def fixed_reward(state: int):
    if state == 0:
        return 2
    if state == N - 1:
        return 10
    return 0


def new_state(state, action):
    """
    :return: Nowy "stan" (tu: pozycja na osi) w zależności od decyzji.
    """
    if action == BACKWARD:
        return 0  # decyzja "w lewo" -- zawsze lądujemy na polu nr 0
    if action == FORWARD:
        if state < N - 1:
            return state + 1
        else:
            return N - 1  # odbijamy się od prawej ściany


def symbol(action):
    return '→' if action == 0 else '←'


LR = 0.1
LAMBDA = 0.9
EPOCHS = 10 ** 7


def get_random_action(state):
    r = randint(0, 10)
    if r > 3:
        action = 0  # →
    else:
        action = 1  # ←
    return action


def get_greedy_action(state, Q):
    # idziemy tam, gdzie Q jest "lepsze"
    reward_right = Q[0][state]
    reward_left = Q[1][state]

    if reward_left > reward_right:
        return 1
    else:
        return 0


if __name__ == '__main__':
    Q = [[0] * N for _ in range(2)]
    state = randint(0, N - 1)  # aktualna pozycja na planszy -- zawsze liczba między 0 i N-1

    st = datetime.now().timestamp()
    steps = 0

    for epoch in range(EPOCHS):
        x = randint(0, EPOCHS - 1)
        if x < epoch:
            action = get_greedy_action(state, Q)    # początkowo używaj losowych akcji
        else:
            action = get_random_action(state)   # później raczej kieruj się już znanymi "expected reward"

        # SYMULATOR
        n_state = new_state(state, action)  # z "symulatora"
        reward = fixed_reward(n_state)  # z "symulatora" + reward system

        # teraz mamy: state, action, new_state, reward (po akcji "action")

        # MAX to największa nagroda którą bot może osiągnąć po wyjściu z nowego stanu n_state, próbując wszystkie
        # możliwe akcje
        MAX = -100
        for action2 in [0, 1]:  # pętla po wszystkich możliwych akcjach z pozycji n_state
            # szukamy maksymalnej oczekiwanej nagrody w nowym stanie
            MAX = max(MAX, Q[action2][n_state])  # szukamy maksymalnej "spodziewanej" nagrody (nie tylko fixed_reward)

        dq = LR * (reward + LAMBDA * MAX - Q[action][state])
        Q[action][state] += dq

        if epoch % 100 == 0:
            state = randint(0, N - 1)  # losowanie nowej pozycji początkowej -- lepsza eksploracja

        if epoch % 100000 == 0:
            # print(f'oldstate:{state}, newstate={n_state} action:{symbol(action)},{action} reward:{reward}')
            print('---')
            print_q(Q, normalize=True)

        else:
            state = n_state
