from datetime import datetime
from random import randint
from time import sleep
from typing import List

N = 20  # ile mamy pól

FORWARD = 0
BACKWARD = 1
from random import choice, random

import torch
from torch import nn, optim
import torch.nn.functional as funct

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
from reinforcement_learning.tic_tac_toe.ttt_utils import apply_move, valid_moves, is_winning, print_state, random_state, \
    is_draw
from supervised.helper import *


class SimpleNet(nn.Module):
    def __init__(self, in_sz, hid, out_sz):
        super().__init__()
        self.hid = hid
        self.in_sz = in_sz
        self.flat1 = nn.Linear(in_sz, hid, True)
        self.flat2 = nn.Linear(hid, out_sz, True)

    def forward(self, x):
        """ Main function for evaluation of input """
        x = x.view(-1, self.in_sz)
        # print(x.size())  # batchsize x self.sz
        x = self.flat1(x)
        # print(x.size()) # batchsize x self.hid
        x = self.flat2(funct.relu(x))
        return funct.relu(x)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

    def save(self, filename):
        torch.save(self.state_dict(), filename)


dtype = torch.float
device = 'cpu'  # lub 'cuda'

EPOCHS = 10000
LR = 0.001
BATCH_SIZE = 500

# Net creation
net = SimpleNet(in_sz=N, hid=3, out_sz=2)   # 2 decyzje
# net = net.double()
# net.load('saves/one.dat')

if device == 'cuda':
    net = net.cuda()  # cała sieć kopiowana na GPU

# Training setup
loss_function = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)  # będzie na GPU, jeśli gpu=True


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




def convert_to_batches_tensors(samples: List[List[float]], outputs: List[List[float]], batch_size):
    # zamiana próbek na tensory (możliwa kopia do pamięci GPU)
    t_sample = tensor(samples, dtype=dtype, device=device)
    t_output = tensor(outputs, dtype=dtype, device=device)

    # przetasowanie całośći
    sample_count = t_sample.size()[0]
    per_torch = torch.randperm(sample_count)
    t_sample = t_sample[per_torch]
    t_output = t_output[per_torch]

    # "krojenie" próbek na "batches" (grupy próbek, krok optymalizacji po przeliczeniu całej grupy)
    b_sample = torch.split(t_sample, batch_size)
    b_output = torch.split(t_output, batch_size)
    return b_sample, b_output


def train_net(net, samples, outputs, n_epochs):
    b_samples, b_outputs = convert_to_batches_tensors(samples, outputs, BATCH_SIZE)

    for epoch in range(n_epochs):
        total_loss = 0
        for (batch_s, batch_o) in zip(b_samples, b_outputs):
            optimizer.zero_grad()
            prediction = net(batch_s)
            loss = loss_function(prediction.view(-1), batch_o.view(-1))

            total_loss += loss

            loss.backward()
            optimizer.step()
        if epoch % 25 == 0:
            print(f' epoch:{epoch}, loss:{total_loss:.6f}')
    pass





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
