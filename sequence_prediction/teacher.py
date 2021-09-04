from math import sin
from traceback import print_list

import torch
from torch import nn, optim
import torch.nn.functional as funct

from sequence_prediction.funkcje import model_sinus, model_lorentz
from sequence_prediction.sample_generator import gen_samples
from supervised.helper import *


class SequenceNet(nn.Module):
    """
        Simple NN: input(sz) ---> flat(hid) ---> 1
    """

    def __init__(self, sz, hid):
        super().__init__()
        self.hid = hid
        self.sz = sz
        self.flat1 = nn.Linear(sz, hid, True)
        self.flat2 = nn.Linear(hid, 1, True)

    def forward(self, x):
        """ Main function for evaluation of input """
        x = x.view(-1, self.sz)
        # print(x.size())  # batchsize x self.sz
        x = self.flat1(x)
        # print(x.size())  # batchsize x self.hid
        x = self.flat2(funct.relu(x))
        return funct.relu(x)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

    def save(self, filename):
        torch.save(self.state_dict(), filename)


dtype = torch.double
device = 'cpu'  # gdzie wykonywać obliczenia
# device = 'cuda'
HISTORY_N = 10  # ile liczb wchodzi (długość listy)
HID = 3  # ile neuronów w warstwie ukrytej

# liczba próbek treningowych zwracających "1"
N_SAMPLE = 3000  # liczba próbej treningowych zwracających "0"
BATCH_SIZE = 500  # liczba próbek losowych

EPOCHS = 4000
LR = 0.3

# Net creation
net = SequenceNet(HISTORY_N, HID)
net = net.double()
net.load('saves/one.dat')

# Czy obliczenia mają być na GPU
if device == 'cuda':
    net = net.cuda()  # cała sieć kopiowana na GPU

# Dane do uczenia sieci
DX = 0.3
# sample, output = gen_samples(n_samples=N_SAMPLE, history_len=HISTORY_N,model_function=model_sinus, x_from=0, x_to=6.28, dx=DX)
sample, output = gen_samples(n_samples=N_SAMPLE, history_len=HISTORY_N,model_function=model_lorentz, x_from=0, x_to=10, dx=DX)

# zamiana próbek na tensory (możliwa kopia do pamięci GPU)
t_sample = tensor(sample, dtype=dtype, device=device)
t_output = tensor(output, dtype=dtype, device=device)

# przetasowanie całośći
sample_count = t_sample.size()[0]
per_torch = torch.randperm(sample_count)
t_sample = t_sample[per_torch]
t_output = t_output[per_torch]

# "krojenie" próbek na "batches" (grupy próbek, krok optymalizacji po przeliczeniu całej grupy)
b_sample = torch.split(t_sample, BATCH_SIZE)
b_output = torch.split(t_output, BATCH_SIZE)


# print(b_sample)
# print(b_output)


def train():
    # Training setup
    loss_function = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)  # będzie na GPU, jeśli gpu=True

    # Training
    for epoch in range(EPOCHS):
        total_loss = 0
        for (batch_s, batch_o) in zip(b_sample, b_output):
            optimizer.zero_grad()
            # print(batch_in)
            prediction = net(batch_s)
            prediction = prediction.view(-1)  # size: [5,1] -> [5] (flat, same as b_out)
            loss = loss_function(prediction, batch_o)

            if EPOCHS - epoch < 2:
                # pokazujemy wyniki dla 30 ostatnich przypadków, by sprawdzić co sieć przewiduje tak naprawdę
                print('---------')
                print(f'input: {batch_s.tolist()}')
                print(f'pred:{format_list(prediction.tolist())}')
                print(f'outp:{format_list(batch_o.tolist())}')

            total_loss += loss

            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            print(f' epoch:{epoch}, loss:{total_loss:.6f}')
    # Optional result save
    net.save('saves/one.dat')
    print('net saved')


def predict():
    history = [model_lorentz(2 + i * DX) for i in range(HISTORY_N)]  # początkowa historia
    full = history.copy()
    for i in range(100):
        history_t = tensor([history], dtype=dtype, device=device)
        history_batch = torch.split(history_t, BATCH_SIZE)
        # print(history_batch)
        nxt = net(history_batch[0])  # szukamy predykcji następnej wartości
        val = float(nxt[0][0])
        print(f'{history} → {val}')
        full.append(val)
        history.append(val)
        history = history[1:]

    import matplotlib.pyplot as plt
    plt.plot(full, linestyle='dotted')
    plt.show()


if __name__ == '__main__':
    # train()
    predict()
