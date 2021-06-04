import torch
from torch import nn, optim
import torch.nn.functional as funct

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
from reinforcement_learning.tic_tac_toe.ttt_utils import apply_move, valid_moves, is_winning
from supervised.helper import *


class TicTacToeNet(nn.Module):
    """
        Simple NN: input(sz) ---> flat(hid) ---> 1
    """

    def __init__(self, sz, hid, out_sz):
        super().__init__()
        self.hid = hid
        self.sz = sz
        self.flat1 = nn.Linear(sz, hid, True)
        self.flat2 = nn.Linear(hid, out_sz, True)

    def forward(self, x):
        """ Main function for evaluation of input """
        x = x.view(-1, self.sz)
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


dtype = torch.double
# device = 'cpu'  # gdzie wykonywać obliczenia
device = 'cpu'
IN_SZ = 18  # ile liczb wchodzi (długość listy)
HID = 5  # ile neuronów w warstwie ukrytej
OUT_SZ = 9  # proponowane ruchy dla "cross" (trzeba sprawdzić czy są "valid")

EPOCHS = 1000
LR = 0.01

# Net creation
net = TicTacToeNet(IN_SZ, HID, OUT_SZ)
net = net.double()
# net.load('saves/one.dat')

if device == 'cuda':
    net = net.cuda()  # cała sieć kopiowana na GPU

# Próbki napewno dodatnie
# sample1, output1 = gen_all_samples_1_distance_given(N, 2)
# Próbki losowe
# sample_r, output_r = gen_random_trainset_1twice_distance_given(N, N_RANDOM, 2)

# jednolita lista sampli, próbki dodatnie NPOSITIVE razy
# sample = []
# output = []
# for _ in range(N_POSITIVE):
#     sample.extend(sample1)
#     output.extend(output1)
# sample.extend(sample_r)
# output.extend(output_r)

# zamiana próbek na tensory (możliwa kopia do pamięci GPU)
# t_sample = tensor(sample, dtype=dtype, device=device)
# t_output = tensor(output, dtype=dtype, device=device)

# przetasowanie całośći
# sample_count = t_sample.size()[0]
# per_torch = torch.randperm(sample_count)
# t_sample = t_sample[per_torch]
# t_output = t_output[per_torch]

# "krojenie" próbek na "batches" (grupy próbek, krok optymalizacji po przeliczeniu całej grupy)
# b_sample = torch.split(t_sample, BATCH_SIZE)
# b_output = torch.split(t_output, BATCH_SIZE)

# Training setup
loss_function = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)  # będzie na GPU, jeśli gpu=True

s = [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
ts = tensor(s, dtype=dtype)
prediction = net(ts)
print(prediction)
pd = prediction.data
print(pd.tolist()[0])


def expected_reward(state, ax, net: TicTacToeNet, discount_lambda):
    """
    Zakładamy, że "o" wybierze taki ruch, by zminimalizować prognozowany przez `net` reward następnej
    pozycji "x"-a.
    :return:
    """
    nstate_o = apply_move(state, ax)
    if is_winning(nstate_o):
        return 100
    valid_o_moves = valid_moves(nstate_o, cross=False)
    best_for_o = 100
    for ao in valid_o_moves:
        nstate_x = apply_move(nstate_o, ao)
        if is_winning(nstate_x[9:]):
            return -100
        best_for_x = max(net(nstate_x)[0].tolist())
        best_for_o = min(best_for_o, best_for_x)
    return discount_lambda * best_for_o

# Training
# for epoch in range(EPOCHS):
#     total_loss = 0
#     for (batch_s, batch_o) in zip(b_sample, b_output):
#         optimizer.zero_grad()
#         # print(batch_in)
#         prediction = net(batch_s)
#         prediction = prediction.view(-1)  # size: [5,1] -> [5] (flat, same as b_out)
#         loss = loss_function(prediction, batch_o)
#
#         if EPOCHS - epoch < 30:
#             # pokazujemy wyniki dla 30 ostatnich przypadków, by sprawdzić co sieć przewiduje tak naprawdę
#             print('---------')
#             print(f'input: {batch_s.tolist()}')
#             print(f'pred:{format_list(prediction.tolist())}')
#             print(f'outp:{format_list(batch_o.tolist())}')
#
#         total_loss += loss
#
#         loss.backward()
#         optimizer.step()
#     if epoch % 20 == 0:
#         print(f' epoch:{epoch}, loss:{total_loss:.6f}')

# Optional result save
# net.save('saves/one.dat')
# print('net saved')
