import torch
from torch import nn, optim
import torch.nn.functional as funct

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
from supervised.helper import *


class MyNet(nn.Module):
    """
        Simple NN: input(sz) ---> flat(hid) ---> 1
    """

    def __init__(self, sz, hid):
        super().__init__()
        self.hid = hid
        self.sz = sz
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=5, stride=1, padding=2)
        self.flat1 = nn.Linear(in_features=sz * 2, out_features=hid, bias=True)
        self.flat2 = nn.Linear(in_features=hid, out_features=1, bias=True)

    def forward(self, x):
        """ Main function for evaluation of input """
        x = x.view(-1, 1, self.sz)  # wchodzi cały batch; czyli [batch][1 channel][SZ]
        # print(x.size())
        x = self.conv1(x)
        x = funct.relu(x)
        # print(x.size())
        x = x.view(-1, 2 * self.sz)
        # print(x.size())
        x = self.flat1(x)
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
N = 30  # ile liczb wchodzi (długość listy)
HID = 1  # ile neuronów w warstwie ukrytej
N_POSITIVE = 10
# liczba próbek treningowych zwracających "1"
N_RANDOM = 200  # liczba próbej treningowych zwracających "0"

BATCH_SIZE = 50  # liczba próbek losowych

EPOCHS = 1000
LR = 0.001

# Net creation
net = MyNet(N, HID)
net = net.double()
# net.load('saves/one.dat')

# Czy obliczenia mają być na GPU
if device == 'cuda':
    net = net.cuda()  # cała sieć kopiowana na GPU

# Próbki napewno dodatnie
sample1, output1 = gen_all_samples_1_distance_given(N, 2)
# Próbki losowe
sample_r, output_r = gen_random_trainset_1twice_distance_given(N, N_RANDOM, 2)

# jednolita lista sampli, próbki dodatnie NPOSITIVE razy
sample = []
output = []
for _ in range(N_POSITIVE):
    sample.extend(sample1)
    output.extend(output1)
sample.extend(sample_r)
output.extend(output_r)

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

        if EPOCHS - epoch < 30:
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
