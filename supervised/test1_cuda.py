import torch
from torch import nn, optim
import torch.nn.functional as funct

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
from supervised.helper import *

gpu = False


class MyNet(nn.Module):
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
        x = self.flat1(x)
        x = self.flat2(funct.relu(x))
        return funct.relu(x)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

    def save(self, filename):
        torch.save(self.state_dict(), filename)


typ = torch.float

N = 6  # ile liczb wchodzi (długość listy)
HID = 4  # ile neuronów w warstwie ukrytej
N_SUCCESS = 100  # liczba próbek treningowych zwracających "1"
N_FAILURE = 400  # liczba próbej treningowych zwracających "0"

BATCH_SIZE = 40  # liczba próbek pokazywanych jednocześnie (zanim nastąpi krok modyfikacji parametrów sieci)

EPOCHS = 2000
LR = 0.0001

# Net creation
net = MyNet(N, HID)
net.load('saves/one.dat')

if gpu:
    net = net.cuda()  # cała sieć kopiowana na GPU

# Training setup
loss_function = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)  # będzie na GPU, jeśli gpu=True

# Training data (tensory na GPU, jeśli gpu=True)
batches_in, batches_ou = generate_2x1_dist2(N, count=N_SUCCESS + N_FAILURE, need_1=N_SUCCESS,
                                            batch_size=BATCH_SIZE, gpu=gpu)

# Training
for epoch in range(EPOCHS):
    total_loss = 0
    for (batch_in, batch_out) in zip(batches_in, batches_ou):
        optimizer.zero_grad()
        # print(batch_in)
        prediction = net(batch_in)
        prediction = prediction.view(-1)  # size: [5,1] -> [5] (flat, same as b_out)
        loss = loss_function(prediction, batch_out)

        if EPOCHS - epoch < 30:
            # pokazujemy wyniki dla 30 ostatnich przypadków, by sprawdzić co sieć przewiduje tak naprawdę
            print('---------')
            print(f'input: {batch_in.tolist()}')
            print(f'pred:{format_list(prediction.tolist())}')
            print(f'outp:{format_list(batch_out.tolist())}')

        total_loss += loss

        loss.backward()
        optimizer.step()
    if epoch % 20 == 0:
        print(f' epoch:{epoch}, loss:{total_loss:.6f}')

# Optional result save
net.save('saves/one.dat')
