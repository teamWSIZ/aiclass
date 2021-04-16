import torch
from torch import nn, optim
import torch.nn.functional as funct

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
from supervised.helper import *
from convnet_sample_generator import *


class MyNet(nn.Module):
    """
        Simple NN: input(sz) ---> flat(hid) ---> 1
    """

    def __init__(self, res, hid):
        super().__init__()
        self.hid = hid
        self.res = res  # 3 channels, 256 x 256 resolution

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2)  # pozostawia RES x RES
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)  # RES → RES/4 (256 → 64)

        self.flat_input_size = 3 * (res // 4) ** 2

        self.flat1 = nn.Linear(self.flat_input_size, hid, bias=True)
        self.flat2 = nn.Linear(hid, 2, bias=True)  # 2: wyjscie ~ [0,0] lub [1,0] lub [0,1] → poszukiwanie dwóch wzorców

    def forward(self, x):
        """ Main function for evaluation of input """

        x = x.view(-1, 3, self.res, self.res)
        # print('input size: ', x.size()) # batch x full; [20, 196608] ; 256 * 256 * 3 = 196608
        x = self.pool1(funct.relu(self.conv1(x)))
        x = x.view(-1, self.flat_input_size)

        x = self.flat1(x)
        x = self.flat2(funct.relu(x))
        return funct.relu(x)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

    def save(self, filename):
        torch.save(self.state_dict(), filename)


dtype = torch.float
# device = 'cpu'  # gdzie wykonywać obliczenia
device = 'cuda'

# Parametry sieci
RES = 256  # ile liczb wchodzi (długość listy)
HID = 6  # ile neuronów w warstwie ukrytej
# Net creation
net = MyNet(res=RES, hid=HID)
net = net.float()

# Parametry training-set-u
N_POSITIVE_TYPE1 = 60
N_POSITIVE_TYPE2 = 60
N_NEGATIVE = 120  # liczba próbej treningowych zwracających "0"
EPOCHS = 200
BATCH_SIZE = 60
LR = 0.001

# Wczytywanie poprzednio-zapisanej sieci
net.load('saves/one.dat')


if device == 'cuda':
    net = net.cuda()  # cała sieć kopiowana na GPU

# Próbki "dodatnie" (zawierające szukany element)
sample1 = generate_sample(N_POSITIVE_TYPE1, 'sign.png')  # tensor
n1 = sample1.size()[0]
output1 = tensor([1, 0] * n1, dtype=dtype, device=device)

# Próbki "dodatnie" (zawierające szukany element typu 2)
sample2 = generate_sample(N_POSITIVE_TYPE2, 'm.png')  # tensor
n2 = sample2.size()[0]
output2 = tensor([0, 1] * n2, dtype=dtype, device=device)



# Próbki losowe
sample0 = generate_sample(N_NEGATIVE, None)
n0 = sample0.size()[0]
output0 = tensor([0, 0] * n0, dtype=dtype, device=device)

# przerzucenie danych na GPU (NVIDIA) jeśli chcemy...
if device == 'cuda':
    sample1 = sample1.cuda()
    sample0 = sample0.cuda()

# input('rozpocząć proces uczenia? ')

# jednolita lista sampli, próbki dodatnie NPOSITIVE razy
sample = torch.cat((sample0, sample1), 0)
output = torch.cat((output0, output1), 0)
output = output.view(-1, 2)

print('sample: ', sample.size())
print('output: ', output.size())

# przetasowanie całośći
sample_count = n0 + n1
per_torch = torch.randperm(sample_count)
t_sample = sample[per_torch]
t_output = output[per_torch]  # todo: to można powtarzać co kilka epok

# "krojenie" próbek na "batches" (grupy próbek, krok optymalizacji po przeliczeniu całej grupy)
b_sample = torch.split(t_sample, BATCH_SIZE)
b_output = torch.split(t_output, BATCH_SIZE)

# Training setup
loss_function = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)  # będzie na GPU, jeśli gpu=True

# Training
for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    wrong = 0
    total = 0
    for (batch_s, batch_o) in zip(b_sample, b_output):
        optimizer.zero_grad()
        # print(batch_in)
        prediction = net(batch_s)
        prediction = prediction.view(-1)  # size: [5,1] -> [5] (flat, takie samo jak batch_out)
        batch_o = batch_o.view(-1)
        # print(f'prediction: {prediction.size()}, output:{batch_o.size()}')  # sprawdzenie wymiarów: muszą być te same
        loss = loss_function(prediction, batch_o)

        n = prediction.size()[0]
        for i in range(n):
            pred = prediction[i]
            corr = batch_o[i]
            total_error = abs(pred- corr)
            if total_error < 0.05:
                correct += 1
            if total_error > 0.40:
                wrong += 1
            total += 1

        if EPOCHS - epoch < 30 and False:
            # pokazujemy wyniki dla 30 ostatnich przypadków, by sprawdzić co sieć przewiduje tak naprawdę
            print('---------')
            print(f'input: {batch_s.tolist()}')
            print(f'pred:{format_list(prediction.tolist())}')
            print(f'outp:{format_list(batch_o.tolist())}')

        total_loss += loss

        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f' epoch:{epoch}, loss:{total_loss:.6f}', f'correct:{correct}/{total}\t wrong:{wrong}/{total}')

# Optional result save
net.save('saves/one.dat')
print('net saved')
