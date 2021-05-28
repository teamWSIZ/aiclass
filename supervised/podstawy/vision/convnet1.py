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

    def __init__(self, res, hid, noutput):
        super().__init__()
        self.hid = hid
        self.res = res  # 3 channels, 256 x 256 resolution
        self.nsigns = noutput

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)  # pozostawia RES x RES
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)  # RES → RES/4 (256 → 64)
        #
        # 0 1 1 0
        # 0 1 1 0
        # 0 2 1 0
        # 0 1 1 0
        #
        # -> 0 1 1 0 0 1 1 0 0 2 1 0 0 1 1 0

        self.flat_input_size = 3 * (res // 4) ** 2

        self.flat1 = nn.Linear(self.flat_input_size, hid, bias=True)
        self.flat2 = nn.Linear(hid, self.nsigns, bias=True)  # 2: wyjscie. → poszukiwanie nsigns wzorców

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
        self.load_state_dict(torch.load(filename, map_location=device))
        self.eval()

    def save(self, filename):
        torch.save(self.state_dict(), filename)


dtype = torch.float
# device = 'cpu'  # gdzie wykonywać obliczenia
device = 'cuda'

# Parametry sieci
RES = 256  # ile liczb wchodzi (długość listy)
HID = 12  # ile neuronów w warstwie ukrytej

# Setup próbek zawierające znaki (SIGNS) które mają być wykrywane i klasyfikowane przez sieć
N_POSITIVE = [160, 160, 160, 480]
SIGNS = ['sign.png', 'm.png', 'kali1.png', None]
nsigns = len(SIGNS)

# Tworzenie sieci neuronowej
net = MyNet(res=RES, hid=HID, noutput=nsigns)
net = net.float()

# N_NEGATIVE = 120  # liczba próbek treningowych zwracających "0"
EPOCHS = 2000
BATCH_SIZE = 120
LR = 0.002  # learning rate

# Wczytywanie poprzednio-zapisanej sieci
# net.load('saves\\one.dat')    # wersja windows
# net.load('saves/one.dat')

if device == 'cuda':
    net = net.cuda()  # cała sieć kopiowana na GPU


# ########################################
# Przygotowanie danych do uczenia sieci


def generate_training_set():
    # Obrazy zawierające próbki (SIGNS) na pozycjach początkowych + ostatni element tylko z tłami
    ssample = []
    for sign, count in zip(SIGNS, N_POSITIVE):
        ssample.append(generate_sample(count, sign))  # krok generowania danych do uczenia sieci

    sample_sizes = [ss.size()[0] for ss in ssample]

    # Ustalenie outputu -- dla "SIGNS" ustawiamy kolejne jedynki
    # nsigns=3 → [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    w = [[0] * i + [1] + [0] * (nsigns - i - 1) for i in range(nsigns)]  # meh... taki oneliner...

    soutput = [tensor(w[i] * sample_sizes[i], dtype=dtype, device=device) for i in range(nsigns)]

    # przerzucenie danych na GPU (NVIDIA) jeśli chcemy...
    if device == 'cuda':
        for i in range(nsigns):
            ssample[i] = ssample[i].cuda()

    # jednolita lista sampli (wszystkie w jednym tensorze)
    sample_ = torch.cat(ssample, 0)
    output_ = torch.cat(soutput, 0)
    output_ = output_.view(-1, nsigns)
    return sample_, output_


# funkcja "tasujaca" karty-próbki stosowane do uczenia sieci
def shuffle_samples_and_outputs(sample_, output_):
    size = sample_.size()[0]
    per_torch = torch.randperm(size)
    shuffled_sample = sample_[per_torch]
    shuffled_output = output_[per_torch]
    return shuffled_sample, shuffled_output


sample, output = generate_training_set()

print('sample dimension: ', sample.size())
print('output dimension: ', output.size())

# tasowanie (shuffle) i "krojenie" próbek na "batches" (grupy próbek, krok optymalizacji po przeliczeniu całej grupy)
t_sample, t_output = shuffle_samples_and_outputs(sample, output)
b_sample = torch.split(t_sample, BATCH_SIZE)
b_output = torch.split(t_output, BATCH_SIZE)

# ########################################
# Proces uczenia sieci
loss_function = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)  # będzie na GPU, jeśli gpu=True

epoch = 0
while epoch < EPOCHS:
    epoch += 1
    total_loss = 0
    correct = 0
    wrong = 0
    total = 0

    for (batch_s, batch_o) in zip(b_sample, b_output):
        # pętla po "batch-ach", czyli grupach próbek
        optimizer.zero_grad()
        prediction = net(batch_s)
        prediction = prediction.view(-1)
        batch_o = batch_o.view(-1)
        loss = loss_function(prediction, batch_o)

        n = prediction.size()[0]
        for i in range(n):
            pred = prediction[i]
            corr = batch_o[i]
            total_error = abs(pred - corr)
            if total_error < 0.05:
                correct += 1
            if total_error > 0.40:
                wrong += 1
            total += 1

        total_loss += loss

        loss.backward()  # sprawdzenie które zmienne sieci wpływają najbardziej na wynik/błąd
        optimizer.step()  # "lekka" modyfikacja zmiennych sieci

    # Kod "periodycznych" raportów i modyfikacji układu próbek i procesu uczenia sieci

    if epoch % 10 == 0:
        print(f' epoch:{epoch}, loss:{total_loss:.6f}', f'correct:{correct}/{total}\t wrong:{wrong}/{total}')

    if epoch % 100 == 0:
        print('shuffle!')
        t_sample, t_output = shuffle_samples_and_outputs(sample, output)
        b_sample = torch.split(t_sample, BATCH_SIZE)
        b_output = torch.split(t_output, BATCH_SIZE)

    if epoch % 200 == 0:
        print('generating new samples')
        sample, output = generate_training_set()
        t_sample, t_output = shuffle_samples_and_outputs(sample, output)
        b_sample = torch.split(t_sample, BATCH_SIZE)
        b_output = torch.split(t_output, BATCH_SIZE)

    if epoch == EPOCHS - 1:
        s = input('dodać 100 epok? (y/n)')
        if s == 'y':
            EPOCHS += 100

# Optional result save
# net.save('saves\\one.dat')    # wersja windows
net.save('saves/one.dat')
print('net saved')
