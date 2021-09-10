from time import sleep

from torch import nn, optim
import torch.nn.functional as funct

from sequence_prediction.text_prediction.sample_generator import gen_samples
from sequence_prediction.text_prediction.translate import encode_char, decode_char, encode_with_channels
from sequence_prediction.text_prediction.utils import compare_lists
from supervised.helper import *


class TextSequenceNet(nn.Module):
    """
        Simple NN: input(sz) ---> flat(hid) ---> 1
    """

    def __init__(self, n_history, hid, n_letters):
        super().__init__()
        self.hid = hid
        self.n_letters = n_letters

        self.conv1 = nn.Conv1d(in_channels=n_letters, out_channels=CH1, kernel_size=5, padding=2)  # rozmiar niezmienny

        self.flat_input_size = CH1 * n_history  # out_channels x n_history

        self.flat1 = nn.Linear(self.flat_input_size, hid, True)
        self.flat2 = nn.Linear(hid, n_letters, True)

    def forward(self, x):
        """ Main function for evaluation of input """

        x = funct.relu(self.conv1(x))  # convolution
        # print('c', x.size())

        x = x.view(-1, self.flat_input_size)  # flatten the output for fully connected

        x = self.flat1(x)
        # print('x', x.size())
        x = self.flat2(funct.relu(x))
        # print(x.size())
        return funct.relu(x)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

    def save(self, filename):
        torch.save(self.state_dict(), filename)


dtype = torch.float
device = 'cpu'  # gdzie wykonywać obliczenia
# device = 'cuda'
N_HISTORY = 13  # ile liczb wchodzi (długość listy)
N_LETTERS = 5

CH1 = 12  # channeli po pierwszej warstwie konwolucyjnej
HID = 6  # ile neuronów w warstwie ukrytej

N_SAMPLE = 3000  # liczba próbej treningowych
BATCH_SIZE = 500  #

EPOCHS = 1000
LR = 0.001


def create_net(load_saved_net=True) -> TextSequenceNet:
    # Net creation
    new_net = TextSequenceNet(N_HISTORY, HID, N_LETTERS)
    new_net = new_net.float()
    if load_saved_net:
        new_net.load('saves/one.dat')

    # Czy obliczenia mają być na GPU
    if device == 'cuda':
        new_net = new_net.cuda()  # cała sieć kopiowana na GPU
    return new_net


def prepare_samples():
    # Dane do uczenia sieci
    sample, output = gen_samples(n_samples=N_SAMPLE, history_len=N_HISTORY)  # N_SAMPLE x N_LETTERS x N_HISTORY

    # zamiana próbek na tensory (możliwa kopia do pamięci GPU)
    t_sample = tensor(sample, dtype=dtype, device=device)
    t_output = tensor(output, dtype=dtype, device=device)

    t_sample = t_sample.view(-1, N_LETTERS, N_HISTORY)
    t_output = t_output.view(-1, N_LETTERS)

    # przetasowanie całośći
    sample_count = t_sample.size()[0]
    per_torch = torch.randperm(sample_count)
    t_sample = t_sample[per_torch]
    t_output = t_output[per_torch]

    # "krojenie" próbek na "batches" (grupy próbek, krok optymalizacji po przeliczeniu całej grupy)
    b_sample = torch.split(t_sample, BATCH_SIZE)
    b_output = torch.split(t_output, BATCH_SIZE)
    return b_sample, b_output


def train(net, b_sample, b_output):
    # Training setup
    loss_function = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)  # będzie na GPU, jeśli gpu=True

    # Training
    for epoch in range(EPOCHS):
        total_loss = 0
        for (batch_s, batch_o) in zip(b_sample, b_output):
            # print('batch sizes: ', batch_s.size(), batch_o.size())
            optimizer.zero_grad()
            prediction = net(batch_s)
            loss = loss_function(prediction, batch_o)

            if EPOCHS - epoch < 2:
                # pokazujemy wyniki dla 30 ostatnich przypadków, by sprawdzić co sieć przewiduje tak naprawdę
                # print('---------')
                # print(f'input: {batch_s.tolist()}')
                # pred_2d = prediction.view(-1)
                same, n = compare_lists(prediction.tolist(), batch_o.tolist(), eps=0.05)
                print(f'correct: {same}/{n} = {same / n * 100 : 3.2f}%')
                pred_list = prediction.tolist().copy()
                round_list_2d(pred_list)

                print(f'pred:{pred_list}')
                print(f'outp:{batch_o.tolist()}')

            total_loss += loss

            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            print(f' epoch:{epoch}, loss:{total_loss:.6f}')
    # Optional result save
    net.save('saves/one.dat')
    print('net saved')
    sleep(2)

def round_list_2d(a):
    for r in range(len(a)):
        for c in range(len(a[0])):
            a[r][c] = round(a[r][c],1)

def predict(net):
    # history = ' aaa bbb'
    history = 'aaa bbb ccc d'
    print(f'starting with {N_HISTORY} chars: [{history}]')
    assert len(history) == N_HISTORY
    full = history
    for i in range(50):
        his_encoded = encode_with_channels(history, N_LETTERS)

        history_t = tensor([his_encoded], dtype=dtype, device=device)
        history_batch = torch.split(history_t, BATCH_SIZE)

        nxt = net(history_batch[0])  # szukamy predykcji następnej wartości
        val = nxt[0].tolist()
        print(f'{history} → {val}')
        best_c = decode_char(val)
        full += best_c
        history += best_c
        history = history[1:]
    print(f'predicted: [{full}]')


if __name__ == '__main__':
    net_ = create_net(load_saved_net=True)
    for i in range(1):
        b_sample_, b_output_ = prepare_samples()
        train(net_, b_sample_, b_output_)
    predict(net_)
