from random import choice

import torch
from torch import nn, optim
import torch.nn.functional as funct

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
from reinforcement_learning.tic_tac_toe.ttt_utils import apply_move, valid_moves, is_winning, print_state, random_state, \
    is_draw
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


dtype = torch.float
device = 'cpu'  # lub 'cuda'
IN_SIZE = 18  # ile liczb wchodzi (długość listy)
HID = 3  # ile neuronów w warstwie ukrytej
OUT_SIZE = 9  # proponowane ruchy dla "cross" (trzeba sprawdzić czy są "valid")

EPOCHS = 10000
LR = 0.01
BATCH_SIZE = 500

# Net creation
net = TicTacToeNet(IN_SIZE, HID, OUT_SIZE)
# net = net.double()
# net.load('saves/one.dat')

if device == 'cuda':
    net = net.cuda()  # cała sieć kopiowana na GPU

# Training setup
loss_function = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)  # będzie na GPU, jeśli gpu=True


def better_prediction_after_move(net: TicTacToeNet, state: List[int], move_x, discount_lambda) -> float:
    """
    Funkcja podająca spodziewaną nagrodę "o 1 ruch dalej" ...
    Funkcja ocenia wyniky wykonania ruchu "ax" (przez "x") w stanie "state"

    Zakładamy, że "o" wybierze taki ruch, by zminimalizować prognozowany przez `net` reward następnej
    pozycji "x"-a.
    :return:
    """
    nstate_o = apply_move(state, move_x)
    if is_winning(nstate_o[9:]):
        return -100  # nawet przed ruchem pozycja była przegrana
    if is_winning(nstate_o[:9]):
        # nasz ruch (x-em) doprowadził do zwycięstwa --> 100pkt
        return 100

    valid_o_moves = valid_moves(nstate_o, cross=False)  # wszystkie ruchy gracza "o"
    best_for_o = 100
    best_final_state = None  #todo: tu sprawdzimy stan po najlepszym ruchu "o"
    for ao in valid_o_moves:
        nstate_x = apply_move(nstate_o, ao)
        if is_winning(nstate_x[9:]):
            return -100  # kółko ma ruch wygrywający, czyli nagroda dla "x" za ruch "ax" w stanie "state" jest -100
        nstate_x_tensor = tensor(nstate_x, dtype=dtype, device=device)

        # predykcja wartości stanu "nstate_x" przez akutalną sieć neuronową
        best_for_x = max(net(nstate_x_tensor)[0].tolist())

        # o szuka stanu w którym najlepszy ruch dla x daje x-owi jak najmniej
        best_for_o = min(best_for_o, best_for_x)
    # todo: !!!!!!!!!!!!!!!
    #  tutaj można zwrócić nie tylko wartość (lepszej predykcji), ale też stan, w którym znajdzie się plansza
    #  po dwóch ruchach:
    #    - ruchu "move" dla x-a,
    #    - ruchu "ao" prowadzącego do "best_for_o"
    return discount_lambda * best_for_o


def get_updated_prediction(net, state: List[int], move, discount_lambda, verbose=False) -> List[float]:
    t_state = tensor(state, dtype=dtype, device=device)  # stan-tensor; na niego można aplikować `net`
    prediction = net(t_state)[0].tolist()

    if verbose:
        print('stan:')
        print_state(state)
        print('predykcja początkowa:', format_list(prediction))
    better_prediction = better_prediction_after_move(net, state, move, discount_lambda)
    # print(f'lepsza predykcja wartości dla ruchu: {move}', better_prediction)
    at = move.index(1)  # move ma tylko jedną jedynkę
    prediction[at] = better_prediction
    if verbose:
        print(f'predykcja aktualna  : {format_list(prediction)}, at={at}')
    return prediction


def set_prediction_of_invalid_moves(prediction: List[float], v_moves: List[List[int]], fixed_value=-30):
    """
    Zmieniamy "prediction" tak by było = fixed_value, dla ruchów których nie można wykonać.
    """
    for position in range(9):
        move = [1 if i == position else 0 for i in range(18)]
        if move not in v_moves:
            prediction[position] = fixed_value


def generate_updated_predictions(net, DISCOUNT_LAMBDA, sample_size, verbose=False,
                                 follow_best_move_chance=0.0) -> Tuple:
    """
    Używając sieci "net" sprawdzamy "krok w przód" dla `sample_size` pozycji-ruchów, generując
    poprawione predykcje dla sieci. Predykcje te można wykorzystać w kolejnej iteracji uczenia sieci.
    :return:
    """
    samples = []
    outputs = []

    while len(samples) < sample_size:
        moves_done = randint(3, 4)
        state = random_state(moves_done, moves_done)  # losujemy stan z moves_done 'x'-w i 'o'

        v_moves = valid_moves(state, cross=True)
        x = randint(0, 100)
        if len(v_moves) > 0 and x / 100 < follow_best_move_chance:
            # select best
            t_state = tensor(state, dtype=dtype, device=device)
            values = net(t_state).tolist()
            move_values = []
            for i in range(9):
                move_values.append((values[0][i], i))
            move_values.sort(reverse=True)
            move = None
            for (value, position) in move_values:
                move_try = [1 if i == position else 0 for i in range(18)]
                if move_try in v_moves:
                    move = move_try
                    break
                    # todo: check this!
        else:
            move = choice(v_moves)  # losowy ruch

        if is_winning(state[:9]):
            prediction = [100] * 9
        elif is_winning(state[9:]):
            prediction = [-100] * 9
        elif is_draw(state):
            prediction = [0] * 9
        else:
            prediction = get_updated_prediction(net, state, move, DISCOUNT_LAMBDA, verbose)

        set_prediction_of_invalid_moves(prediction, v_moves, fixed_value=0)

        samples.append(state)
        outputs.append(prediction)
        if verbose: print('-.-' * 20)
    return samples, outputs


"""
Uczenie przez "granie" w x-o; 
... to musi być funkcja do "update prediction"...
- musi zaczynać od "state = [0]*18", czyli pusta plansza
- wykonujemy ruch x-a na podstawie net(state)... czyli aktualnych predykcji wartości ruchów...
- ?? jaki ruch ma wykonać 'o'... zróbmy "optymalny"... czyli taki (z dozwolonych), by net(state') 
  (po jego wykonaniu) była dla nas (x-ów) jak najgorsza...
- oprócz zmiany predykcji, zmieniamy stan.. 


"""


def format_list(list) -> str:
    s = '['
    for x in list:
        s += f'{x:5.2f}, '
    return s[:-2] + ']'


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
            # print(batch_s.size())
            # print(batch_o.size())
            # print(prediction.size())
            loss = loss_function(prediction.view(-1), batch_o.view(-1))

            total_loss += loss

            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print(f' epoch:{epoch}, loss:{total_loss:.6f}')
    pass


# kod uzywamy sieci neuronowej "net"; na poczatku losowej...
for i in range(2000):
    # generujemy nowe dane do uczenia sieci -- dane ktore zawieraja "lepsze" oceny pozycji
    ss, oo = generate_updated_predictions(net, 0.95, sample_size=1000,
                                          verbose=(i % 5 == 0),
                                          follow_best_move_chance=i / 200)

    # uczenie sieci neuronowej "lepszymi" predykcjami
    train_net(net, samples=ss, outputs=oo, n_epochs=max(20, i // 10))
    print('-' * 10)

# Optional result save
# net.save('saves/one.dat')
# print('net saved')
