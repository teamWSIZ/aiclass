from random import choice, random

import torch.nn.functional as funct
from torch import nn, optim

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
from reinforcement_learning.tic_tac_toe.ttt_utils import apply_move, valid_moves, is_winning, print_state, random_state, \
    is_draw
from supervised.helper import *


class TicTacToeNet(nn.Module):
    """
        Simple NN: input(sz) ---> flat(hid) ---> 1
    """

    def __init__(self, sz, hid, hid2, out_sz):
        super().__init__()
        self.hid = hid
        self.hid2 = hid2
        self.sz = sz
        self.flat1 = nn.Linear(sz, hid, True)
        self.flatH = nn.Linear(hid, hid2, True)
        self.flat2 = nn.Linear(hid2, out_sz, True)

    def forward(self, x):
        """ Main function for evaluation of input """
        x = x.view(-1, self.sz)
        # print(x.size())  # batchsize x self.sz
        x = self.flat1(x)
        # print(x.size()) # batchsize x self.hid
        x = self.flatH(funct.relu(x))  # extra HID2 layer
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
HID = 20  # ile neuronów w warstwie ukrytej
HID2 = 20  # ile neuronów w warstwie ukrytej
OUT_SIZE = 9  # proponowane ruchy dla "cross" (trzeba sprawdzić czy są "valid")

EPOCHS = 1200
ROUNDS = 60
LR = 0.001
BATCH_SIZE = 10000

VICTORY = 100
DRAW = 50
DEFEAT = 0

# Wybór pozycji dla których będziemy poprawiali predykcje
MIN_DONE = 2
MAX_DONE = 4

# Net creation
net = TicTacToeNet(IN_SIZE, HID, HID2, OUT_SIZE)
# net = net.double()
net.load('saves/two.dat')

if device == 'cuda':
    net = net.cuda()  # cała sieć kopiowana na GPU

# Training setup
loss_function = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)  # będzie na GPU, jeśli gpu=True


def format_list(w: List) -> str:
    s = '['
    for x in w:
        s += f'{x:5.2f}, '
    return s[:-2] + ']'


def better_prediction_after_move(net: TicTacToeNet, state: List[int], move_x, discount_lambda) -> Tuple[float, List]:
    """
    Funkcja podająca spodziewaną nagrodę "o 1 ruch dalej" ...
    Funkcja ocenia wynik wykonania ruchu `move_x` (przez 'x') w stanie "state"

    Zakładamy, że "o" wybierze taki ruch, by zminimalizować prognozowany przez `net` reward następnej
    pozycji "x"-a.
    :return: (discounted_better_prediction, state_for_x_after_optimal_play)
    """
    nstate_o = apply_move(state, move_x)
    if is_winning(nstate_o[9:]):
        return DEFEAT, state  # nawet przed ruchem pozycja była przegrana
    if is_winning(nstate_o[:9]):
        # nasz ruch (x-em) doprowadził do zwycięstwa --> VICTORY
        return VICTORY, state
    if is_draw(nstate_o):
        return DRAW, state

    worst_case_for_o = VICTORY
    best_final_state = None
    # ↑↑ Stan planszy po najlepszym ruchu 'o'; potrzebne jeśli chcemy te ruchy wykonać
    # Jest to stan planszy po:
    #    - ruchu "move" dla 'x',
    #    - takim ruchu 'ao' dla 'o', prowadzącego do nnstate_x, aby
    #           oczekiwana wartość najlepszego ruchu 'x' była jak najmniejsza

    valid_o_moves = valid_moves(nstate_o, cross=False)  # wszystkie ruchy gracza "o"
    for ao in valid_o_moves:
        nnstate_x = apply_move(nstate_o, ao)
        if is_winning(nnstate_x[9:]):
            # kółko ma ruch wygrywający, czyli nagroda dla "x" za ruch "ax" w stanie "state" jest DEFEAT
            return -DEFEAT, state
        nnstate_x_t = tensor(nnstate_x, dtype=dtype, device=device)

        # predykcja wartości stanu "nnstate_x" przez akutalną sieć neuronową
        best_value_for_x = max(net(nnstate_x_t)[0].tolist())

        # 'o' szuka stanu w którym najlepszy ruch dla 'x' daje 'x'-owi jak najmniej
        # worst_case_for_o = min(worst_case_for_o, best_value_for_x)
        if best_value_for_x < worst_case_for_o:
            worst_case_for_o = best_value_for_x
            best_final_state = nnstate_x
    return discount_lambda * worst_case_for_o, best_final_state


def get_updated_prediction(net, state: List[int], move, discount_lambda, verbose=False) -> Tuple[
    List[float], List[int]]:
    prediction = get_move_values(state)

    if verbose:
        print('stan:')
        print_state(state)
        print('predykcja początkowa:', format_list(prediction))
    better_prediction, state_after_optimal_play = better_prediction_after_move(net, state, move, discount_lambda)
    at = move.index(1)  # move ma tylko jedną jedynkę
    prediction[at] = better_prediction
    if verbose:
        print(f'predykcja aktualna  : {format_list(prediction)}, at={at}')
    return prediction, state_after_optimal_play


def set_prediction_of_invalid_moves(prediction: List[float], v_moves: List[List[int]]):
    """
    Zmieniamy "prediction" tak by było = DEFEAT, dla ruchów których nie można wykonać.
    """
    for position in range(9):
        move = [1 if i == position else 0 for i in range(18)]
        if move not in v_moves:
            prediction[position] = DEFEAT


def select_best_valid_move(state: List[int], values: List[float]) -> List[int]:
    """
    Często powtarzająca się czynność: mamy stan (w którym 'x' może wykonać pewne ruchy), i mamy
    predykcje wartości tych ruchów (values). Cel - wybrać ruch o najwyższej wartości, ale tylko z
    grupy ruchów dozwolonych.
    :return:
    """
    v_moves = valid_moves(state, cross=True)
    valid_move_positions = set([m.index(1) for m in v_moves])

    if len(v_moves) == 0:
        raise RuntimeError(f'No valid moves found for state: {state}; cannot select any.')

    best_move = None
    best_value = -100
    for at in range(9):
        if at not in valid_move_positions:
            continue
        if values[at] > best_value:
            best_value = values[at]
            best_move = [1 if i == at else 0 for i in range(18)]
    return best_move


def get_move_values(state: List[int]) -> List[float]:
    """
    :return: Wartości ruchów 'x'-a dla stanu 'state'
    """
    t_state = tensor(state, dtype=dtype, device=device)
    return net(t_state).tolist()[0]  # wartości ruchów dla stanu `state`


def generate_updated_predictions(net, DISCOUNT_LAMBDA, sample_size, verbose=False,
                                 follow_best_move_chance=0.0, play_game=False) -> Tuple:
    """
    Używając sieci "net" generujemy poprawione predykcje wartości ruchów dla `sample_size` pozycji.
    Predykcje te można wykorzystać w kolejnej iteracji uczenia sieci.
    :arg follow_best_move_chance - szansa, że wygenerujemy lepszą "prediction" dla ruchu aktualnie uważanego za najlepszy
    :return:
    """
    samples = []
    outputs = []
    current_board = None
    continued_plays = 0

    while len(samples) < sample_size:
        # wybór pozycji początkowej
        if current_board is None:
            moves_done = randint(MIN_DONE, MAX_DONE)
            state = random_state(moves_done, moves_done)  # losujemy stan z moves_done 'x'-w i 'o'
        else:
            state = current_board
            continued_plays += 1

        # wybór ruchu do wykonania/sprawdzenia lepszej predykcji
        v_moves = valid_moves(state, cross=True)
        if len(v_moves) == 0:
            continue  # brak dozwolonych ruchów
        if random() < follow_best_move_chance:
            # wybieramy ruch który spodziewamy się jest najlepszym w pozycji `state`
            move = select_best_valid_move(state, get_move_values(state))
        else:
            move = choice(v_moves)  # losowy ruch

        # optymalizacja #1: jeśli aktualna pozycja jest wygrywajca/przegrywajca, to jej wartość ma być ustalona
        # → dla danego stanu update'ujemy nie tylko wynik dla pozycji `move`, ale też dla innych
        state_after_optimal_play = None
        if is_winning(state[:9]):
            prediction = [VICTORY] * 9
        elif is_winning(state[9:]):
            prediction = [DEFEAT] * 9  # obecna pozycja jest już przegrana
        elif is_draw(state):
            prediction = [DRAW] * 9  # obecna pozycja jest remisowa (plansza pełna)
        else:
            # krok faktycznego liczenia lepszej predykcji
            prediction, state_after_optimal_play = get_updated_prediction(net, state, move, DISCOUNT_LAMBDA,
                                                                          verbose=len(samples) < 10 & verbose)

        # ustalenie następnego stanu do rozpatrzenia - możliwość kontynuacji gry
        if play_game and state_after_optimal_play is not None:
            if prediction != VICTORY and prediction != DEFEAT:  # fixme: prediction jest listą
                current_board = state_after_optimal_play
            else:
                current_board = None
        else:
            current_board = None

        # optymalizacja #2: ustalamy predykcje wartości ruchów których nie można wykonać na DEFEAT
        set_prediction_of_invalid_moves(prediction, v_moves)

        samples.append(state)
        outputs.append(prediction)
        if verbose:
            print('-.-' * 20)

    return samples, outputs


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
            b_samples, b_outputs = convert_to_batches_tensors(samples, outputs, BATCH_SIZE)  # też "shuffle"

    pass


def RL_teaching_loop():
    # używamy sieci neuronowej "net"; na początku losowej...
    for i in range(ROUNDS):
        print(f'---- ROUND {i}')
        # generujemy nowe dane do uczenia sieci -- dane ktore zawieraja "lepsze" oceny pozycji
        ss, oo = generate_updated_predictions(net, DISCOUNT_LAMBDA=0.95, sample_size=50000,
                                              verbose=(i % 10 == 0),
                                              follow_best_move_chance=0.30, play_game=False)

        # uczenie sieci neuronowej "lepszymi" predykcjami
        train_net(net, samples=ss, outputs=oo, n_epochs=EPOCHS)
        print('-' * 10)

    # Optional result save
    net.save('saves/two.dat')
    print('net saved')


def interactive_play():
    state = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]
    at = randint(0, 8)
    state = [1 if i == at else 0 for i in range(18)]
    while True:
        print_state(state)
        o_move_at = int(input('> ')) + 9
        move = [1 if i == o_move_at else 0 for i in range(18)]
        nxstate = apply_move(state, move)
        print_state(nxstate)

        # wybór najlepszego ruchu bota...
        # todo
        t_state = tensor(nxstate, dtype=dtype, device=device)
        values = net(t_state).tolist()[0]
        print('predykcja wartości ruchów x-a:', format_list(values))

        move_values = []
        for i in range(9):
            move_values.append((values[i], i))
        move_values.sort(reverse=True)  # posortowane od ruchów o największej wartości
        move = None
        v_moves = valid_moves(nxstate)
        for (value, position) in move_values:
            # sprawdzenie który ruch o największej wartości jest dozwolony
            move_try = [1 if i == position else 0 for i in range(18)]
            if move_try in v_moves:
                move = move_try
                break
        no_state = apply_move(nxstate, move)
        state = no_state
        print('-------')


if __name__ == '__main__':
    # RL_teaching_loop()
    interactive_play()
