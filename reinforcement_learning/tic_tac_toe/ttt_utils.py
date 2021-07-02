from random import randint
from typing import List

# [... ... ...   ... ... ...] == stan == 18 liczb
from cachetools import cached, LRUCache

from reinforcement_learning.tic_tac_toe.bity import encode_state


def is_winning(state: List[int]):
    """
    :param state: płaska reprezentacja planszy 3x3 (wszystkie liczby zebrane do pojedynczej listy), lista na 9 elementów
    :return: True jeśli pozycja wygrywa w TTT, False jeśli nie wygrywa (może być "otwarta")
    """
    if state[0:3] == [1] * 3 or state[3:6] == [1] * 3 or state[6:9] == [1] * 3:
        return True  # sprawdzenie rzędów
    for c in range(3):
        # sprawdzenie kolumny nr "c"
        if state[c] == 1 and state[c + 3] == 1 and state[c + 6] == 1:
            return True
    # przekątna
    if state[0] == 1 and state[4] == 1 and state[8] == 1:
        return True
    # "anty-przekątna"
    if state[2] == 1 and state[4] == 1 and state[6] == 1:
        return True
    return False


def is_draw(state):
    """
    :return: True jeśli plansza jest pełna i nikt nie wygrał
    """
    s = sum(state)
    if s == 18 and not is_winning(state[:9]) and not is_winning(state[9:]):
        return True
    else:
        return False


def print_state(state: List[int]):
    """
    Drukuje planszę z pionkami; `state` to lista z 18 elementami
    """
    board = ['.'] * 9
    for i in range(9):
        if state[i] == 1:
            board[i] = 'x'
        if state[9 + i] == 1:
            board[i] = 'o'
    for i in range(3):
        print(*board[i * 3:(i + 1) * 3])


# @cached(cache=LRUCache(maxsize=10000))
def valid_moves(state: List[int], cross=True):
    """
    Ruch to tablica 18 elementów; pierwsze 9 odpowiadają ruchom "x"-a; pozostałe ruchom "o"-ka;
    w całej tablicy jest tylko jedna 1; pozostałe to 0.

    Np. ruch w rzędzie 1 (numerowane od 0), i kolumnie 0 "x"-a ma postać:
    [0,0,0,1,0,0,0,0,0, 0,0,0,0,0,0,0,0,0]

    :param state: obecny stan planszy
    :param cross: czy szukamy ruchów dla "x" (False → "o")
    :return: tablica dostępnych ruchów (tablic 18-elementowych)

    """
    enc = encode_state(state)
    if enc in valid_moves_cache:
        return valid_moves_cache[enc]

    moves = []
    offset = 0 if cross else 9
    for i in range(9):
        if state[i] == 0 and state[9 + i] == 0:
            move = [0] * 18
            move[i + offset] = 1
            moves.append(move)

    valid_moves_cache[enc] = moves
    return moves


valid_moves_cache = dict()


def apply_move(state: List[float], move):
    """
    :return: stan po zaaplikowaniu ruchu
    """
    nstate = []
    for (s, m) in zip(state, move):
        nstate.append(s + m)
    return nstate


def best_moves(state, qvalues):
    """
    :return: Lista dozwolonych ruchów "cross"-a, z ich wartościami,
                posortowana od największych wartości qvalues (expected reward)
    """
    valid = valid_moves(state, cross=True)
    moves = []
    for i in range(9):
        a = [1 if j == i else 0 for j in range(16)]
        if a in valid:
            moves.append((qvalues[i], a))
    moves.sort(reverse=True)
    return moves


def random_state(n_cross, n_circle) -> List[int]:
    s = [0] * 18

    # losowanie "x"-ów
    while sum(s) < n_cross:
        pos = randint(0, 8)
        s[pos] = 1

    # losowanie "o"-ek
    while sum(s) < n_cross + n_circle:
        pos = randint(9, 17)
        if s[pos - 9] == 0:
            s[pos] = 1

    return s


"""
Todo: spróbować "z-cache-ować" informację, którą obliczamy w valid_moves(state)... czyli 
tak zapamiętać wszystkie "valid moves", by je potem zwracać bezpośrednio ze struktury danych, bez
obliczania. 


Cache: mapa (lub dict) odwzorowująca argument funkcji (tu: stan) w wartości zwracane z funkcji (tu listę list int-ów)


"""

if __name__ == '__main__':
    """
x x o
o x .
x o o
"""
    print(is_winning([1, 1, 0, 0, 1, 0, 1, 0, 0]), True)
    # print(is_winning([1, 1, 0, 0, 0, 0, 0, 0, 0]), False)
    # print(is_winning([1, 1, 0, 1, 0, 0, 1, 0, 0]), True)
    # print(is_winning([1, 0, 0, 0, 1, 0, 1, 0, 1]), True)
    # s = [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    # print_state(s)
    # move = [0,0,0,0,0,0,0,0,0, 0,0,0,1,0,0,0,0,0]

    # s = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    # print_state(s)
    # vm = valid_moves(s, cross=True)
    # print(vm)
    # for m in vm:
    #     print('---')
    #     print_state(m)
    # print(is_winning(s[9:]))

    # nstate = apply_move(s, move)
    # print(nstate)
    # print_state(nstate)
    # next_moves = valid_moves(s, cross=False)
    # for m in next_moves:
    #     print('---')
    #     print_state(apply_move(s, m))
