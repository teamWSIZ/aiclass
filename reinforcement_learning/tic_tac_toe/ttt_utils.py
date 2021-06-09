from random import randint
from typing import List


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
    moves = []
    offset = 0 if cross else 9
    for i in range(9):
        if state[i] == 0 and state[9 + i] == 0:
            move = [0] * 18
            move[i + offset] = 1
            moves.append(move)
    return moves


def apply_move(state, move):
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



def random_state(n_cross, n_circle):
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

if __name__ == '__main__':
    # print(is_winning([1, 1, 1, 0, 0, 0, 0, 0, 0]), True)
    # print(is_winning([1, 1, 0, 0, 0, 0, 0, 0, 0]), False)
    # print(is_winning([1, 1, 0, 1, 0, 0, 1, 0, 0]), True)
    print(is_winning([1, 0, 0, 0, 1, 0, 1, 0, 1]), True)
    s = [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    print_state(s)
    # move = [0,0,0,0,0,0,0,0,0, 0,0,0,1,0,0,0,0,0]
    # nstate = apply_move(s, move)
    # print(nstate)
    # print_state(nstate)
    next_moves = valid_moves(s, cross=False)
    for m in next_moves:
        print('---')
        print_state(apply_move(s, m))
