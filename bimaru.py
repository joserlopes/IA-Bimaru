# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 80:
# 103938 José António Lopes
# 104139 Rodrigo Manuel Friães

from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)

import sys
import numpy as np


class BimaruState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = BimaruState.state_id
        BimaruState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    def place_4boat_horizontally(self, row, col):
        board = self.board
        new_board = board.place_4boat_horizontally(row, col)
        return BimaruState(new_board)

    def place_4boat_vertically(self, row, col):
        board = self.board
        new_board = board.place_4boat_vertically(row, col)
        return BimaruState(new_board)

    def place_3boat_horizontally(self, row, col):
        board = self.board
        new_board = board.place_3boat_horizontally(row, col)
        return BimaruState(new_board)

    def place_3boat_vertically(self, row, col):
        board = self.board
        new_board = board.place_3boat_vertically(row, col)
        return BimaruState(new_board)

    def place_2boat_horizontally(self, row, col):
        board = self.board
        new_board = board.place_2boat_horizontally(row, col)
        return BimaruState(new_board)

    def place_2boat_vertically(self, row, col):
        board = self.board
        new_board = board.place_2boat_vertically(row, col)
        return BimaruState(new_board)

    def place_1boat(self, row, col):
        board = self.board
        new_board = board.place_1boat(row, col)
        return BimaruState(new_board)


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(self, board_representation: list, board_info: list, board_occupied: list, boat_info: dict, hints=0):
        self.board_representation = board_representation
        self.board_info = board_info
        self.board_occupied = board_occupied
        self.boat_info = boat_info
        self.hints = hints

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board_representation[row][col]

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        board = self.board_representation
        adjacents = ()
        rows_num = len(board)
        for i in (-1, 1):
            new_row = row + i
            if 0 <= new_row < rows_num:
                adjacents += (self.get_value(new_row, col),)
            else:
                adjacents += (None,)
        return adjacents

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        board = self.board_representation
        adjacents = ()
        cols_num = len(board[0])
        for i in (-1, 1):
            new_col = col + i
            if 0 <= new_col < cols_num:
                adjacents += (self.get_value(row, new_col),)
            else:
                adjacents += (None,)
        return adjacents

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 bimaru.py < input_T01

            > from sys import stdin
            > line = stdin.readline().split()
        """
        board = [[None for x in range(10)] for y in range(10)]
        # Transform the 2D list into an np.array to easily acess the columns
        # and make a deep copy of the board
        board = np.array(board)
        # The structure that holds the information relative
        # to the rows and columns. The first row refers to the rows and
        # the second to the columns
        board_info = [[None for x in range(10)] for y in range(2)]

        board_occupied = [[0 for x in range(10)] for y in range(2)]
        # Transform the 2D list into an np.array to easily make a
        # deep copy of the list
        board_occupied = np.array(board_occupied)

        boat_info = {"4piece": 1, "3piece": 2, "2piece": 3, "1piece": 4}

        hints = 0

        # Read everything until EOF is reached
        line = sys.stdin.readline().split()
        while line:
            if line[0] == "ROW":
                for row_board_info, index in zip(line[1:], range(10)):
                    board_info[0][index] = int(row_board_info)
            elif line[0] == "COLUMN":
                for col_board_info, index in zip(line[1:], range(10)):
                    board_info[1][index] = int(col_board_info)
            elif line[0].isnumeric():
                hints = int(line[0])
            elif line[0] == "HINT":
                row = int(line[1])
                col = int(line[2])
                hint = line[3]
                board[row][col] = hint
                # "Occupy" the spot
                if hint != 'W':
                    board_occupied[0][row] += 1
                    board_occupied[1][col] += 1
                if hint == 'C':
                    boat_info["1piece"] -= 1

            line = sys.stdin.readline().split()

        return Board(board, board_info, board_occupied, boat_info, hints)

    def print(self):
        for i in range(len(self.board_representation)):
            for j in range(len(self.board_representation)):
                print(f"{self.get_value(i, j)}", end="")
            print("")

    def fill_with_water(self, row_col):
        """ Fills a given row or column with water on avaliable spots."""
        for i in range(len(row_col)):
            if not row_col[i]:
                row_col[i] = '.'

    def neighbors_for_water(self, row, col, piece_type) -> tuple:
        """ Finds the avaliable neighbors positions tu put water in"""
        board = self.board_representation
        rows_num = len(board)
        cols_num = len(board[0])
        adjacents_coord = ()
        directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1))

        if piece_type in ('t', 'T'):
            directions = directions[:6] + (directions[7],)
        elif piece_type in ('b', 'B'):
            directions = (directions[0],) + directions[2:]
        elif piece_type in ('m', 'M'):
            vertical_neighbors = self.adjacent_vertical_values(row, col)
            horizontal_neighbors = self.adjacent_horizontal_values(row, col)
            if '.' in vertical_neighbors or 'W' in vertical_neighbors:
                directions = (directions[0],) + (directions[1],) \
                    + (directions[2],) + (directions[5],) \
                    + (directions[6],) + (directions[7],)
            elif '.' in horizontal_neighbors or 'W' in horizontal_neighbors:
                directions = (directions[0],) + (directions[2],) \
                    + (directions[3],) + (directions[4],) \
                    + (directions[5],) + (directions[7],)
            else:
                directions = (directions[0],) + (directions[2],) \
                    + (directions[5],) + (directions[7],)
        elif piece_type in ('l', 'L'):
            directions = directions[:4] + directions[5:]
        elif piece_type in ('r', 'R'):
            directions = directions[:3] + directions[4:]

        for direction in directions:
            new_row = row + direction[0]
            new_col = col + direction[1]
            if 0 <= new_row < rows_num and 0 <= new_col < cols_num:
                adjacents_coord += ((new_row, new_col),)

        return adjacents_coord

    def beggining_check(self):
        """ Fills with water the initial rows, columns and neighbors """
        board = self.board_representation
        board_info = self.board_info
        board_occupied = self.board_occupied
        rows_num = len(board)
        cols_num = len(board[0])

        for row in range(rows_num):
            if board_info[0][row] == board_occupied[0][row]:
                self.fill_with_water(board[row])

        for col in range(cols_num):
            if board_info[1][col] == board_occupied[1][col]:
                self.fill_with_water(board[0:, col])

        for row in range(rows_num):
            for col in range(cols_num):
                piece_type = self.get_value(row, col)
                if piece_type not in (None, 'W', '.'):
                    adjacents_coord = self.neighbors_for_water(
                        row, col, piece_type)
                    for new_row, new_col in adjacents_coord:
                        if not board[new_row][new_col]:
                            board[new_row][new_col] = '.'

    def complete_boat_hints(self):
        board = self.board_representation
        board_info = self.boat_info
        horizontal_boats = np.where(board == 'L')
        vertical_boats = np.where(board == 'T')

        if np.any(horizontal_boats):
            for i in range(len(horizontal_boats[0])):
                row = horizontal_boats[0][i]
                col = horizontal_boats[1][i]
                if board[row][col+1] is None:
                    continue
                elif board[row][col+1] == 'R':
                    board_info['2piece'] -= 1
                elif board[row][col+1] == 'M' and\
                        board[row][col+2] == 'R':
                    board_info['3piece'] -= 1
                elif board[row][col+1] == 'M' and\
                        board[row][col+2] == 'M' and\
                        board[row][col+3] == 'R':
                    board_info['4piece'] -= 1

        if np.any(vertical_boats):
            for i in range(len(vertical_boats[0])):
                row = vertical_boats[0][i]
                col = vertical_boats[1][i]
                if board[row+1][col] is None:
                    continue
                elif board[row+1][col] == 'B':
                    board_info['2piece'] -= 1
                elif board[row+1][col] == 'M'\
                        and board[row+2][col] == 'B':
                    board_info['3piece'] -= 1
                elif board[row+1][col] == 'M'\
                        and board[row+2][col] == 'M'\
                        and board[row+3][col] == 'B':
                    board_info['4piece'] -= 1

    def put_piece(self, row, col, piece_type):
        board = self.board_representation
        board_occupied = self.board_occupied
        board_info = self.board_info
        board[row][col] = piece_type

        if piece_type not in ('.', 'W'):
            board_occupied[0][row] += 1
            board_occupied[1][col] += 1

            if board_occupied[0][row] == board_info[0][row]:
                self.fill_with_water(board[row])

            if board_occupied[1][col] == board_info[1][col]:
                self.fill_with_water(board[0:, col])

            adjacents_coord = self.neighbors_for_water(row, col, piece_type)
            for new_row, new_col in adjacents_coord:
                if not board[new_row][new_col]:
                    board[new_row][new_col] = '.'

    def possible_4boat_horizontal_positions(self):
        """This position refers to the top piece of the boat"""
        board = self.board_representation
        board_info = self.board_info
        board_occupied = self.board_occupied
        hint_number = self.hints
        possible_positions = []
        rows_len = cols_len = len(board)
        already_occupied = 0

        for i in range(rows_len):
            if len(possible_positions) >= 5:
                return possible_positions
            if hint_number == 0:
                if board_info[0][i] - board_occupied[0][i] >= 4:
                    for j in range(cols_len - 3):
                        if (board_info[1][j] - board_occupied[1][j] >= 1 or board[i][j] == 'L') \
                                and (board_info[1][j + 1] - board_occupied[1][j + 1] >= 1 or board[i][j + 1] == 'M')\
                                and (board_info[1][j + 2] - board_occupied[1][j + 2] >= 1 or board[i][j + 2] == 'M')\
                                and (board_info[1][j + 3] - board_occupied[1][j + 3] >= 1 or board[i][j + 3] == 'R')\
                                and board[i][j] in ('L', None) and board[i][j + 1] in ('M', None) \
                                and board[i][j + 2] in ('M', None) and board[i][j + 3] in ('R', None):
                            if board_occupied[0][i] + 4 <= board_info[0][i]:
                                possible_positions.append((i, j))
            elif board_info[0][i] != board_occupied[0][i] and board_info[0][i] >= 4:
                # -3 because we want to count with the length of the boat
                for j in range(cols_len - 3):
                    if (board_info[1][j] - board_occupied[1][j] >= 1 or board[i][j] == 'L') \
                            and (board_info[1][j + 1] - board_occupied[1][j + 1] >= 1 or board[i][j + 1] == 'M')\
                            and (board_info[1][j + 2] - board_occupied[1][j + 2] >= 1 or board[i][j + 2] == 'M')\
                            and (board_info[1][j + 3] - board_occupied[1][j + 3] >= 1 or board[i][j + 3] == 'R')\
                            and board[i][j] in ('L', None) and board[i][j + 1] in ('M', None) \
                            and board[i][j + 2] in ('M', None) and board[i][j + 3] in ('R', None):
                        if board[i][j] == 'L':
                            already_occupied += 1
                        if board[i][j + 1] == 'M':
                            already_occupied += 1
                        if board[i][j + 2] == 'M':
                            already_occupied += 1
                        if board[i][j + 3] == 'R':
                            already_occupied += 1
                        if board_occupied[0][i] - already_occupied + 4 <= board_info[0][i]:
                            possible_positions.append((i, j))
        return possible_positions

    def possible_4boat_vertical_positions(self):
        """This position refers to the top piece of the boat"""
        board = self.board_representation
        board_info = self.board_info
        board_occupied = self.board_occupied
        hint_number = self.hints
        possible_positions = []
        rows_len = cols_len = len(board)
        already_occupied = 0

        for i in range(cols_len):
            if len(possible_positions) >= 5:
                return possible_positions
            if hint_number == 0:
                if board_info[1][i] - board_occupied[1][i] >= 4:
                    for j in range(rows_len - 3):
                        if (board_info[0][j] - board_occupied[0][j] >= 1 or board[j][i] == 'T') \
                                and (board_info[0][j + 1] - board_occupied[0][j + 1] >= 1 or board[j + 1][i] == 'M')\
                                and (board_info[0][j + 2] - board_occupied[0][j + 2] >= 1 or board[j + 2][i] == 'M')\
                                and (board_info[0][j + 3] - board_occupied[0][j + 3] >= 1 or board[j + 3][i] == 'B')\
                                and board[j][i] in ('T', None) and board[j + 1][i] in ('M', None) \
                                and board[j + 2][i] in ('M', None) and board[j + 3][i] in ('B', None):
                            if board_occupied[1][i] + 4 <= board_info[1][i]:
                                possible_positions.append((j, i))
            elif board_info[1][i] != board_occupied[1][i] and board_info[1][i] >= 4:
                # -3 because we want to count with the length of the boat
                for j in range(rows_len - 3):
                    if (board_info[0][j] - board_occupied[0][j] >= 1 or board[j][i] == 'T') \
                            and (board_info[0][j + 1] - board_occupied[0][j + 1] >= 1 or board[j + 1][i] == 'M')\
                            and (board_info[0][j + 2] - board_occupied[0][j + 2] >= 1 or board[j + 2][i] == 'M')\
                            and (board_info[0][j + 3] - board_occupied[0][j + 3] >= 1 or board[j + 3][i] == 'B')\
                            and board[j][i] in ('T', None) and board[j + 1][i] in ('M', None) \
                            and board[j + 2][i] in ('M', None) and board[j + 3][i] in ('B', None):
                        if board[j][i] == 'T':
                            already_occupied += 1
                        if board[j + 1][i] == 'M':
                            already_occupied += 1
                        if board[j + 2][i] == 'M':
                            already_occupied += 1
                        if board[j + 3][i] == 'B':
                            already_occupied += 1
                        if board_occupied[1][i] - already_occupied + 4 <= board_info[1][i]:
                            possible_positions.append((j, i))

        return possible_positions

    def possible_3boat_horizontal_positions(self):
        board = self.board_representation
        board_info = self.board_info
        board_occupied = self.board_occupied
        hint_number = self.hints
        possible_positions = []
        rows_len = cols_len = len(board)
        already_occupied = 0

        for i in range(rows_len):
            if len(possible_positions) >= 6:
                return possible_positions
            if hint_number == 0:
                if board_info[0][i] - board_occupied[0][i] >= 3:
                    for j in range(cols_len - 2):
                        if (board_info[1][j] - board_occupied[1][j] >= 1 or board[i][j] == 'L') \
                                and (board_info[1][j + 1] - board_occupied[1][j + 1] >= 1 or board[i][j + 1] == 'M')\
                                and (board_info[1][j + 2] - board_occupied[1][j + 2] >= 1 or board[i][j + 2] == 'R')\
                                and board[i][j] in ('L', None) and board[i][j + 1] in ('M', None) \
                                and board[i][j + 2] in ('R', None):
                            if board_occupied[0][i] + 3 <= board_info[0][i]:
                                possible_positions.append((i, j))
            elif board_info[0][i] - board_occupied[0][i] != 0 and board_info[0][i] >= 3:
                # -2 because we want to count with the length of the boat
                for j in range(cols_len - 2):
                    if (board_info[1][j] - board_occupied[1][j] >= 1 or board[i][j] == 'L') \
                            and (board_info[1][j + 1] - board_occupied[1][j + 1] >= 1 or board[i][j + 1] == 'M')\
                            and (board_info[1][j + 2] - board_occupied[1][j + 2] >= 1 or board[i][j + 2] == 'R')\
                            and board[i][j] in ('L', None) and board[i][j + 1] in ('M', None) \
                            and board[i][j + 2] in ('R', None):
                        if board[i][j] == 'L':
                            already_occupied += 1
                        if board[i][j + 1] == 'M':
                            already_occupied += 1
                        if board[i][j + 2] == 'R':
                            already_occupied += 1
                        if board_occupied[0][i] - already_occupied + 3 <= board_info[0][i]:
                            possible_positions.append((i, j))
        return possible_positions

    def possible_3boat_vertical_positions(self):
        """This position refers to the top piece of the boat"""
        board = self.board_representation
        board_info = self.board_info
        board_occupied = self.board_occupied
        hint_number = self.hints
        possible_positions = []
        rows_len = cols_len = len(board)
        already_occupied = 0

        for i in range(cols_len):
            if len(possible_positions) >= 6:
                return possible_positions
            if hint_number == 0:
                if board_info[1][i] - board_occupied[1][i] >= 3:
                    for j in range(rows_len - 2):
                        if (board_info[0][j] - board_occupied[0][j] >= 1 or board[j][i] == 'T') \
                                and (board_info[0][j + 1] - board_occupied[0][j + 1] >= 1 or board[j + 1][i] == 'M')\
                                and (board_info[0][j + 2] - board_occupied[0][j + 2] >= 1 or board[j + 2][i] == 'B')\
                                and board[j][i] in ('T', None) and board[j + 1][i] in ('M', None) \
                                and board[j + 2][i] in ('B', None):
                            possible_positions.append((j, i))
            elif board_info[1][i] - board_occupied[1][i] != 0 and board_info[1][i] >= 3:
                # -2 because we want to count with the length of the boat
                for j in range(rows_len - 2):
                    if (board_info[0][j] - board_occupied[0][j] >= 1 or board[j][i] == 'T') \
                            and (board_info[0][j + 1] - board_occupied[0][j + 1] >= 1 or board[j + 1][i] == 'M')\
                            and (board_info[0][j + 2] - board_occupied[0][j + 2] >= 1 or board[j + 2][i] == 'B')\
                            and board[j][i] in ('T', None) and board[j + 1][i] in ('M', None) \
                            and board[j + 2][i] in ('B', None):
                        if board[j][i] == 'T':
                            already_occupied += 1
                        if board[j + 1][i] == 'M':
                            already_occupied += 1
                        if board[j + 2][i] == 'B':
                            already_occupied += 1
                        if board_occupied[1][i] - already_occupied + 3 <= board_info[1][i]:
                            possible_positions.append((j, i))
        return possible_positions

    def possible_2boat_horizontal_positions(self):
        board = self.board_representation
        board_info = self.board_info
        board_occupied = self.board_occupied
        hint_number = self.hints
        possible_positions = []
        rows_len = cols_len = len(board)
        already_occupied = 0

        for i in range(rows_len):
            if len(possible_positions) >= 3:
                return possible_positions
            if hint_number == 0:
                if board_info[0][i] - board_occupied[0][i] >= 2:
                    for j in range(cols_len - 1):
                        if (board_info[1][j] - board_occupied[1][j] >= 1 or board[i][j] == 'L') \
                                and (board_info[1][j + 1] - board_occupied[1][j + 1] >= 1 or board[i][j + 1] == 'R')\
                                and board[i][j] in ('L', None) and board[i][j + 1] in ('R', None):
                            if board_occupied[0][i] + 2 <= board_info[0][i]:
                                possible_positions.append((i, j))
            elif board_info[0][i] - board_occupied[0][i] != 0 and board_info[0][i] >= 2:
                # -1 because we want to count with the length of the boat
                for j in range(cols_len - 1):
                    if (board_info[1][j] - board_occupied[1][j] >= 1 or board[i][j] == 'L') \
                            and (board_info[1][j + 1] - board_occupied[1][j + 1] >= 1 or board[i][j + 1] == 'R')\
                            and board[i][j] in ('L', None) and board[i][j + 1] in ('R', None):
                        if board[i][j] == 'L':
                            already_occupied += 1
                        if board[i][j + 1] == 'R':
                            already_occupied += 1
                        if board_occupied[0][i] - already_occupied + 2 <= board_info[0][i]:
                            possible_positions.append((i, j))
        return possible_positions

    def possible_2boat_vertical_positions(self):
        """This position refers to the top piece of the boat"""
        board = self.board_representation
        board_info = self.board_info
        board_occupied = self.board_occupied
        hint_number = self.hints
        possible_positions = []
        rows_len = cols_len = len(board)
        already_occupied = 0

        for i in range(cols_len):
            if len(possible_positions) >= 3:
                return possible_positions
            if hint_number == 0:
                if board_info[1][i] - board_occupied[1][i] >= 2:
                    for j in range(rows_len - 1):
                        if (board_info[0][j] - board_occupied[0][j] >= 1 or board[j][i] == 'T') \
                                and (board_info[0][j + 1] - board_occupied[0][j + 1] >= 1 or board[j + 1][i] == 'B')\
                                and board[j][i] in ('T', None) and board[j + 1][i] in ('B', None):
                            if board_occupied[1][i] + 2 <= board_info[1][i]:
                                possible_positions.append((j, i))
            elif board_info[1][i] != board_occupied[1][i] and board_info[1][i] >= 2:
                # -1 because we want to count with the length of the boat
                for j in range(rows_len - 1):
                    if (board_info[0][j] - board_occupied[0][j] >= 1 or board[j][i] == 'T') \
                            and (board_info[0][j + 1] - board_occupied[0][j + 1] >= 1 or board[j + 1][i] == 'B')\
                            and board[j][i] in ('T', None) and board[j + 1][i] in ('B', None):
                        if board[j][i] == 'T':
                            already_occupied += 1
                        if board[j + 1][i] == 'B':
                            already_occupied += 1
                        if board_occupied[1][i] - already_occupied + 2 <= board_info[1][i]:
                            possible_positions.append((j, i))
        return possible_positions

    def possible_1boat_positions(self):
        board = self.board_representation
        board_info = self.board_info
        board_occupied = self.board_occupied
        possible_positions = []
        rows_len = cols_len = len(board)
        for i in range(rows_len):
            if len(possible_positions) >= 3:
                return possible_positions
            if board_info[0][i] - board_occupied[0][i] >= 1:
                for j in range(cols_len):
                    if board_info[1][j] - board_occupied[1][j] >= 1 \
                            and board[i][j] is None:
                        possible_positions.append((i, j))

        return possible_positions

    def place_4boat_horizontally(self, row, col):
        board_rep = self.board_representation.copy()
        boat_info = self.boat_info.copy()
        board_occupied = self.board_occupied.copy()
        board_info = self.board_info
        hints = self.hints
        new_board = Board(board_rep, board_info,
                          board_occupied, boat_info, hints)
        for i in range(4):
            if self.get_value(row, col + i) is None:
                if i == 0:
                    Board.put_piece(new_board, row, col + i, 'l')
                elif i == 1:
                    Board.put_piece(new_board, row, col + i, 'm')
                elif i == 2:
                    Board.put_piece(new_board, row, col + i, 'm')
                elif i == 3:
                    Board.put_piece(new_board, row, col + i, 'r')

        new_board.boat_info["4piece"] -= 1

        return new_board

    def place_4boat_vertically(self, row, col):
        board_rep = self.board_representation.copy()
        boat_info = self.boat_info.copy()
        board_occupied = self.board_occupied.copy()
        board_info = self.board_info
        hints = self.hints
        new_board = Board(board_rep, board_info,
                          board_occupied, boat_info, hints)
        for i in range(4):
            if self.get_value(row + i, col) is None:
                if i == 0:
                    Board.put_piece(new_board, row + i, col, 't')
                elif i == 1:
                    Board.put_piece(new_board, row + i, col, 'm')
                elif i == 2:
                    Board.put_piece(new_board, row + i, col, 'm')
                elif i == 3:
                    Board.put_piece(new_board, row + i, col, 'b')

        new_board.boat_info["4piece"] -= 1

        return new_board

    def place_3boat_horizontally(self, row, col):
        board_rep = self.board_representation.copy()
        boat_info = self.boat_info.copy()
        board_occupied = self.board_occupied.copy()
        board_info = self.board_info
        hints = self.hints
        new_board = Board(board_rep, board_info,
                          board_occupied, boat_info, hints)
        for i in range(3):
            if self.get_value(row, col + i) is None:
                if i == 0:
                    Board.put_piece(new_board, row, col + i, 'l')
                elif i == 1:
                    Board.put_piece(new_board, row, col + i, 'm')
                elif i == 2:
                    Board.put_piece(new_board, row, col + i, 'r')

        new_board.boat_info["3piece"] -= 1

        return new_board

    def place_3boat_vertically(self, row, col):
        board_rep = self.board_representation.copy()
        boat_info = self.boat_info.copy()
        board_occupied = self.board_occupied.copy()
        board_info = self.board_info
        hints = self.hints
        new_board = Board(board_rep, board_info,
                          board_occupied, boat_info, hints)
        for i in range(3):
            if self.get_value(row + i, col) is None:
                if i == 0:
                    Board.put_piece(new_board, row + i, col, 't')
                elif i == 1:
                    Board.put_piece(new_board, row + i, col, 'm')
                elif i == 2:
                    Board.put_piece(new_board, row + i, col, 'b')

        new_board.boat_info["3piece"] -= 1

        return new_board

    def place_2boat_horizontally(self, row, col):
        board_rep = self.board_representation.copy()
        boat_info = self.boat_info.copy()
        board_occupied = self.board_occupied.copy()
        board_info = self.board_info
        hints = self.hints
        new_board = Board(board_rep, board_info,
                          board_occupied, boat_info, hints)
        for i in range(2):
            if self.get_value(row, col + i) is None:
                if i == 0:
                    Board.put_piece(new_board, row, col + i, 'l')
                elif i == 1:
                    Board.put_piece(new_board, row, col + i, 'r')

        new_board.boat_info["2piece"] -= 1

        return new_board

    def place_2boat_vertically(self, row, col):
        board_rep = self.board_representation.copy()
        boat_info = self.boat_info.copy()
        board_occupied = self.board_occupied.copy()
        board_info = self.board_info
        hints = self.hints
        new_board = Board(board_rep, board_info,
                          board_occupied, boat_info, hints)
        for i in range(2):
            if self.get_value(row + i, col) is None:
                if i == 0:
                    Board.put_piece(new_board, row + i, col, 't')
                elif i == 1:
                    Board.put_piece(new_board, row + i, col, 'b')

        new_board.boat_info["2piece"] -= 1

        return new_board

    def place_1boat(self, row, col):
        board_rep = self.board_representation.copy()
        boat_info = self.boat_info.copy()
        board_occupied = self.board_occupied.copy()
        board_info = self.board_info
        hints = self.hints
        new_board = Board(board_rep, board_info,
                          board_occupied, boat_info, hints)
        if self.get_value(row, col) is None:
            Board.put_piece(new_board, row, col, 'c')

        new_board.boat_info["1piece"] -= 1

        return new_board


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        super().__init__(BimaruState(board))

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        board = state.board
        possible_actions = []
        if state.board.boat_info["4piece"] == 1:
            for avaliable_spot in board.possible_4boat_horizontal_positions():
                possible_actions.append(["4boat_horizontal", avaliable_spot])
            for avaliable_spot in board.possible_4boat_vertical_positions():
                possible_actions.append(["4boat_vertical", avaliable_spot])
        elif state.board.boat_info["3piece"] >= 1:
            for avaliable_spot in board.possible_3boat_horizontal_positions():
                possible_actions.append(["3boat_horizontal", avaliable_spot])
            for avaliable_spot in board.possible_3boat_vertical_positions():
                possible_actions.append(["3boat_vertical", avaliable_spot])
        elif state.board.boat_info["2piece"] >= 1:
            for avaliable_spot in board.possible_2boat_horizontal_positions():
                possible_actions.append(["2boat_horizontal", avaliable_spot])
            for avaliable_spot in board.possible_2boat_vertical_positions():
                possible_actions.append(["2boat_vertical", avaliable_spot])
        elif state.board.boat_info["1piece"] >= 1:
            for avaliable_spot in board.possible_1boat_positions():
                possible_actions.append(["1boat", avaliable_spot])

        return possible_actions

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        action_name = action[0]
        row, col = action[1][0], action[1][1]
        if action_name == "4boat_horizontal":
            return state.place_4boat_horizontally(row, col)
        elif action_name == "4boat_vertical":
            return state.place_4boat_vertically(row, col)
        elif action_name == "3boat_horizontal":
            return state.place_3boat_horizontally(row, col)
        elif action_name == "3boat_vertical":
            return state.place_3boat_vertically(row, col)
        elif action_name == "2boat_horizontal":
            return state.place_2boat_horizontally(row, col)
        elif action_name == "2boat_vertical":
            return state.place_2boat_vertically(row, col)
        elif action_name == "1boat":
            return state.place_1boat(row, col)

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        board_info = state.board.board_info
        board_occupied = state.board.board_occupied
        boat_info = state.board.boat_info

        for indiviual_boat_info in boat_info.values():
            if indiviual_boat_info != 0:
                return False

        return np.array_equal(board_info, board_occupied)

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        pass


if __name__ == "__main__":

    board = Board.parse_instance()
    board.beggining_check()
    if board.hints > 1:
        board.complete_boat_hints()
    problem = Bimaru(board)
    goal_node = depth_first_tree_search(problem)
    goal_node.state.board.print()
