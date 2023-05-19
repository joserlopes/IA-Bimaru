# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 80:
# 103938 José António Lopes
# 104139 Rodrigo Manuel Friães

import sys
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)

import numpy as np


class BimaruState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = BimaruState.state_id
        BimaruState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    def put_water_piece(self):
        board = self.board
        first_avaliable_spot = board.first_avaliable_spot('.')
        if first_avaliable_spot:
            row, col = first_avaliable_spot[0], first_avaliable_spot[1]
            board.put_piece(row, col, '.')
        return BimaruState(board)

    def put_center_piece(self):
        board = self.board
        if board.first_avaliable_spot('c'):
            row, col = board.first_avaliable_spot('c')
            board.put_piece(row, col, 'c')
        return BimaruState(board)

    def put_left_piece(self):
        board = self.board
        if board.first_avaliable_spot('l'):
            row, col = board.first_avaliable_spot('l')
            board.put_piece(row, col, 'l')
        return BimaruState(board)

    def put_right_piece(self):
        board = self.board
        if board.first_avaliable_spot('r'):
            row, col = board.first_avaliable_spot('r')
            board.put_piece(row, col, 'r')
        return BimaruState(board)

    def put_top_piece(self):
        board = self.board
        if board.first_avaliable_spot('t'):
            row, col = board.first_avaliable_spot('t')
            board.put_piece(row, col, 't')
        return BimaruState(board)

    def put_bottom_piece(self):
        board = self.board
        if board.first_avaliable_spot('b'):
            row, col = board.first_avaliable_spot('b')
            board.put_piece(row, col, 'b')
        return BimaruState(board)

    def put_middle_piece(self):
        board = self.board
        first_avaliable_spot = board.first_avaliable_spot('m')
        if first_avaliable_spot:
            row, col = first_avaliable_spot[0], first_avaliable_spot[1]
            board.put_piece(row, col, 'm')
        return BimaruState(board)


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(self, board_representation: list, board_info: list, board_occupied:list, avaliable_boats: dict):
        self.board_representation = board_representation
        self.board_info = board_info
        self.avaliable_boats = avaliable_boats
        self.board_occupied = board_occupied

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

    def adjacent_diagonal_values_ascending(self, row: int, col: int) -> (str, str):
        board = self.board_representation
        adjacents = ()
        cols_num = len(board[0])
        rows_num = len(board)
        for i in ((1, -1), (-1, 1)):
            new_row = row + i[0]
            new_col = col + i[1]
            if 0 <= new_row < rows_num and 0 <= new_col < cols_num:
                adjacents += (self.get_value(new_row, new_col),)
            else:
                adjacents += (None,)
        return adjacents

    def adjacent_diagonal_values_descending(self, row: int, col: int) -> (str, str):
        board = self.board_representation
        adjacents = ()
        cols_num = len(board[0])
        rows_num = len(board)
        for i in ((-1, -1), (1, 1)):
            new_row = row + i[0]
            new_col = col + i[1]
            if 0 <= new_row < rows_num and 0 <= new_col < cols_num:
                adjacents += (self.get_value(new_row, new_col),)
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
        board = np.array(board)
        # The structure that holds the information relative
        # to the rows and columns. The first row refers to the rows and
        # the second to the columns
        board_info = [[None for x in range(10)] for y in range(2)]
        board_occupied = [[0 for x in range(10)] for y in range(2)]
        # A dictionary with the avaliable boats of each type
        avaliable_boats = {"4_piece": 1, "3_piece": 2, "2_piece": 3, "1_piece": 4}
        # Read everything until EOF is reached
        while line := sys.stdin.readline().split():
            if line[0] == "ROW":
                for row_board_info, index in zip(line[1:], range(10)):
                    board_info[0][index] = int(row_board_info)
            elif line[0] == "COLUMN":
                for col_board_info, index in zip(line[1:], range(10)):
                    board_info[1][index] = int(col_board_info)
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
                    avaliable_boats["1_piece"] -= 1
        return Board(board, board_info, board_occupied, avaliable_boats)

    def print(self):
        for i in range(len(self.board_representation)):
            for j in range(len(self.board_representation[0])):
                print(f"{self.get_value(i, j)}", end="")
            print("")

    def fill_with_water(self, row_col):
        """ Fills a given row or column with water on avaliable spots."""
        for i in range(len(row_col)):
            if not row_col[i]:
                row_col[i] = '.'

    def put_piece(self, row, col, piece_type):
        board = self.board_representation
        board_occupied = self.board_occupied
        board_info = self.board_info
        board[row][col] = piece_type

        if piece_type not in ('.', 'W'):
            board_occupied[0][row] += 1
            board_occupied[0][col] += 1

            if board_occupied[0][row] == board_info[0][row]:
                self.fill_with_water(board[row])

            if board_occupied[1][col] == board_info[1][col]:
                self.fill_with_water(board[0:, col])

            adjacents_coord = self.neighbours_for_water(row, col, piece_type)
            for new_row, new_col in adjacents_coord:
                if not board[new_row][new_col]:
                    board[new_row][new_col] = '.'

    def neighbours_for_water(self, row, col, piece_type) -> tuple:
        """ Finds the avaliable neighbours positions tu put water in"""
        board = self.board_representation
        rows_num = len(board)
        cols_num = len(board[0])
        adjacens_coord = ()
        directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1))

        if piece_type in ('t', 'T'):
            directions = directions[:6] + (directions[7],)
        elif piece_type in ('b', 'B'):
            directions = (directions[0],) + directions[2:]
        elif piece_type in ('m', 'M'):
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
                adjacens_coord += ((new_row, new_col),)
        return adjacens_coord

    def beggining_check(self):
        """ Fills with water the initial rows, columns and neighbours """
        # TODO POR AS ÁGUAS NOS VIZINHOS DE CADA PEÇA INICIAL
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
                    adjacents_coord = self.neighbours_for_water(row, col, piece_type)
                    for new_row, new_col in adjacents_coord:
                        if not board[new_row][new_col]:
                            board[new_row][new_col] = '.'

    def first_avaliable_spot(self, piece_type):
        board = self.board_representation
        rows_num = len(board)
        cols_num = len(board[0])

        for row in range(rows_num):
            for col in range(cols_num):
                if not self.get_value(row, col):
                    if self.is_valid_position(row, col, piece_type):
                        return (row, col)
        return None

    def neighbours(self, row, col) -> tuple:
        board = self.board_representation
        rows_num = len(board)
        cols_num = len(board[0])
        adjacens_coord = ()
        directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1))

        for direction in directions:
            new_row = row + direction[0]
            new_col = col + direction[1]
            if 0 <= new_row < rows_num and 0 <= new_col < cols_num:
                adjacens_coord += ((new_row, new_col),)
            else:
                adjacens_coord += ((None, None),)

        return adjacens_coord

    def is_valid_position(self, row, col, piece_type):
        neighbour_values = []

        adjacents_coord = self.neighbours(row, col)
        # A list with the values of all neighbouring positions
        print(row, col)
        for new_row, new_col in adjacents_coord:
            if new_row is not None and new_col is not None:
                neighbour_values.append(self.get_value(new_row, new_col))
            else:
                neighbour_values.append('.')

        if piece_type == 'c':
            if (element is None or element == '.'
                   for element in neighbour_values):
                return True
        elif piece_type == 'l':
            if (element is None or element == '.'
                   for element in neighbour_values[:4]) \
                       and (element is None or element == '.'
                               for element in neighbour_values[5:]) \
                       and neighbour_values[4] in (None, 'M', 'm', 'r', 'R'):
                return True
        elif piece_type == 'r':
            if (element is None or element == '.'
                   for element in neighbour_values[:3]) \
                       and (element is None or element == '.'
                               for element in neighbour_values[4:]) \
                       and neighbour_values[3] in (None, 'M', 'm', 'l', 'L'):
                return True
        elif piece_type == 't':
            if row == 9:
                return False
            elif (element is None or element == '.'
                  for element in neighbour_values[:6]) \
                and neighbour_values[7] in (None, '.') \
                    and neighbour_values[6] in (None, 'M', 'm', 'b', 'B'):
                return True
        elif piece_type == 'b':
            if row == 0:
                return False
            elif (element is None or element == '.'
                    for element in neighbour_values[2:]) \
                       and neighbour_values[0] in (None, '.') \
                       and neighbour_values[1] in (None, 'M', 'm', 't', 'T'):
                return True
        elif piece_type == 'm':
            if neighbour_values[0] in (None, '.') \
                and neighbour_values[2] in (None, '.') \
                and neighbour_values[5] in (None, '.') \
                and neighbour_values[7] in (None, '.') \
                and neighbour_values[1] in (None, 't', 'T', 'm', 'M') \
                and neighbour_values[3] in (None, 'l', 'L', 'm', 'M') \
                and neighbour_values[4] in (None, 'r', 'R', 'm', 'M') \
                    and neighbour_values[6] in (None, 'b', 'B', 'm', 'M'):
                return True
        elif piece_type == '.':
            return True

        return False


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = BimaruState(board)

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        return ["put_center_piece", "put_left_piece", "put_right_piece",
                "put_top_piece", "put_bottom_piece", "put_middle_piece"]

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        if action == "put_water_piece":
            return state.put_water_piece()
        elif action == "put_center_piece":
            return state.put_center_piece()
        elif action == "put_left_piece":
            return state.put_left_piece()
        elif action == "put_right_piece":
            return state.put_right_piece()
        elif action == "put_top_piece":
            return state.put_top_piece()
        elif action == "put_bottom_piece":
            return state.put_bottom_piece()
        elif action == "put_middle_piece":
            return state.put_middle_piece()
        else:
            return NotImplementedError

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        board_info = state.board.board_info
        board_occupied = state.board.board_occupied

        if board_info == board_occupied:
            return True

        return False

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":

   # board_completed = np.array([['T','.','.','.','.','.','t','.','.','.',],
    #                ['b','.','.','.','.','.','M','.','.','t'],
     #               ['.','.','.','.','.','.','b','.','.','m'],
      #              ['.','.','C','.','.','.','.','.','.','m'],
       #             ['c','.','.','.','.','.','.','c','.','b'],
        #            ['.','.','.','.','.','.','.','.','.','.'],
         #           ['W','.','.','.','t','.','.','.','.','.'],
          #          ['t','.','.','.','b','.','.','.','t','.'],
           #         ['m','.','.','.','.','.','.','.','B','.'],
            #        ['b','.','.','.','.','C','.','.','.','.',]])


    # board = Board(board_completed, [[1, 2, 2, 1, 3, 0, 1, 3, 1, 1], [5, 0, 0, 0, 2, 0, 2, 1, 1, 4]], [[1, 2, 2, 1, 3, 0, 1, 3, 1, 1], [5, 0, 0, 0, 2, 0, 2, 1, 1, 4]],  {'4_piece': 1, '3_piece': 2, '2_piece': 3, '1_piece': 2})

    # problem = Bimaru(board)
    # state = BimaruState(board)
    # print("Is goal?", problem.goal_test(state))

    # print(board.board_info)
    # print(board.board_representation)
    # print(board.avaliable_boats)

    board = Board.parse_instance()
    board.beggining_check()
    print(board.board_info)
    print(board.board_occupied)
    print(board.board_representation)

    problem = Bimaru(board)

    initial_state = BimaruState(board)
    result_state1 = problem.result(initial_state, "put_middle_piece")

    print(result_state1.board.board_representation)
    # TODO
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
