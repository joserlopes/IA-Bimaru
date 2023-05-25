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


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(self, board_representation: list, board_info: list, board_occupied: list, boat_info: dict):
        self.board_representation = board_representation
        self.board_info = board_info
        self.board_occupied = board_occupied
        self.boat_info = boat_info

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

        boat_info = {"4_piece": 1, "3_piece": 2, "2_piece": 3, "1_piece": 4}

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
                    boat_info["1_piece"] -= 1

        return Board(board, board_info, board_occupied, boat_info)

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
                    adjacents_coord = self.neighbors_for_water(row, col, piece_type)
                    for new_row, new_col in adjacents_coord:
                        if not board[new_row][new_col]:
                            board[new_row][new_col] = '.'

    def put_piece(self, row, col, piece_type):
        board = self.board_representation.copy()
        board_occupied = self.board_occupied.copy()
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

        return Board(board, board_info, board_occupied)


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        super().__init__(BimaruState(board))

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        if state.board.boat_info["4_piece"] == 1:
            pass

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        pass

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        pass

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        # ver a cena de ser o número de barcos que ainda falta por


if __name__ == "__main__":

    board = Board.parse_instance()
    board.beggining_check()
    print(board.board_info)
    print(board.board_occupied)
    print(board.board_representation)
    print(board.boat_info)

    # TODO
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
