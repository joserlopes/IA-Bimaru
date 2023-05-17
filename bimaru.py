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

    def put_1_piece(self):
        board = self.board
        row, col = board.first_avaliable_spot()
        board.put_piece(row, col, 'c')

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(self, board_representation: list, avaliable_boats: dict):
        self.board_representation = board_representation
        self.avaliable_boats = avaliable_boats

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        # + 2 to count with the two hint-related lines
        return self.board_representation[row + 2][col]

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        board = self.board_representation
        adjacents = ()
        rows_num = len(board) - 2
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
        # The first row of the board will be related to the line and the second to the columns
        board = [[None for x in range(10)] for y in range(12)]
        # Transform the 2D list into an np.array to easily acess the columns
        board = np.array(board)
        # A dictionary with the avaliable boats of each type
        avaliable_boats = {"4_piece": 1, "3_piece": 2, "2_piece": 3, "1_piece": 4}
        # Read everything until EOF is reached
        while line := sys.stdin.readline().split():
            if line[0] == "ROW":
                for row_hint, index in zip(line[1:], range(10)):
                    board[0][index] = int(row_hint)
            elif line[0] == "COLUMN":
                for col_hint, index in zip(line[1:], range(10)):
                    board[1][index] = int(col_hint)
            elif line[0] == "HINT":
                row = int(line[1])
                col = int(line[2])
                hint = line[3]
                board[row + 2][col] = hint
                # "Occupy" the spot
                if hint != 'W':
                    board[0][row] -= 1
                    board[1][col] -= 1
                if hint == 'C':
                    avaliable_boats["1_piece"] -= 1
        return Board(board, avaliable_boats)

    def print(self):
        for i in range(len(self.board_representation) - 2):
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
        board[row + 2][col] = piece_type
        adjacents_coord = self.neighbours(row, col, piece_type)
        for new_row, new_col in adjacents_coord:
            if piece_type == 'c':
                print(new_row, new_col)
            if not board[new_row][new_col]:
                board[new_row][new_col] = '.'

    def neighbours(self, row, col, piece_type) -> tuple:
        """ Finds the avaliable neighbours positions """
        board = self.board_representation
        row += 2
        rows_num = len(board)
        cols_num = len(board[0])
        adjacens_coord = ()
        directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1))
        for direction in directions:
            new_row = row + direction[0]
            new_col = col + direction[1]
            if 2 <= new_row < rows_num and 0 <= new_col < cols_num:
                adjacens_coord += ((new_row, new_col),)
        return adjacens_coord

    def beggining_check(self):
        """ Fills with water the initial rows, columns and neighbours """
        # TODO POR AS ÁGUAS NOS VIZINHOS DE CADA PEÇA INICIAL
        board = self.board_representation
        rows_num = len(board) - 2
        cols_num = len(board[0])

        for row in range(rows_num):
            if board[0][row] == 0:
                self.fill_with_water(board[row + 2])

        for col in range(cols_num):
            if board[1][col] == 0:
                self.fill_with_water(board[2:, col])

        for row in range(rows_num):
            for col in range(cols_num):
                piece_type = self.get_value(row, col)
                if piece_type == 'C':
                    adjacents_coord = self.neighbours(row, col, piece_type)
                    for new_row, new_col in adjacents_coord:
                        if not board[new_row][new_col]:
                            board[new_row][new_col] = '.'

    def first_avaliable_spot(self):
        board = self.board_representation
        rows_num = len(board) - 2
        cols_num = len(board[0])

        for row in range(rows_num):
            for col in range(cols_num):
                if not self.get_value(row, col):
                    return (row, col)
        return None


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = BimaruState(board)

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO
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
        # TODO
        line_sum = 0
        column_sum = 0
        # Line check
        # - 2 to count for the + 2 after
        for i in range(len(state.board.board_representation) - 2):
            for j in range(len(state.board.board_representation[0])):
                if state.board.board_representation[i + 2][j] not in ('.', 'W'):
                    line_sum += 1
            if state.board.board_representation[0][i] != line_sum:
                return False
            line_sum = 0

        # Column check
        # - 2 to count for the + 2 after
        for j in range(len(state.board.board_representation) - 2):
            for i in range(len(state.board.board_representation[0])):
                if state.board.board_representation[i + 2][j] not in ('.', 'W'):
                    column_sum += 1
            if state.board.board_representation[1][j] != column_sum:
                return False
            column_sum = 0

        return True

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":

    board = Board.parse_instance()
    board.beggining_check()
    print(board.board_representation)

    problem = Bimaru(board)

    initial_state = BimaruState(board)

    initial_state.put_1_piece()
    print(initial_state.board.board_representation)

    initial_state.put_1_piece()
    print(initial_state.board.board_representation)
    # TODO
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
