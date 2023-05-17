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

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(self, board_representation: list):
        self.board_representation = board_representation

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        # + 2 to count with the two hint-related lines
        return self.board_representation[row + 2][col]

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        if 0 < row < 9:
            return (self.get_value(row - 1, col), self.get_value(row + 1, col))
        elif row == 0:
            return (None, self.get_value(row + 1, col))
        elif row == 9:
            return (self.get_value(row - 1, col), None)
        else:
            return (None, None)

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        if 0 < col < 9:
            return (self.get_value(row, col - 1), self.get_value(row, col + 1))
        elif col == 0:
            return (None, self.get_value(row, col + 1))
        elif col == 9:
            return (self.get_value(row, col - 1), None)
        else:
            return (None, None)

    def adjacent_diagonal_values_ascending(self, row: int, col: int) -> (str, str):
        if 0 < row < 9 and 0 < col < 9:
            return (self.get_value(row + 1, col - 1), self.get_value(row - 1, col + 1))
        # Only has the right diagonal
        elif (col == 0 and row != 0) or (row == 9 and col != 9):
            return (None, self.get_value(row - 1, col + 1))
        # Only has the left diagonal
        elif (col == 9 and row != 9) or (row == 0 and col != 0):
            return (self.get_value(row + 1, col - 1), None)
        else:
            return (None, None)

    def adjacent_diagonal_values_descending(self, row: int, col: int) -> (str, str):
        if 0 < row < 9 and 0 < col < 9:
            return (self.get_value(row - 1, col - 1), self.get_value(row + 1, col + 1))
        # Only has the right diagonal
        elif (col == 0 and row != 0) or (row == 0 and col != 9):
            return (None, self.get_value(row + 1, col + 1))
        # Only has the left diagonal
        elif (col == 9 and row != 9) or (row == 0 and col != 0):
            return (self.get_value(row - 1, col - 1), None)
        else:
            return (None, None)

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 bimaru.py < input_T01

            > from sys import stdin
            > line = stdin.readline().split()
        """
        # The first "line" of the board will be related to the line and the second to the columns
        board = [[None for x in range(10)] for y in range(12)]
        # Transform the 2D list into an np.array to easily acess the columns
        board = np.array(board)
        # Read everything until a EOF is reached
        while line := sys.stdin.readline().split():
            if line[0] == "ROW":
                for row_hint, index in zip(line[1:], range(10)):
                    board[0][index] = int(row_hint)
                    if int(row_hint) == 0:
                        # index + 2 because that's where the actual board starts
                        Board.fill_with_water(board[index + 2])
            elif line[0] == "COLUMN":
                for col_hint, index in zip(line[1:], range(10)):
                    board[1][index] = int(col_hint)
                    if int(col_hint) == 0:
                        # 2: to count with the first two lines
                        Board.fill_with_water(board[2:, index])
            elif line[0] == "HINT":
                row = int(line[1])
                col = int(line[2])
                hint = line[3]
                # + 2 to count with the two hint-related lines
                board[row + 2][col] = hint
                if hint != 'W':
                    board[0][row] -= 1
                    board[1][col] -= 1
                # If the piece made a line count go to zero fill it with water
                if board[0][row] == 0:
                    Board.fill_with_water(board[row + 2])
                # If the piece made a column count go to zero fill it with water
                if board[1][col] == 0:
                    print(board[2:, col])
                    Board.fill_with_water(board[2:, col])
                print(board)
        return Board(board)

    def print(self):
        for i in range(len(self.board_representation) - 2):
            for j in range(len(self.board_representation[0])):
                if not self.get_value(i, j):
                    print(".", end="")
                else:
                    print(f"{self.get_value(i, j)}", end="")
            print("")

    def fill_with_water(row_col):
        """ Fills a given row or column with water on avaliable spots."""
        for i in range(len(row_col)):
            if not row_col[i]:
                row_col[i] = '.'


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = BimaruState(board)

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO
        return ["put_submarine", "put_2_boat_horizontal"]

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        if action == "put_submarine":
            return state.put_submarine()
        elif action == "put_2_boat_horizontal":
            return state.put_2_piece_horizontal()

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        # TODO
        pass

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":

    board = Board.parse_instance()
    # TODO:
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
