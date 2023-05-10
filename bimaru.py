# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 00000 Nome1
# 00000 Nome2

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
        else:
            return (self.get_value(row - 1, col), None)
    
    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        if 0 < col < 9:
            return (self.get_value(row, col - 1), self.get_value(row, col + 1)) 
        elif col == 0:
            return (None, self.get_value(row, col + 1)) 
        else:
            return(self.get_value(row, col - 1), None)

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
        # Read everything until a EOF is reached
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
                # + 2 to count with the two hint-related lines
                board[row + 2][col] = hint
        return Board(board)

    # TODO: outros metodos da classe


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        # TODO
        pass

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
        # TODO
        pass

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

    print(board.adjacent_vertical_values(3, 3))
    print(board.adjacent_horizontal_values(3, 3))
    print(board.adjacent_vertical_values(9, 0))
    print(board.adjacent_horizontal_values(0, 9))
    # TODO:
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    pass