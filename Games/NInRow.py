#python 3.8

from string import ascii_lowercase
from typing import List

class NInRow(object):
    def __init__(self, n: int = 3, grid_size: int = 3):
        self.grid_size = grid_size
        self.n = n
        self.current_player = 1
        self.winner = None

        self.board = self.constructBoard()

    def makeMove(self, coordinates: List[int], player: int = None) -> int:
        player = self.current_player if not player else player
        reward = None

        if(self.winner == None and self.legalMove(coordinates)):
            greatest_row = self.greatestRow(coordinates, player)
            #place players stone
            self.board[coordinates[1]][coordinates[0]] = player

            #winning row
            if(greatest_row >= self.n):
                self.winner = player
                reward = player

            #if new player can move
            elif(self.canMove()):
                #switch active player
                self.current_player = -player

            #neither player can move -> draw
            else:
                self.winner = 0
                self.current_player = 0
                reward = 0
        return reward

    def constructBoard(self) -> List[List[int]]:
        return [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]

    def validMoves(self, player: int = None) -> List[int]:
        possible_moves = []

        if(self.winner == None):
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    if (self.legalMove([x, y])):
                        possible_moves.append([x, y])

        return possible_moves

    def canMove(self, player = None) -> bool:
        possible_moves = self.validMoves()
        return False if(len(possible_moves) == 0) else True

    def legalMove(self, coordinates: List[int], player = None) -> bool:
        return False if(self.board[coordinates[1]][coordinates[0]] != 0) else True

    def greatestRow(self, coordinates: List[int], player: int = None) -> int:
        neighbours, captured_squares = [], []
        greatest_path = 0
        player = self.current_player if not player else player

        #find occupied squares neighbouring coordinates
        for i in range(max(0, coordinates[0]-1), min(coordinates[0]+2, self.grid_size)):
            for j in range(max(0, coordinates[1]-1), min(coordinates[1]+2, self.grid_size)):
                if self.board[j][i] != 0:
                    neighbours.append([i,j])

        #find diagonal, horizontal and vertical capture paths
        for neighbour in neighbours:
            #neighbouring square doesnt belong to player
            if(self.board[neighbour[1]][neighbour[0]] != player):
                continue

            #direction of neighbour from coordinates
            delta_x = neighbour[0] - coordinates[0]
            delta_y = neighbour[1] - coordinates[1]

            new_x, new_y = neighbour[0], neighbour[1]

            #step through indicated direction and store captured squares in path
            path = [coordinates]
            
            while 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                #path ended with an empty square
                if (self.board[new_y][new_x] == 0):
                    break

                #path ended with a opponent square
                if self.board[new_y][new_x] != player:
                    #append path coordinas to captured_squares
                    captured_squares += path
                    break

                path.append([new_x, new_y])
                
                new_x += delta_x
                new_y += delta_y
                
            if(len(path) > greatest_path):
                greatest_path = len(path)

        return greatest_path
        
    def __repr__(self) -> str:
        representation = {-1: u'\u25CB', 1: u'\u25CF', 0: ' '}
        
        horizontal_line = f'   {4 * self.grid_size * "-"}'
        x_label = '     '

        for i in ascii_lowercase[:self.grid_size]:
            x_label += f'{i} | '
            
        s = f'{x_label}\n{horizontal_line}\n'
        
        for idx, i in enumerate(self.board):
            row = '{0:2} |'.format(idx + 1)
            for j in i:
                row += f' {representation[j]} |'
                
            s += f'{row}\n{horizontal_line}\n'

        s += f'\n{"Current player" if self.winner == None else "Winner"}: {representation[self.current_player] if self.winner == None else ("draw" if self.winner == 0 else representation[self.winner])}'
        return s

def _test():
    game = NInRow()
    game.makeMove([1,1])
    print(game)
    game.makeMove([1,0])
    print(game)
    game.makeMove([0,1])
    print(game)
    game.makeMove([0,0])
    print(game)
    game.makeMove([2,1])
    print(game)

if __name__ == '__main__':
    _test()
