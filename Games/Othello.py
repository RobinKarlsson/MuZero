#python 3.8

from string import ascii_lowercase
from re import match
from typing import List

from ctypes import CDLL
from os.path import abspath



class Othello(object):
    def __init__(self, grid_size: int = 8):
        if(grid_size % 2 != 0 or grid_size < 4):
            raise Exception(f'Grid size need to be an even integer greater than 2. Was {grid_size}')

        self.grid_size = grid_size
        self.winner = None
        self.current_player = 1

        self.board = self.constructBoard()

    #process user input
    # in s - string representing moves, example 'f4 f3 d6'
    def play(self, s: str):
        for move in s.split():
            if(self.winner != None):
                break

            #map input string to x,y coordinates
            pattern = r'([a-{}])([0-9]+)(F?)'.format(ascii_lowercase[self.grid_size - 1])
            validinput = match(pattern, move.strip())

            if(validinput):
                y = int(validinput.group(2)) - 1
                x = ascii_lowercase.index(validinput.group(1))

                #coordinates outside board
                if(x < 0 or y < 0 or x > self.grid_size or y > self.grid_size):
                    return False

            #ensure move is legal
            if(self.legalMove([x, y], self.current_player)):
                #execute move
                self.makeMove([x, y], self.current_player)

    #get score for each player
    #return dict with squares owned by player 1, unclaimed and player -1
    def getScore(self) -> dict:
        score = {1:0, 0:0, -1:0}

        #count stones
        for row in self.board:
            for square in row:
                score[square] += 1

        return score

    #check if player can move
    #return True if player can move, otherwise False
    def canMove(self, player: int = None) -> bool:
        possible_moves = self.validMoves(self.current_player if not player else player)
        return False if(len(possible_moves) == 0) else True

    #perform a move, assumes the move is legal
    #in int x,y - coordinates to place a stone at
    def makeMove(self, coordinates: List[int], player: int = None) -> int:
        player = self.current_player if not player else player
            
        if(self.winner == None):
            #squares taken with move
            paths = self.getCapturePaths(coordinates)

            #place players stone at captures squares
            self.board[coordinates[1]][coordinates[0]] = player
            for square in paths:
                self.board[square[1]][square[0]] = player

            #if new player can move
            if(self.canMove(-player)):
                #switch active player
                self.current_player = -player

            #new player cant move -> gameover
            else:
                #determine victor
                score = self.getScore()
                
                if(score[1] > score[-1]):
                    self.winner = 1
                elif(score[1] < score[-1]):
                    self.winner = -1
                else:
                    self.winner = 0

                self.current_player = 0

        reward = 0
        if(self.winner):
            if(self.winner == player):
                reward = 1
            elif(self.winner == 0):
                pass
            else:
                reward = -1

        return reward
            
    #returns list of legal moves available for player
    def validMoves(self, player: int = None) -> List[int]:
        player = self.current_player if not player else player
        possible_moves = []

        if(self.winner == None):
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    if (self.legalMove([x, y], player)):
                        possible_moves.append([x, y])

        return possible_moves

    #check whether a move is legal for player
    #return True if move legal, else False
    #in: coordinates - list with x, y coordinates (eg [x,y])
    def legalMove(self, coordinates: List[int], player: int = None) -> bool:
        player = self.current_player if not player else player
        
        #move legal iff it captures opponent pieces
        return False if(self.board[coordinates[1]][coordinates[0]] != 0 or
                        len(self.getCapturePaths(coordinates, player)) == 0) else True

    #find capture paths from a move, assumes move is legal
    #return list of tuples with x,y coordinates of captured squares
    #in: coordinates - list with x, y coordinates (eg [x,y])
    def getCapturePaths(self, coordinates: List[int], player: int = None) -> List[List[int]]:
        neighbours, captured_squares = [], []
        player = self.current_player if not player else player

        #find occupied squares neighbouring coordinates
        for i in range(max(0, coordinates[0]-1), min(coordinates[0]+2, self.grid_size)):
            for j in range(max(0, coordinates[1]-1), min(coordinates[1]+2, self.grid_size)):
                if self.board[j][i] != 0:
                    neighbours.append([i,j])

        #find diagonal, horizontal and vertical capture paths
        for neighbour in neighbours:
            #neighbouring square belongs to player
            if(self.board[neighbour[1]][neighbour[0]] == player):
                continue

            #direction of neighbour from coordinates
            delta_x = neighbour[0] - coordinates[0]
            delta_y = neighbour[1] - coordinates[1]

            new_x, new_y = neighbour[0], neighbour[1]

            #step through indicated direction and store captured squares in path
            path = []
            while 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                #path ended with an empty square
                if (self.board[new_y][new_x] == 0):
                    break

                #path ended with a friendly square
                if self.board[new_y][new_x] == player:
                    #append path coordinas to captured_squares
                    captured_squares += path
                    break

                path.append([new_x, new_y])
                
                new_x += delta_x
                new_y += delta_y

        return captured_squares

    #set up a board
    #return list of tuples with 0 - empty square, 1 - player 1 square, -1 - player 2 square
    def constructBoard(self) -> List[List[int]]:
        #set up board
        board = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        #place starting stones at the four middle squares
        middle = self.grid_size // 2
        board[middle - 1][middle - 1] = 1
        board[middle][middle] = 1
        board[middle][middle - 1] = -1
        board[middle - 1][middle] = -1

        return board

    #printable string representation of game board
    def __repr__(self) -> str:
        #characters for representing board squares based on ownership
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

        score = self.getScore()

        s += f'\nScore: {representation[1]}: {score[1]}, {representation[-1]}: {score[-1]} | {"Current player" if self.winner == None else "Winner"}: {representation[self.current_player] if self.winner == None else ("draw" if self.winner == 0 else representation[self.winner])}'
        return s

def _test():
    game = Othello(8)
    assert game.legalMove([2, 4]) == True
    assert game.validMoves() == [[2, 4], [3, 5], [4, 2], [5, 3]]
    assert len(game.board[0]) * len(game.board) == game.grid_size ** 2
    player = game.current_player
    game.play('f4 f3 d6')
    assert game.current_player == -player
    assert game.getScore() == {1: 5, 0: 57, -1: 2}
    assert game.validMoves() == [[2, 3], [2, 5], [4, 5], [5, 4], [6, 3]]
    game.play('c6')
    game.board[3][3] = 0
    assert game.getCapturePaths([3,3]) == [[3, 4], [4, 3]]
    game.board[2][5] = 0
    game.board[3][4] = 0
    game.board[4][3] = 0
    game.play('b6')
    assert game.winner == 1

    game = Othello(8)
    game.board[4][4] = 0
    game.board[4][3] = 0
    game.board[3][3] = 1

    game.board[0][0] = -1
    game.board[0][1] = -1
    game.board[0][2] = -1
    game.play('f4')
    assert game.winner == 0

    game = Othello(8)
    game.board[4][4] = 0
    game.board[4][3] = 0
    game.board[3][3] = 1
    game.board[3][6] = -1
    game.play('f4 c4')
    assert game.winner == -1

    print('tests passed')

if __name__ == '__main__':
    _test()
        
