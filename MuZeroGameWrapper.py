#python 3.8

from typing import List
from numpy import zeros, stack, array, bool, float32

from Action import Action, ActionHistory
from Player import Player

class MuZeroGameWrapper(object):
    def __init__(self, game, action_history: ActionHistory = ActionHistory()):
        self.game = game
        self.action_history = action_history
        self.state_history = [self.getGameState()]
        self.moves = 0

    def gameOver(self):
        return False if self.game.winner == None else True

    def currentPlayer(self):
        return Player(self.game.current_player)

    def performAction(self, action: Action, player: Player):
        self.action_history.add_action(action)
        self.game.makeMove(action.coordinates, player.player)
        self.state_history.append(self.getGameState())
        self.moves += 1

    def legalMoves(self, player: Player) -> List[int]:
        return self.game.validMoves(player.player)

    def getGameState(self) -> array:
        board = array([array(x) for x in self.game.board])
        return stack((board == -1, board == 1))

    def getImage(self, idx: int) -> array:
        image = zeros((9, 2, 8, 8), dtype = bool)

        for i in range(idx + 1 if idx < 4 else 4):
            image[i] = self.state_history[-i-1]
        image[-1] = self.game.current_player
        return image.astype(float32)
