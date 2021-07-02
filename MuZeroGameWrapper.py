#python 3.8

from typing import List

from Action import Action, ActionHistory
from Player import Player

class MuZeroGameWrapper(object):
    def __init__(self, game, action_history: ActionHistory = ActionHistory()):
        self.game = game
        self.action_history = action_history

    def performAction(self, action: Action, player: Player):
        self.action_history.add_action(action)
        self.game.makeMove(action.coordinates, player.player )

    def legalMoves(self, player: Player) -> List[int]:
        return self.game.validMoves(player.player)
        
