#python 3.8

from typing import List
from numpy import zeros, stack, array, bool, float32

from MuZeroConfig import MuZeroConfig
from Action import Action, ActionHistory
from Player import Player
from Node import Node

class MuZeroGameWrapper(object):
    def __init__(self, game, action_history: ActionHistory = ActionHistory(), config: MuZeroConfig = MuZeroConfig()):
        self.game_object = game
        self.game = game()
        self.config = config
        self.action_history = action_history
        self.state_history = [self.getGameState()]
        self.moves = 0
        self.rewards = []
        self.root_node_values = []
        self.subnode_visits = []

    def gameOver(self):
        return False if self.game.winner == None else True

    def currentPlayer(self):
        return Player(self.game.current_player)

    def performAction(self, action: Action):
        self.action_history.add_action(action)
        self.game.makeMove(action.coordinates, action.player.player)

        reward = 0
        if(self.game.winner != None):
            reward = 1 if self.game.current_player == self.game.winner else -1
        self.rewards.append(reward)
        
        self.state_history.append(self.getGameState())
        self.moves += 1

    def legalMoves(self, player: Player) -> List[int]:
        return self.game.validMoves(player.player)

    def getGameState(self) -> array:
        board = array([array(x) for x in self.game.board])
        return stack((board == -1, board == 1), axis = 0).astype(bool)

    def getImage(self, idx: int) -> array:
        #idx 0-6: 3 latest gamestates, idx 7: True for player 1, False for player -1
        image = zeros((7, self.config.board_rows, self.config.board_columns), dtype = bool)

        for i in range(idx + 1 if idx < 3 else 3):
            image[(2-i)*2: (3-i)*2] = self.state_history[idx-i]

        image[-1] = 1 if self.game.current_player == -1 else 0
        return image.astype(float32)

    def saveStats(self, node: Node):
        visit_counts = [(subnode.visit_count, action) for action, subnode in node.children]
        sum_visits = sum([v[0] for v in visit_counts])

        self.root_node_values.append(node.value())

        for action, subnode in node.children:
            self.subnode_visits.append(subnode.visit_count/sum_visits)

    def getTargets(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player):
        targets = []

        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps

            value = self.root_node_values[bootstrap_index] if bootstrap_index < len(self.root_node_values) else 0
            for reward in self.rewards[current_index: bootstrap_index]:
                value += reward

            if(current_index < len(self.root_node_values)):
                targets.append((value, self.rewards[current_index], self.subnode_visits[current_index]))
            else:
                targets.append((0,0,[]))

        return targets
