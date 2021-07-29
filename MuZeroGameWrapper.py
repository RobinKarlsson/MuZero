#python 3.8

from typing import List
from numpy import zeros, stack, array, bool, float32
from string import ascii_lowercase
from re import match

from MuZeroConfig import MuZeroConfig
from Action import Action, ActionHistory
from Player import Player
from Node import Node

class MuZeroGameWrapper(object):
    def __init__(self, game, config: MuZeroConfig, action_history: ActionHistory = ActionHistory()):
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

    def humanInput(self, s: str, player: Player):
        #map input string to x,y coordinates
        pattern = r'([a-{}])([0-9]+)(F?)'.format(ascii_lowercase[self.game.grid_size - 1])
        validinput = match(pattern, s.strip())

        if(validinput):
            y = int(validinput.group(2)) - 1
            x = ascii_lowercase.index(validinput.group(1))

            if(x < 0 or y < 0 or x > self.game.grid_size or y > self.game.grid_size):
                return False

        if(self.game.legalMove([x, y], self.game.current_player)):
            action = Action(self.moves, player, [x, y])
            self.performAction(action)
            return True
        return False

    def performAction(self, action: Action):
        moving_player = self.game.current_player
        self.action_history.add_action(action)
        self.game.makeMove(action.coordinates, action.player.player)

        #action rewards:
        #   -1 - other player won
        #   0 - game not over or draw
        #   1 - moving player won
        reward = 0
        if(self.game.winner != None):
            if(moving_player == self.game.winner):
                reward = 1
            elif(self.game.winner == 0):
                pass
            else:
                reward = -1
                
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
        image = zeros((self.config.consider_backward_states + 1, self.config.board_gridsize, self.config.board_gridsize), dtype = bool)

        for i in range(idx + 1 if idx < 3 else 3):
            image[(2-i)*2: (3-i)*2] = self.state_history[idx-i]

        image[-1] = 1 if self.game.current_player == -1 else 0
        return image.astype(float32)

    def saveStats(self, node: Node):
        visit_counts = [(subnode.visit_count, action) for action, subnode in node.children]
        sum_visits = sum([v[0] for v in visit_counts])

        self.root_node_values.append(node.value())

        actions = [Action(idx, node.to_play, divmod(idx, self.config.board_gridsize)) for idx in range(self.config.action_space_size)]
        child_actions = [c[0] for c in node.children]
        stats = []

        for action in actions:
            if(action in child_actions):
                for child_action, subnode in node.children:
                    if(child_action == action):
                        stats.append(subnode.visit_count/sum_visits)
                        break
            else:
                stats.append(0)

        self.subnode_visits.append(stats)

    def getTargets(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player):
        targets = []

        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps

            value = self.root_node_values[bootstrap_index] * self.config.discount ** td_steps if bootstrap_index < len(self.root_node_values) else 0
            for i, reward in enumerate(self.rewards[current_index: bootstrap_index]):
                value += reward * self.config.discount ** i

            if(current_index < len(self.root_node_values)):
                targets.append((value, self.rewards[current_index], self.subnode_visits[current_index]))
            else:
                targets.append((0, 0, []))

        return targets
