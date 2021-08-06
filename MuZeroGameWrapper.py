#python 3.8

from typing import List
from numpy import zeros, stack, array, float32
from numpy import bool as np_bool
from string import ascii_lowercase
from re import match

from MuZeroConfig import MuZeroConfig
from Action import Action, ActionHistory
from Player import Player
from Node import Node

class MuZeroGameWrapper(object):
    def __init__(self, game, config: MuZeroConfig, action_history: ActionHistory = None):
        self.game_object = game
        self.game = game()
        self.config = config
        self.action_history = action_history if action_history else ActionHistory()
        self.state_history = [self.getGameState()]
        self.moves = 0
        self.rewards = []
        self.root_node_values = []
        self.subnode_visits = []

    #true if gameover, otherwise false
    def gameOver(self) -> bool:
        return False if self.game.winner == None else True

    #get player object of current player
    def currentPlayer(self) -> Player:
        return Player(self.game.current_player)

    #Transorm a string input into coorinates and perform indicated move
    #in:    string representation of move (for example a1 for coordinates [0,0])
    #       player object performing move
    #out:   true if move was performed, else false
    def humanInput(self, s: str, player: Player) -> bool:
        #map input string to x,y coordinates
        pattern = r'([a-{}])([0-9]+)(F?)'.format(ascii_lowercase[self.game.grid_size - 1])
        validinput = match(pattern, s.strip())

        if(validinput):
            y = int(validinput.group(2)) - 1
            x = ascii_lowercase.index(validinput.group(1))

            if(x < 0 or y < 0 or x > self.game.grid_size or y > self.game.grid_size):
                return False
        else:
            return False

        #execute move iff legal
        if(self.game.legalMove([x, y], self.game.current_player)):
            action = Action(self.moves, player, [x, y])
            self.performAction(action)
            return True
        return False

    #Perform an action agains the game environment
    #in:    Action object
    #out:   Player object that moved
    def performAction(self, action: Action) -> Player:
        self.action_history.add_action(action)

        #reward from performed action. In games like Othello this is 0 until a player has won
        reward = self.game.makeMove(action.coordinates, action.player.player)

        self.rewards.append(reward)
        self.state_history.append(self.getGameState())
        self.moves += 1
        return action.player

    #return list of coordinates for legal moves
    def legalMoves(self, player: Player) -> List[int]:
        return self.game.validMoves(player.player)

    #representation of current gamestate
    def getGameState(self) -> array:
        board = array([array(x) for x in self.game.board])
        return stack((board == -1, board == 1), axis = 0).astype(np_bool)

    #image of game
    def getImage(self, idx: int) -> array:
        image = zeros((self.config.consider_backward_states * 2 + 1, self.config.board_gridsize, self.config.board_gridsize), np_bool)

        for i in range(min(idx, self.config.consider_backward_states)):
            image[(self.config.consider_backward_states - i - 1) * 2:
                  (self.config.consider_backward_states - i) * 2
                  ] = self.state_history[idx - i]

        image[-1] = 0 if self.game.current_player == -1 else 1

        return image.astype(float32)

    #save search statistics
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


def _test():
    from Games.NInRow import NInRow
    config = MuZeroConfig(max_moves = 3**2, window_size = 30, num_simulations = 30, action_space_size = 3**2, board_gridsize = 3, td_steps = 3*3)

    w1 = MuZeroGameWrapper(NInRow, config)

    a1 = Action(0, w1.currentPlayer(), [1,1])
    w1.performAction(a1)
    assert w1.action_history.history[-1].coordinates == [1,1]


if __name__ == '__main__':
    _test()
