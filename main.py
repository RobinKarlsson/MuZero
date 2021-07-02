#python 3.8

import numpy as np
from copy import deepcopy
from random import choice

from Games.Othello import Othello
from MuZeroGameWrapper import MuZeroGameWrapper
from Network import Network
from Action import Action
from Node import Node
from MuZeroConfig import MuZeroConfig
from Player import Player

def MCTS(config: MuZeroConfig, root_node: Node, game_wrapper: MuZeroGameWrapper):

    for epoch in range(config.num_simulations):
        wrapper = deepcopy(game_wrapper)
        node = root_node
        history = [action for action in game_wrapper.action_history.history]
        
        path = [root_node]

        #traverse the tree by ucb until leaf found
        while node.expanded():
            player = Player(wrapper.game.current_player)
            #select subnode by ucb
            top_score, top_action, top_subnode = -np.inf, None, None
            for action, subnode in node.children.items():
                pb_c = (np.log((1 + node.visit_count + config.pb_c_base) / config.pb_c_base) + config.pb_c_init) * (np.sqrt(node.visit_count) / (subnode.visit_count + 1)) 

                score = pb_c * subnode.prior
                
                if subnode.expanded():
                    qvalue = subnode.discount * subnode.value() + subnode.reward

                    if qvalue > 1:
                        qvalue = 1.
                    elif qvalue < 0:
                        qvalue = 0.
                        
                    score += qvalue

                if score > top_score:
                    top_score = score
                    top_action = action
                    top_subnode = subnode

            #move to selected node
            node = top_subnode
            path.append(top_subnode)
            history.append(top_action)

            wrapper.performAction(top_action, player)

    #propagate the evaluation up the tree
    value = 1
    for node in path:
        node.visit_count += 1
        node.value_sum += value if node.to_play == history[-1].player else -value
        value = value * config.discount + node.reward

    for action, node in root_node.children.items():
        pass

def play(config: MuZeroConfig, wrapper: MuZeroGameWrapper):
    while wrapper.game.canMove():
        player = Player(wrapper.game.current_player)
        node = Node(0, player)
        image = wrapper.game.board

        idx = len(wrapper.action_history.history)
        move = choice(wrapper.legalMoves(player))
        action = Action(idx, move, player)
        wrapper.performAction(action, player)
        
        node.expand([Action(idx + 1, a, player) for a in wrapper.legalMoves(player)])
        node.addNoise(config)

        MCTS(config, node, wrapper)

    return wrapper
    

def _test():
    game = Othello()

    wrapper = MuZeroGameWrapper(game)
    config = MuZeroConfig()

    play(config, wrapper)

if __name__ == '__main__':
    _test()
    
    option = None
    while option not in [str(i) for i in range(1,4)]:
        option = input('Options:\n 1 - play Othello without MuZero\n 2 - Play Othello agains MuZero\n 3 - Train MuZero\n')

    option = int(option)

    if option == 1:
        game = Othello()

        while game.canMove():
            print(game)
            command = input('Type for example f4 to place a stone on f4: ')
            game.play(command)
        print(game)

    elif option == 2:
        pass

    elif option == 3:
        pass

