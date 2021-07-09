#python 3.8

import numpy as np
from copy import deepcopy
from random import choice
from torch import no_grad

from Games.Othello import Othello
from MuZeroGameWrapper import MuZeroGameWrapper
from Network import Network
from Action import Action
from Node import Node
from MuZeroConfig import MuZeroConfig
from Player import Player
from MinMax import MinMax

def MCTS(config: MuZeroConfig, root_node: Node, game_wrapper: MuZeroGameWrapper, network: Network, minmax: MinMax):
    for _ in range(config.num_simulations):
        wrapper = deepcopy(game_wrapper)
        node = root_node
        history = [action for action in game_wrapper.action_history.history]
        
        path = [root_node]

        #traverse the tree by ucb until leaf found
        while node.expanded():

            player = wrapper.currentPlayer()
            #select subnode by ucb
            top_score, top_action, top_subnode = -np.inf, None, None

            for action, subnode in node.children:

                pb_c = (np.log((1 + node.visit_count + config.pb_c_base) / config.pb_c_base) + config.pb_c_init) * (
                    np.sqrt(node.visit_count) / (subnode.visit_count + 1)) 

                score = pb_c * subnode.prior

                if subnode.expanded():
                    qvalue = subnode.discount * subnode.value() + subnode.reward

                    if qvalue > 1:
                        qvalue = 1.
                    elif qvalue < 0:
                        qvalue = 0.
                        
                    score += minmax.normalize(qvalue)

                if score > top_score:
                    top_score = score
                    top_action = action
                    top_subnode = subnode

            #move to selected node
            node = top_subnode
            path.append(top_subnode)
            history.append(top_action)
            wrapper.performAction(top_action, player)

    hidden_state, policy, value = network.recurrent_inference(path[-2].hidden_state, history[-1])

    node.expand([Action(history[-1].index + 1, coordinates, wrapper.currentPlayer())
                 for coordinates in wrapper.legalMoves(wrapper.currentPlayer())], hidden_state, policy, value)

    #propagate the evaluation up the tree
    for node in path:
        node.visit_count += 1
        node.value_sum += value if node.to_play == history[-1].player else -value

        minmax.newBoundary(node.discount * node.value() + node.reward)
        value = value * config.discount + node.reward

def play(wrapper: MuZeroGameWrapper, network: Network, config: MuZeroConfig):
    while True:
        player = wrapper.currentPlayer()

        image = wrapper.getImage(wrapper.moves)

        hidden_state, policy, value = network.initial_inference(image)

        node = Node(0, player, hidden_state)
        
        idx = len(wrapper.action_history.history)
        move = choice(wrapper.legalMoves(player))
        action = Action(idx, move, player)

        wrapper.performAction(action, player)

        if(wrapper.gameOver()):
            break

        node.expand([Action(idx + 1, coordinates, wrapper.currentPlayer())
                     for coordinates in wrapper.legalMoves(wrapper.currentPlayer())], hidden_state, policy, value)

        node.addNoise(config)

        MCTS(config, node, wrapper, network, MinMax(config.boundary_min, config.boundary_max))

    return wrapper
    

def _test():
    game = Othello()

    wrapper = MuZeroGameWrapper(game)
    config = MuZeroConfig()
    network = Network(config)

    wrapper = play(wrapper, network, config)

if __name__ == '__main__':
    _test()
    
    option = None
    while option not in [str(i) for i in range(1, 4)]:
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

