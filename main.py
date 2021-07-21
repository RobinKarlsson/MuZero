#python 3.8

import numpy as np
from psutil import virtual_memory, cpu_percent
from datetime import datetime
from os import listdir
from numpy.random import choice
from torch import no_grad, from_numpy
from torch import sum as torchsum
from torch import mean as torchmean
from torch import log as torchlog

from Games.Othello import Othello
from Games.NInRow import NInRow
from MuZeroGameWrapper import MuZeroGameWrapper
from Network import Network, getOptimizer, saveNetwork, loadNetwork
from NeuralNetworks import Representation, Prediction, Dynamics, Block
from Action import Action
from Node import Node
from MuZeroConfig import MuZeroConfig, visit_softmax_temperature
from Player import Player
from MinMax import MinMax
from Storage import SharedStorage
from ReplayBuffer import ReplayBuffer

def currentTime():
    ram_usage = virtual_memory().percent
    cpu_usage = cpu_percent()
    return f'{datetime.now().strftime("%H:%M:%S")} ram/cpu usage: {ram_usage}/{cpu_usage}:'

def muzero(config: MuZeroConfig, game, num_self_play: int = 50, storage: SharedStorage = SharedStorage()):
    replay_buffer = ReplayBuffer(config)
    network = storage.latest_network(config)

    optimizer = getOptimizer(config, network)

    #update weights
    for epoch in range(network.steps+1, config.training_steps+1):
        if(epoch % config.checkpoint_interval == 0 and epoch > 1):
            print(f'{currentTime()} saving network as Data/{epoch-1}')
            storage.save_network(epoch-1, network)
            saveNetwork(str(epoch-1), network)

        selfPlay(config, storage, replay_buffer, game, num_self_play)
        print(f'{currentTime()} training step {epoch} of {config.training_steps}')

        network.train()

        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        policy_loss, value_loss = 0, 0
        for image, actions, targets in batch:
            #initial inference from current position
            hidden_state, policy, value = network.initial_inference(image)

            num_actions = len(actions)
            pred = [[1., value, 0, policy]]

            #recurrent inference from action and previous hidden state
            for action in actions:
                hidden_state, policy, value = network.recurrent_inference(hidden_state, action, config.board_gridsize)
                pred.append([1./num_actions, value, 0, policy])
                
            #policy & value loss
            for i in range(min(len(pred), len(targets))):
                prediction_value = pred[i][1]
                prediction_policy = pred[i][3]
                target_value = targets[i][0]
                target_policy = targets[i][2]

                if(target_policy == None):
                    continue

                policy_loss += torchmean(torchsum(-from_numpy(np.array(target_policy)) * torchlog(prediction_policy)))
                value_loss += torchmean(torchsum((from_numpy(np.array([target_value])) - prediction_value) ** 2))

        #set gradients to zero in preparation of backpropragation
        optimizer.zero_grad()

        #gradient of loss tensor with respect to leaves
        (policy_loss + value_loss).backward()

        #step based on gradients
        optimizer.step()
            
        network.steps += 1

        print(f'{currentTime()} policy_loss: {policy_loss}, value_loss: {value_loss}')
    storage.save_network(i, network)
    saveNetwork(str(config.training_steps), network)

    return storage.latest_network()

def selfPlay(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, game, nb_games: int):
    for i in range(1, nb_games+1):
        print(f'{currentTime()} selfplay {i} of {nb_games}')
        network = storage.latest_network(config)
        
        wrapper = MuZeroGameWrapper(game, config)
        wrapper = play(wrapper, network, config)

        replay_buffer.save_game(wrapper)

def MCTS(config: MuZeroConfig, root_node: Node, game_wrapper: MuZeroGameWrapper, network: Network, minmax: MinMax):
    for epoch in range(config.num_simulations):
        #print(f'{currentTime()} epoch: {epoch}')
        node = root_node
        history = game_wrapper.action_history.clone()
        path = [root_node]
        
        #traverse the tree by ucb until leaf found
        while node.expanded():
            #select subnode by ucb
            top_score, top_action, top_subnode = -np.inf, None, None

            for action, subnode in node.children:
                pb_c = (np.log((1 + node.visit_count + config.pb_c_base) / config.pb_c_base) + config.pb_c_init) * (
                    np.sqrt(node.visit_count) / (subnode.visit_count + 1)) 

                score = pb_c * subnode.prior

                #if subnode.expanded():
                #    qvalue = subnode.discount * subnode.value() + subnode.reward

                #    if qvalue > 1:
                #        qvalue = 1.
                #    elif qvalue < 0:
                #        qvalue = 0.
                        
                #    score += minmax.normalize(qvalue)
                score += minmax.normalize(subnode.value())

                if score > top_score:
                    top_score = score
                    top_action = action
                    top_subnode = subnode

            #move to selected node
            node = top_subnode
            path.append(top_subnode)
            history.add_action(top_action)

        #next state based on previous hidden state and action
        with no_grad():
            hidden_state, policy, value = network.recurrent_inference(path[-2].hidden_state, history.last_action(), config.board_gridsize)

        #every possible move in action space
        actions = [Action(idx, node.to_play, divmod(idx, config.board_gridsize)) for idx in range(config.action_space_size)]
        node.expand(actions, hidden_state, policy, value)

        #propagate the evaluation up the tree
        for node in path:
            node.visit_count += 1
            node.value_sum += value if node.to_play == history.last_action().player else -value

            #minmax.newBoundary(node.discount * node.value() + node.reward)
            minmax.newBoundary(node.value())
            value = value * config.discount + node.reward

def play(wrapper: MuZeroGameWrapper, network: Network, config: MuZeroConfig):
    while not wrapper.gameOver():
        player = wrapper.currentPlayer()

        action = getMuZeroAction(player, network, config, wrapper)

        #execute selected action
        wrapper.performAction(action)
        #print(wrapper.game)

    return wrapper

def getMuZeroAction(player: Player, network: Network, config: MuZeroConfig, wrapper: MuZeroGameWrapper):
    image = wrapper.getImage(wrapper.moves)

    with no_grad():
        hidden_state, policy, value = network.initial_inference(image)

    node = Node(0, player, hidden_state)

    #expand node with legal actions
    legal_moves = [Action(idx, node.to_play, coordinates) for idx, coordinates in enumerate(wrapper.legalMoves(wrapper.currentPlayer()))]
    node.expand(legal_moves, hidden_state, policy, value)

    node.addNoise(config)

    #run mcts with node as root
    MCTS(config, node, wrapper, network, MinMax(config.boundary_min, config.boundary_max))

    #select next action based on most visited node
    visit_counts = [(subnode.visit_count, action) for action, subnode in node.children]

    #save search statistics
    visits = [v[0] for v in visit_counts]
    actions = [a[1] for a in visit_counts]

    if(len(wrapper.action_history.history) < 21):
        probabilities = [x/sum(visits) for x in visits]
        probabilities
        action = np.random.choice(actions, p=probabilities)

    else:
        idx = visits.index(max(visits))
        action = actions[idx]

    wrapper.saveStats(node)
    return action

def randomvsMuzero(player_random: Player, player_ai: Player, network: Network, config: MuZeroConfig, game = Othello):
    wrapper = MuZeroGameWrapper(game, config)

    while not wrapper.gameOver():
        legal_moves = wrapper.legalMoves(player_random)
        if(wrapper.currentPlayer() == player_random):
            action = Action(wrapper.moves, player_random,
                            legal_moves[choice(len(legal_moves))])
            wrapper.performAction(action)
        else:
            action = getMuZeroAction(player_ai, network, config, wrapper)
            wrapper.performAction(action)
    return wrapper.game.winner

def humanVsMuZero(player_colour: int, network: Network, config: MuZeroConfig, game = Othello):
    wrapper = MuZeroGameWrapper(game, config)
    player_human, player_ai = Player(player_colour), Player(-player_colour)

    while not wrapper.gameOver():
        if(wrapper.currentPlayer() == player_human):
            print(wrapper.game)

            command = 'a0'
            while not wrapper.humanInput(command, player_human):
                command = input('Type for example f4 to place a stone on f4: ')

        else:
            action = getMuZeroAction(player_ai, network, config, wrapper)

            #execute selected action
            wrapper.performAction(action)

    print(wrapper.game)

if __name__ == '__main__':
    option = None
    while option not in [str(i) for i in range(1, 5)]:
        option = input('Options:\n 1 - play without MuZero\n 2 - Play agains MuZero\n 3 - Train MuZero\n')

    option = int(option)
    config = MuZeroConfig()
    storage = SharedStorage()
    game = Othello

    if option == 1:
        game = game()

        while game.canMove():
            print(game)
            command = input('Type for example f4 to place a stone on f4: ')
            game.play(command)
        print(game)

    elif option == 2:
        networks = listdir('Data/')
        if(len(networks) != 0):
            file = None
            while file not in networks + ['-1']:
                print(f'Available networks: {", ".join(networks)}')
                file = input('Enter name of network to import, -1 for latest ')

            colour = 1 #let player play black

            steps, network = loadNetwork(config, file if file != "-1" else None)

            humanVsMuZero(colour, network, config, game)

    elif option == 3:
        #load latest saved network

        steps, network = loadNetwork(config)

        if(type(steps) == int):
            print(f'{currentTime()} loading network Data/{steps}')
            storage.save_network(steps, network)

        muzero(config, game, storage = storage, num_self_play = 5)

    elif option == 4:
        networks = listdir('Data/')
        if(len(networks) != 0):
            file = None
            while file not in networks + ['-1']:
                print(f'Available networks: {", ".join(networks)}')
                file = input('Enter name of network to import, -1 for latest ')

            colour = 1 #muzero plays black
            
            steps, network = loadNetwork(config, file if file != "-1" else None)
            victories = {1:0, 0:0, -1:0}
            num_games = 10
            for i in range(num_games):
                print(f'game {i} of {num_games}')
                winner = randomvsMuzero(Player(colour), Player(-colour), network, config, game)

                victories[winner] += 1

            print(f'MuZero won {victories[1]}/{num_games}, draws: {victories[0]}/{num_games}')
