#python 3.8

import numpy as np
from psutil import virtual_memory, cpu_percent
from datetime import datetime
from os import listdir
from pathlib import Path as pathlibPath
from numpy.random import choice
from threading import Thread
from time import sleep
from torch import no_grad, from_numpy, cuda, int64, Tensor
from torch.nn.functional import log_softmax
from torch import sum as torchsum
from torch import mean as torchmean
from torch import log as torchlog

from Games.Othello import Othello
from MuZeroGameWrapper import MuZeroGameWrapper
from Network import Network, getOptimizer, saveNetwork, loadNetwork
from NeuralNetworks import Representation, Prediction, Dynamics, Block
from Action import Action
from Node import Node
from MuZeroConfig import MuZeroConfig
from Player import Player
from MinMax import MinMax
from Storage import SharedStorage
from ReplayBuffer import ReplayBuffer

from ctypes import CDLL, c_float, c_double, c_int
from os.path import abspath

so_file = "c_functions.so"
f = CDLL(abspath(so_file))
f.calc_pb_c.argtypes = [c_double, c_double, c_float, c_float]
f.calc_pb_c.restype = c_double

f.ucb_score.argtypes = [c_double, c_double, c_float, c_float, c_double, c_int, c_float, c_float]
f.ucb_score.restype = c_double

pathlibPath('Data').mkdir(parents=True, exist_ok=True)
pathlibPath('Games').mkdir(parents=True, exist_ok=True)

def currentTime():
    ram_usage = virtual_memory().percent
    cpu_usage = cpu_percent()
    return f'{datetime.now().strftime("%H:%M:%S")} ram/cpu usage: {ram_usage}/{cpu_usage}:'

def muzero(config: MuZeroConfig, game, optimizer, network: Network = None, storage: SharedStorage = SharedStorage(), game_history = None):
    if not network:
        network = storage.latest_network(config)
    replay_buffer = ReplayBuffer(config)
    
    if(game_history):
        replay_buffer.buffer = game_history

    num_decimals = 3

    selfplay_threads = []
    for _ in range(config.num_threads):
        selfplay_threads.append(
            Thread(target = selfPlay, args = (config, storage, replay_buffer, game)).start()
            )

    while len(replay_buffer.buffer) == 0:
        sleep(1)

    for epoch in range(network.steps+1, config.training_steps+1):
        if((epoch - 1) % config.checkpoint_interval == 0 and epoch > 1):
            print(f'{currentTime()} saving network as Data/{epoch-1}')
            saveNetwork(str(epoch-1), network, optimizer, replay_buffer.buffer)

        #populate replay_buffer with selfplay games
        #if((epoch - 1) % config.refresh_replaybuffer == 0 or len(replay_buffer.buffer) == 0):
            #print(f'{currentTime()} updating replay buffer with {config.num_selfplay} selfplay games')
            #selfPlay(config, storage, replay_buffer, game)

        network.train()

        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        policy_loss, value_loss = 0, 0

        #run through selected games and calculate loss
        for image, actions, targets in batch:
            #initial inference from current position
            hidden_state, policy, value = network.initial_inference(config, image)

            num_actions = len(actions)
            pred = [[1., value, policy]]

            #step through actions
            for action in actions:
                #recurrent inference from action and previous hidden state
                hidden_state, policy, value = network.recurrent_inference(config, hidden_state, action)

                pred.append([1./num_actions, value, policy])

            #policy & value loss
            for p, t in zip(pred, targets):
                _, prediction_value, prediction_policy = p
                target_value, _, target_policy = t

                if(len(target_policy) == 0):
                    continue

                target_policy = from_numpy(np.array(target_policy)).to(config.torch_device)
                
                log_probs = log_softmax(prediction_policy, dim=1)
                p_l = -(log_probs * target_policy).sum() / log_probs.shape[1]

                policy_loss += p_l
                
                value_loss += ((Tensor([target_value]).to(config.torch_device) - prediction_value)**2).sum()

        #set gradients to zero
        optimizer.zero_grad()

        #compute gradient of loss tensor with respect to leaves
        (policy_loss + value_loss).backward()

        #step based on gradients
        optimizer.step()
        network.steps += 1

        storage.save_network(epoch-1, network)
        if((epoch) % 1 == 0):
            print(f'{currentTime()} epoch {epoch}/{config.training_steps}, buffer: {len(replay_buffer.buffer)}/{config.window_size}, p_loss: {round(policy_loss.item(), num_decimals)}, v_loss: {round(value_loss.item(), num_decimals)}')
        
    for t in selfplay_threads:
        t.stop()
    
    storage.save_network(i, network)
    saveNetwork(str(config.training_steps), network, replay_buffer.buffer)

    return storage.latest_network()

def selfPlay(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, game):
    while True:
    #for i in range(1, config.num_selfplay + 1):
        #print(f'{currentTime()} selfplay {i} of {config.num_selfplay}')
        network = storage.latest_network(config)

        wrapper = MuZeroGameWrapper(game, config)
        wrapper = play(wrapper, network, config)

        replay_buffer.save_game(wrapper)

def MCTS(config: MuZeroConfig, root_node: Node, game_wrapper: MuZeroGameWrapper, network: Network, minmax: MinMax):
    game_wrapper_history_len = len(game_wrapper.action_history.history)
    for epoch in range(config.num_simulations):
        #print(f'{currentTime()} epoch: {epoch}')
        node = root_node
        game_wrapper.action_history.history = game_wrapper.action_history.history[:game_wrapper_history_len]
        history = game_wrapper.action_history.clone()
        path = [root_node]
        #traverse the tree by ucb until leaf found
        while node.expanded():
            #select subnode by ucb
            top_score, top_action, top_subnode = -np.inf, None, None

            for action, subnode in node.children:
                try:
                    pb_c = f.calc_pb_c(node.visit_count, subnode.visit_count, config.pb_c_base, config.pb_c_init)
                    score = f.ucb_score(pb_c, subnode.prior, subnode.discount, subnode.reward, subnode.value(),
                                        len(subnode.children), minmax.maximum, minmax.minimum)

                except Exception as inst:
                    print('MCTS c failed')
                    print(type(inst))
                    print(inst.args)
                    print(inst)
                    
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
            history.add_action(top_action)

        #next state based on previous hidden state and action
        with no_grad():
            hidden_state, policy, value = network.recurrent_inference(config, path[-2].hidden_state, history.last_action())

        #every possible move in action space
        actions = [Action(idx, node.to_play, divmod(idx, config.board_gridsize)) for idx in range(config.action_space_size)]
        node.expand(actions, hidden_state, policy, value)

        #propagate the evaluation up the tree
        for node in path:
            node.visit_count += 1
            node.value_sum += value.item() if node.to_play == history.last_action().player else -value.item()

            minmax.newBoundary(node.discount * node.value() + node.reward)
            value = value * config.discount + node.reward

    game_wrapper.action_history.history = game_wrapper.action_history.history[:game_wrapper_history_len]

def play(wrapper: MuZeroGameWrapper, network: Network, config: MuZeroConfig):
    while not wrapper.gameOver():
        player = wrapper.currentPlayer()

        action = getMuZeroAction(player, network, config, wrapper)
        
        #execute selected action
        wrapper.performAction(action)

    return wrapper

def getMuZeroAction(player: Player, network: Network, config: MuZeroConfig, wrapper: MuZeroGameWrapper):
    image = wrapper.getImage(wrapper.moves)

    with no_grad():
        hidden_state, policy, value = network.initial_inference(config, image)

    node = Node(0, player, hidden_state)

    #expand node with legal actions
    legal_moves = [Action(idx, node.to_play, coordinates) for idx, coordinates in enumerate(wrapper.legalMoves(wrapper.currentPlayer()))]
    node.expand(legal_moves, hidden_state, policy, value)

    node.addNoise(config)

    #run mcts with node as root
    MCTS(config, node, wrapper, network, MinMax(config.boundary_min, config.boundary_max))

    #select next action based on search statistics
    visit_counts = [(subnode.visit_count, action) for action, subnode in node.children]

    #search statistics
    visits = [v[0] for v in visit_counts]
    actions = [a[1] for a in visit_counts]

    sum_visits = sum(visits)
    if(len(wrapper.action_history.history) < config.softmax_threshold):
        #probability of each node being visited
        probabilities = [x / sum_visits for x in visits]
        #select action based on visit probabilities
        action = np.random.choice(actions, p = probabilities)

    else:
        idx = visits.index(max(visits))
        action = actions[idx]
    
    #save search statistics
    wrapper.saveStats(node)
    return action

def randomvsMuzero(player_random: Player, player_ai: Player, network: Network, config: MuZeroConfig, game = Othello):
    wrapper = MuZeroGameWrapper(game, config)

    while not wrapper.gameOver():
        legal_moves = wrapper.legalMoves(player_random)
        if(wrapper.currentPlayer() == player_random):
            action = Action(wrapper.moves, player_random,
                            legal_moves[choice(len(legal_moves))])
        else:
            action = getMuZeroAction(player_ai, network, config, wrapper)
        wrapper.performAction(action)
    return wrapper.game.winner

def muzerovsMuzero(player_1: Player, player_2: Player, network_1: Network, network_2: Network, config: MuZeroConfig, game = Othello):
    wrapper = MuZeroGameWrapper(game, config)

    while not wrapper.gameOver():
        legal_moves = wrapper.legalMoves(player_random)
        if(wrapper.currentPlayer() == player_1):
            action = getMuZeroAction(player_1, network_1, config, wrapper)
        else:
            action = getMuZeroAction(player_2, network_2, config, wrapper)
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
    if(cuda.is_available()):
        print('utilizing GPU CUDA operations')
        torch_device = 'cuda'
    else:
        print('CUDA not available')
        torch_device = 'cpu'

    option = None
    while option not in [str(i) for i in range(1, 5)]:
        option = input('Options:\n 1 - play without MuZero\n 2 - Play against MuZero\n 3 - Train MuZero\n')

    option = int(option)
    storage = SharedStorage()

    config = MuZeroConfig(torch_device = torch_device)
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
            file = '-1'
            while file not in networks + ['-1']:
                print(f'Available networks: {", ".join(networks)}')
                file = input('Enter name of network to import, -1 for latest ')

            colour = 1 #let player play black

            steps, network, _, _ = loadNetwork(config, file if file != "-1" else None)

            humanVsMuZero(colour, network, config, game)

    elif option == 3:
        #load latest saved network
        steps, network, optimizer, game_history = loadNetwork(config)

        if(type(steps) == int):
            print(f'{currentTime()} loading network Data/{steps}')
            storage.save_network(steps, network)

        muzero(config, game, optimizer, network, storage, game_history)

    elif option == 4:
        networks = listdir('Data/')
        if(len(networks) != 0):
            file = '-1'
            while file not in networks + ['-1']:
                print(f'Available networks: {", ".join(networks)}')
                file = input('Enter name of network to import, -1 for latest ')

            colour = 1 #muzero plays black
            
            steps, network, _, _ = loadNetwork(config, file if file != "-1" else None)
            victories = {1:0, 0:0, -1:0}
            num_games = 100
            for i in range(1, num_games + 1):
                print(f'game {i} of {num_games}')
                winner = randomvsMuzero(Player(colour), Player(-colour), network, config, game)

                victories[winner] += 1

                print(f'MuZero won {victories[1]}/{i}, draws: {victories[0]}/{i}')
