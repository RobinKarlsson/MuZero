#python 3.8

import numpy as np

from Games.Othello import Othello
from MuZeroGameWrapper import MuZeroGameWrapper
from Network import Network, getOptimizer, npToTensor, torchSum, torchMean, torchLog
from Action import Action
from Node import Node
from MuZeroConfig import MuZeroConfig
from Player import Player
from MinMax import MinMax
from Storage import SharedStorage
from ReplayBuffer import ReplayBuffer

def muzero(config: MuZeroConfig = MuZeroConfig(), game = Othello, num_self_play: int = 1):
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)
    network = Network(config)

    optimizer = getOptimizer(config, network)

    #update weights
    for epoch in range(1, config.training_steps+1):
        print(f'training step {epoch} of {config.training_steps}')
        selfPlay(config, storage, replay_buffer, game, num_self_play)
            
        if(epoch % config.checkpoint_interval == 0):
            storage.save_network(i, network)

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
                hidden_state, policy, value = network.recurrent_inference(hidden_state, action)
                pred.append([1./num_actions, value, 0, policy])
                
            #policy & value loss
            for i in range(min(len(pred), len(targets))):
                prediction_value = pred[i][1]
                prediction_policy = pred[i][3]
                target_value = targets[i][0]
                target_policy = targets[i][2]

                policy_loss += torchMean(torchSum(-npToTensor(np.array(target_policy)) * torchLog(prediction_policy)))
                value_loss += torchMean(torchSum((npToTensor(np.array([target_value])) - prediction_value) ** 2))

        #set gradients to zero in preparation of backpropragation
        optimizer.zero_grad()

        #gradient of loss tensor with respect to leaves
        (policy_loss + value_loss).backward()

        #step based on gradients
        optimizer.step()
            
        network.steps += 1

        print(f'policy_loss: {policy_loss}, value_loss: {value_loss}')
    storage.save_network(i, network)

    return storage.latest_network()

def selfPlay(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, game, nb_games: int):
    for _ in range(nb_games):
        network = storage.latest_network()

        wrapper = MuZeroGameWrapper(game)
        wrapper = play(wrapper, network, config)

        replay_buffer.save_game(wrapper)

def MCTS(config: MuZeroConfig, root_node: Node, game_wrapper: MuZeroGameWrapper, network: Network, minmax: MinMax):
    for epoch in range(config.num_simulations):
        #print(f'epoch: {epoch}')
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
        hidden_state, policy, value = network.recurrent_inference(path[-2].hidden_state, history.last_action())

        #every possible move in action space
        actions = [Action(idx, node.to_play, divmod(idx, 8)) for idx in range(config.action_space_size)]
        node.expand(actions, hidden_state, policy, value)

        #propagate the evaluation up the tree
        for node in path:
            node.visit_count += 1
            node.value_sum += value if node.to_play == history.last_action().player else -value

            minmax.newBoundary(node.discount * node.value() + node.reward)
            value = value * config.discount + node.reward

def play(wrapper: MuZeroGameWrapper, network: Network, config: MuZeroConfig):
    while not wrapper.gameOver():
        player = wrapper.currentPlayer()

        image = wrapper.getImage(wrapper.moves)

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
            action = np.random.choice(actions, p=probabilities)

        else:
            idx = visits.index(max(visits))
            action = actions[idx]

        wrapper.saveStats(node)

        #execute selected action
        wrapper.performAction(action)
        #print(wrapper.game)

    return wrapper
    

def _test():
    game = Othello

    config = MuZeroConfig()
    network = muzero(config)

if __name__ == '__main__':
    #_test()
    
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

