#python 3.8

import torch
from os import listdir
from numpy import ndarray, zeros, bool
from Action import Action

from MuZeroConfig import MuZeroConfig
from NeuralNetworks import Representation, Prediction, Dynamics

class Network(torch.nn.Module):
    def __init__(self, config: MuZeroConfig,
                 representation: Representation = Representation,
                 prediction: Prediction = Prediction,
                 dynamics: Dynamics = Dynamics):
        
        super().__init__()

        self.representation = representation(config.board_gridsize, config.channels, config.consider_backward_states + 1).to(config.torch_device)
        self.dynamics = dynamics(config.board_gridsize, config.channels).to(config.torch_device)
        self.prediction = prediction(config.board_gridsize, config.board_size, config.action_space_size).to(config.torch_device)
        self.steps = 0

    def initial_inference(self, config: MuZeroConfig, image):
        if type(image) is ndarray:
            image = torch.from_numpy(image).to(config.torch_device)

            if image.ndim != 4:
                image = image.unsqueeze(0)
            
        # representation + prediction function
        hidden_state = self.representation(image)
        policy, value = self.prediction(hidden_state)
        return hidden_state, policy, value
    
    def recurrent_inference(self, config: MuZeroConfig, hidden_state, action: Action):
        # dynamics + prediction function
        action_representation = torch.from_numpy(action.representation(config.board_gridsize)).to(config.torch_device)
        
        if hidden_state.ndim == 3:
            hidden_state = hidden_state.unsqueeze(0)
        if action_representation.ndim == 3:
            action_representation = action_representation.unsqueeze(0)

        hidden_state = self.dynamics(hidden_state, action_representation)
        policy, value = self.prediction(hidden_state)
        return hidden_state, policy, value

def getOptimizer(config, network):
    #optimizer = torch.optim.SGD(network.parameters(), lr = config.lr_init, weight_decay = config.weight_decay, momentum = config.momentum)
    optimizer = torch.optim.ASGD(network.parameters(), lr = config.lr_init, weight_decay = config.weight_decay)
    #optimizer = torch.optim.Adam(network.parameters(), lr = config.lr_init)
    #optimizer = torch.optim.Adagrad(network.parameters(), lr = config.lr_init, weight_decay = config.weight_decay)
    return optimizer

def saveNetwork(file: str, network: Network):
    if not 'Data/' in file:
        file = 'Data/' + file

    torch.save(network, file)

def loadNetwork(config: MuZeroConfig, file: str = None):
    if(file == None):
        greatest_step = 0
        for f in listdir('Data/'):
            try:
                f = int(f)
            except ValueError:
                continue

            if(f > greatest_step):
                greatest_step = f

        if(greatest_step != 0):
            file = str(greatest_step)
        else:
            return 0, Network(config).to(config.torch_device)
    
    if not 'Data/' in file:
        file = 'Data/' + file

    network = torch.load(file)

    file = file.replace('Data/', '')
    try:
        file = int(file)
    except ValueError:
        pass
    
    return file, network.to(config.torch_device)

