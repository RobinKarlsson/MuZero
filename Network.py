#python 3.8

import torch
from numpy import ndarray, zeros, bool

from MuZeroConfig import MuZeroConfig
from NeuralNetworks import Representation, Prediction, Dynamics

class Network(torch.nn.Module):
    def __init__(self, config: MuZeroConfig = MuZeroConfig(),
                 representation: Representation = Representation,
                 prediction: Prediction = Prediction,
                 dynamics: Dynamics = Dynamics):
        
        super().__init__()

        self.config = config
        self.representation = representation(config)
        self.dynamics = dynamics(config)
        self.prediction = prediction(config)
        self.steps = 0
        

    def initial_inference(self, image):
        if type(image) is ndarray:
            image = npToTensor(image)

            if image.ndim != 4:
                image = image.unsqueeze(0)
            
        # representation + prediction function
        hidden_state = self.representation(image)
        policy, value = self.prediction(hidden_state)
        return hidden_state, policy, value
    
    def recurrent_inference(self, hidden_state, action):
        # dynamics + prediction function
        action_representation = torch.from_numpy(action.representation(self.config))
        
        if hidden_state.ndim == 3:
            hidden_state = hidden_state.unsqueeze(0)
        if action_representation.ndim == 3:
            action_representation = action_representation.unsqueeze(0)

        hidden_state = self.dynamics(hidden_state, action_representation)
        policy, value = self.prediction(hidden_state)
        return hidden_state, policy, value
    
    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.steps

def getOptimizer(config, network):
    return torch.optim.SGD(network.parameters(), lr = config.lr_init,
                           weight_decay = config.weight_decay, momentum = config.momentum)

def npToTensor(arr: ndarray):
    return torch.from_numpy(arr)
    
def torchSum(tensor: torch.Tensor):
    return torch.sum(tensor)

def torchMean(tensor: torch.Tensor):
    return torch.mean(tensor)

def torchLog(tensor: torch.Tensor):
    return torch.log(tensor)
