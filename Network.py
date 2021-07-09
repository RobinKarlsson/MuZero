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
        self.representation = representation(config.channels)
        self.dynamics = dynamics(config.channels)
        self.prediction = prediction(config.action_space_size, config.board_size)
        self.steps = 0
        

    def initial_inference(self, image):
        if type(image) is ndarray:
            image = torch.from_numpy(image)
            
        # representation + prediction function
        hidden_state = self.representation(image)
        policy, value = self.prediction(hidden_state)
        return hidden_state, policy, torch.mean(value).item()
    
    def recurrent_inference(self, hidden_state, action):
        # dynamics + prediction function
        action_representation = torch.from_numpy(action.representation())
        
        if hidden_state.ndim == 3:
            hidden_state = hidden_state.unsqueeze(0)
        if action_representation.ndim == 3:
            action_representation = action_representation.unsqueeze(0)

        hidden_state = self.dynamics(hidden_state, action_representation)
        policy, value = self.prediction(hidden_state)
        return hidden_state, policy, torch.mean(value).item()
    
    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.steps

