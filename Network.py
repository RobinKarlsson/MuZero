#python 3.8

import torch

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
        # representation + prediction function
        hidden_state = self.representation(image)
        policy, value = self.prediction(hidden_state)
        return hidden_state, policy, value
    
    def recurrent_inference(self, hidden_state, action):
        # dynamics + prediction function
        hidden_state = self.dynamics(hidden_state, action)
        policy, value = self.prediction(hidden_state)
        return hidden_state, policy, value
    
    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.steps
