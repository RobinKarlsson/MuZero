#python 3.8

import torch.nn

class Block(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        
        self.nn = torch.nn.Sequential(torch.nn.Conv2d(in_channels = channels, #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                                                         out_channels = channels,
                                                         kernel_size = 3,
                                                         padding = 1),
                                         torch.nn.ReLU(), #https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
                                         torch.nn.Conv2d(in_channels = channels, #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                                                         out_channels = channels,
                                                         kernel_size = 3,
                                                         padding = 1))

    def forward(self, x):
        return torch.nn.functional.relu(x + self.nn(x)) #https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html#torch.nn.functional.relu

#board -> hidden
class Representation(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.nn = torch.nn.Sequential( #https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
            torch.nn.Conv2d(in_channels = 2, #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                            out_channels = channels,
                            kernel_size = 3,
                            padding = 1),
            torch.nn.ReLU(), #https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            Block(channels),
            Block(channels),
            Block(channels),
            torch.nn.Conv2d(in_channels = channels, #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                            out_channels = 8,
                            kernel_size = 3,
                            padding = 1),
            torch.nn.ReLU()) #https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html

    def forward(self, image):
        return self.nn(image)

#hidden -> value, policy
class Prediction(torch.nn.Module):
    def __init__(self, action_space_size: int, board_size: int):
        super().__init__()

        d_board_size = board_size * 2

        self.nn_policy = torch.nn.Sequential( #https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
            torch.nn.Conv2d(in_channels = 8, #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                            out_channels = 2,
                            kernel_size = 1),
            torch.nn.ReLU(), #https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            torch.nn.Flatten(), #https://pytorch.org/docs/stable/generated/torch.flatten.html
            torch.nn.Linear(in_features = d_board_size, #https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
                            out_features = d_board_size),
            torch.nn.ReLU(), #https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            torch.nn.Linear(in_features = d_board_size, #https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
                            out_features = action_space_size),
            torch.nn.Softmax(dim=1)) #https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html

        self.nn_value = torch.nn.Sequential( #https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
            torch.nn.Conv2d(in_channels = 8, #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                            out_channels = 2,
                            kernel_size = 1),
            torch.nn.ReLU(), #https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            torch.nn.Flatten(), #https://pytorch.org/docs/stable/generated/torch.flatten.html
            torch.nn.Linear(in_features = d_board_size, #https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
                            out_features = 1),
            torch.nn.Tanh()) #https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html

    def forward(self, hidden_state):
        return self.nn_policy(hidden_state), self.nn_value(hidden_state)

class Dynamics(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.nn = torch.nn.Sequential( #https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
            torch.nn.Conv2d(in_channels = 10, #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                            out_channels = channels,
                            kernel_size = 3,
                            padding = 1,
                            bias = False),
            torch.nn.ReLU(), #https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            Block(channels),
            Block(channels),
            Block(channels),
            torch.nn.Conv2d(in_channels = channels, #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                            out_channels = 8,
                            kernel_size = 3,
                            padding = 1,
                            bias = False),
            torch.nn.ReLU()) #https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html

    def forward(self, hidden_state, action):
        return self.nn(torch.cat(tensors = (hidden_state, action), #https://pytorch.org/docs/stable/generated/torch.cat.html
                                 dim = 1))
