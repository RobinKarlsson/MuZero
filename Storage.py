#python 3.8

from Network import Network
from MuZeroConfig import MuZeroConfig

class SharedStorage(object):
    def __init__(self):
        self._networks = {}

    def latest_network(self, config: MuZeroConfig) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            return Network(config.board_gridsize, config.board_size, config.channels,
                           config.consider_backward_states, config.action_space_size)

    def save_network(self, step: int, network: Network):
        self._networks[step] = network
