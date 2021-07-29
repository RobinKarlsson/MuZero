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
            return Network(config).to(config.torch_device)

    def save_network(self, step: int, network: Network):
        self._networks[step] = network
