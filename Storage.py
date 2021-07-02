#python 3.8

from Network import Network

class SharedStorage(object):
    def __init__(self):
        self._networks = {}

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            return make_uniform_network()

    def save_network(self, step: int, network: Network):
        self._networks[step] = network
