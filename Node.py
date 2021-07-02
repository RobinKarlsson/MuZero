#python 3.8

from numpy.random import dirichlet
from typing import List

from MuZeroConfig import MuZeroConfig
from Action import Action
from Player import Player

class Node(object):
    def __init__(self, prior: int, to_play: Player):
        self.visit_count = 0
        self.prior = prior
        self.to_play = to_play
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expand(self, actions: List[Action]):
        for a in actions:
            self.children[a] = Node(0, Player(-self.to_play.player))

    def addNoise(self, config: MuZeroConfig):
        actions = list(self.children.keys())
        noise = dirichlet([config.root_dirichlet_alpha] * len(actions))

        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (
                1 - config.root_exploration_fraction) + n * config.root_exploration_fraction

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        return 0 if self.visit_count == 0 else self.value_sum / self.visit_count
