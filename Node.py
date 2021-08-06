#python 3.8

from numpy.random import dirichlet
from typing import List

from MuZeroConfig import MuZeroConfig
from Action import Action
from Player import Player

class Node(object):
    def __init__(self, prior: int, to_play: Player, hidden_state = None, discount: float = 1.):
        self.visit_count = 0
        self.prior = prior
        self.to_play = to_play
        self.value_sum = 0
        self.children = []
        self.hidden_state = hidden_state
        self.discount = discount
        self.reward = 0

    def expand(self, actions: List[Action], hidden_state, policy, value):
        moves = [[action, policy[0, action.index].item()] for action in actions]
        p_sum = policy.sum().item()
        next_player = Player(-self.to_play.player)
        
        for action, p in moves:
            subnode = Node(p / p_sum, next_player, hidden_state)
            if not subnode:
                raise(policy)
            self.children.append([action, subnode])

    def addNoise(self, config: MuZeroConfig):
        noise = dirichlet([config.root_dirichlet_alpha] * len(self.children))

        for i in range(len(self.children)):
            self.children[i][1].prior = self.children[i][1].prior * (
                1 - config.root_exploration_fraction) + noise[i] * config.root_exploration_fraction

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        return 0 if self.visit_count == 0 else self.value_sum / self.visit_count
